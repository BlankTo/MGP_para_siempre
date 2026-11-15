import hashlib
import math
import random
import time

from mgp.archives.codecs import packed
from mgp.archives.diagnostics import noise_audit
from mgp.archives.observer import DerivationObserver, selected_measurement, construction_summary
from mgp.archives.records import pack_dataset
from mgp.archives.reuse_memory import ReuseMemory, policy_config
from mgp.engine.derivation import Derivation
from mgp.engine.fitness import object_fitness
from mgp.engine.problem import make_workspace, random_dataset
from mgp.engine.protocol import noisy_labels, stream_seed
from mgp.engine.symbolic import EFFORT as SYMBOLIC_EFFORT, symbolic_data
from mgp.engine.targets import TARGETS
from mgp.experiments.plan import FACTORS, SINGLE_STAGES, paired_seed, signature

SCHEMA_VERSION = 2


def job_settings(options, job):
    settings = {**options['settings'], **{k: job[k] for k in FACTORS}, **job.get('settings_overrides', {}), 'target': job['target'], 'dataset_seed': job['dataset_seed'], 'noise_seed': job['dataset_seed'], 'noise_level': job['noise_level'], 'test_seed': stream_seed(job['dataset_seed'], 'all', 'v35_test'), 'count_archive_diagnostics': options['version'] >= 36 and (job['stage'] in SINGLE_STAGES or options['version'] >= 38), 'fixed_noise_across_repetitions': True}
    if options['version'] in (42, 43, 44):
        settings.update(noise_seed=stream_seed(options['master_seed'], job['target'], 'v42_noise', job['dataset_replica']), test_seed=stream_seed(options['master_seed'], job['target'], 'v42_test_' + options['test_stream']))
    return policy_config(settings, job['policy'])


def dataset_cache_key(options, job, settings=None):
    settings = job_settings(options, job) if settings is None else settings
    cache_key = (job['dataset_seed'], job['target'], job['noise_level'])
    if options['version'] in (42, 43, 44):
        cache_key += (settings['noise_seed'], settings['test_seed'], settings['dataset_sampling'], settings['sample_count'], settings['test_sample_count'], settings['sample_min'], settings['sample_max'])
    return cache_key


def make_job_dataset(options, job, settings=None):
    settings = job_settings(options, job) if settings is None else settings
    variables, clean = random_dataset(TARGETS[job['target']][0], settings)
    labels, noise_seed = noisy_labels(clean, settings, job['target'], 0)
    test = random_dataset(TARGETS[job['target']][0], settings, 'test')
    audit = noise_audit(variables, clean, labels, test, settings, noise_seed)
    if options['version'] in (42, 43, 44):
        audit.update(dataset_sampling=settings['dataset_sampling'], test_seed=settings['test_seed'], test_samples=settings['test_sample_count'], dataset_replica=job['dataset_replica'])
    return (variables, clean, labels, test, audit)


ARCHIVE_FIELDS = ('archive', 'key', 'matched_id', 'train_rmse', 'clean_rmse')


def inherited_digest(key):
    return hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]


def run_job(options, job, dataset_cache=None, dataset_sink=None, archive_sink=None, progress_callback=None, lexicon=None, archive_fields=ARCHIVE_FIELDS, additions=None, observation_mode='full', dataset_payload=True, identity_sink=None):
    if observation_mode not in ('full', 'archive_only'):
        raise ValueError('unknown observation mode')
    settings = job_settings(options, job)
    if observation_mode != 'archive_only' and settings.get('defer_building_block_credit', False):
        settings = {**settings, 'defer_building_block_credit': False}
    cache = {} if dataset_cache is None else dataset_cache
    cache_key = dataset_cache_key(options, job, settings)
    fresh_dataset = cache_key not in cache
    if fresh_dataset:
        cache[cache_key] = make_job_dataset(options, job, settings)
    variables, clean, labels, test, audit = cache[cache_key]
    if dataset_sink is not None and (fresh_dataset or not dataset_payload):
        dataset_sink(signature(audit), audit, pack_dataset(variables, clean, labels, test) if dataset_payload else None)
    memory = None if job['policy'] == 'observe' else ReuseMemory(job['policy'], job['capacity'], job['shortlist'], job['injected'])
    results, ever_memory = ([], set())
    best, best_measure = (None, None)
    cumulative_merges, cumulative_evaluations = (0, 0)
    any_symbolic, any_numeric = (False, False)
    cumulative_seconds = 0.0
    for index in range(1, job['derivations'] + 1):
        started = time.perf_counter()
        cpu_started = time.process_time()
        seed_index = job.get('restart_index', index)
        workspace_seed = paired_seed(options, job, 'workspace', seed_index)
        derivation_seed = paired_seed(options, job, 'derivation', seed_index)
        effort_before = dict(SYMBOLIC_EFFORT)
        workspace_started = time.perf_counter()
        workspace = make_workspace(variables, settings, workspace_seed, lexicon, additions)
        workspace_seconds = time.perf_counter() - workspace_started
        sampled = [symbolic_data(node)[0] for node in workspace if getattr(node, 'archive_origin', False)]
        injection = memory.inject(workspace, random.Random(paired_seed(options, job, 'injection', seed_index))) if memory is not None else []
        setup_seconds = time.perf_counter() - started
        setup_cpu_seconds = time.process_time() - cpu_started
        if observation_mode == 'archive_only':
            from mgp.archives.observer import ArchiveOnlyObserver
            observer = ArchiveOnlyObserver(settings, variables, labels, clean)
        else:
            observer = DerivationObserver(settings, variables, labels, clean)
        observer.include_similarity_archive = 'credit_by_depth' in archive_fields
        callback_seconds = 0.0
        callback_cpu_seconds = 0.0
        observer_seconds = 0.0
        observer_cpu_seconds = 0.0
        budget_checkpoints = []

        def timed_candidate(node):
            nonlocal callback_seconds, callback_cpu_seconds
            tick = time.perf_counter()
            cpu_tick = time.process_time()
            observer.on_candidate(node)
            callback_seconds += time.perf_counter() - tick
            callback_cpu_seconds += time.process_time() - cpu_tick

        def timed_choice(candidates, weights, selected):
            nonlocal callback_seconds, callback_cpu_seconds
            tick = time.perf_counter()
            cpu_tick = time.process_time()
            observer.on_choice(candidates, weights, selected)
            callback_seconds += time.perf_counter() - tick
            callback_cpu_seconds += time.process_time() - cpu_tick
        tick = time.perf_counter()
        cpu_tick = time.process_time()
        derivation = Derivation(labels, settings, derivation_seed, clean, candidate_observer=timed_candidate, choice_observer=timed_choice)
        online_seconds = setup_seconds + time.perf_counter() - tick
        online_cpu_seconds = setup_cpu_seconds + time.process_time() - cpu_tick
        iterator = iter(derivation.steps(workspace))
        while True:
            tick = time.perf_counter()
            cpu_tick = time.process_time()
            try:
                next(iterator)
            except StopIteration:
                online_seconds += time.perf_counter() - tick
                online_cpu_seconds += time.process_time() - cpu_tick
                break
            online_seconds += time.perf_counter() - tick
            online_cpu_seconds += time.process_time() - cpu_tick
            observer_tick = time.perf_counter()
            observer_cpu_tick = time.process_time()
            observer.observe(derivation)
            step = derivation.merge_count
            if observation_mode == 'archive_only' and (step == 0 or step & step - 1 == 0 or step % 250 == 0):
                point = observer.trace[-1]
                budget_checkpoints.append({'merge': step, 'observed_wall_seconds': time.perf_counter() - started, 'observed_cpu_seconds': time.process_time() - cpu_started, 'operator_evaluations': derivation.external_merge.operator_evaluations, 'fitness_evaluations': derivation.external_merge.fitness_evaluations, 'internal_candidates_inspected': derivation.internal_candidates_inspected, 'ic_decisions': derivation.external_merge.ic_decisions, 'ms_decisions': derivation.external_merge.ms_decisions, 'replay_operator_evaluations': derivation.external_merge.replay_operator_evaluations, 'retained_unique': point.get('retained_unique'), 'evaluated_unique': point.get('evaluated_unique'), 'retained_composite_coverage': point.get('retained_composite_coverage'), 'archive_composite_coverage': point.get('archive_composite_coverage'), 'best_train_rmse': derivation.best_fitness, 'best_clean_rmse': point.get('selected_clean_rmse'), 'best_symbolic_recovery': point.get('selected_symbolic_recovery'), 'evaluated_target_ever': point.get('evaluated_target_ever'), 'retained_target_ever': point.get('retained_target_ever'), **{name: point.get(name) for name in ('workspace_target', 'workspace_component_coverage', 'workspace_composite_coverage', 'workspace_cooccurrence_coverage', 'retained_component_coverage', 'retained_cooccurrence_coverage', 'archive_component_coverage', 'archive_cooccurrence_coverage', 'retained_components_present', 'retained_composites_present', 'retained_component_sets_present', 'archive_components_present', 'archive_composites_present', 'archive_component_sets_present')}})
            observer_seconds += time.perf_counter() - observer_tick
            observer_cpu_seconds += time.process_time() - observer_cpu_tick
            if progress_callback is not None:
                progress_callback()
        if settings.get('single_archive_max_size') is not None and derivation.building_block_archive is not None:
            derivation.building_block_archive.finalize_deferred()
            derivation.building_block_archive.limit(settings['single_archive_max_size'])
        if observation_mode == 'archive_only':
            observer.finalize_coverage(derivation)
        credit_finalize_started = time.perf_counter()
        credit_finalize_cpu_started = time.process_time()
        if derivation.building_block_archive is not None:
            derivation.building_block_archive.finalize_deferred()
            if derivation.metric_steps:
                derivation.metric_steps[-1]['building_block_score'] = derivation.building_block_archive.total_score
            if observation_mode == 'archive_only' and observer.trace:
                observer.trace[-1]['building_block_score'] = derivation.building_block_archive.total_score
        credit_finalize_seconds = time.perf_counter() - credit_finalize_started
        credit_finalize_cpu_seconds = time.process_time() - credit_finalize_cpu_started
        derivation.archive_observation_seconds += credit_finalize_seconds
        derivation.archive_observation_cpu_seconds += credit_finalize_cpu_seconds
        observation_seconds = callback_seconds + derivation.archive_observation_seconds - credit_finalize_seconds + sum(derivation.base_archive_seconds.values()) + derivation.best_bb_seconds + derivation.metrics_observation_seconds + derivation.symbolic_observation_seconds
        search_seconds = max(0.0, online_seconds - observation_seconds)
        observation_cpu_seconds = callback_cpu_seconds + derivation.archive_observation_cpu_seconds - credit_finalize_cpu_seconds + sum(derivation.base_archive_cpu_seconds.values()) + derivation.best_bb_cpu_seconds + derivation.metrics_observation_cpu_seconds + derivation.symbolic_observation_cpu_seconds
        search_cpu_seconds = max(0.0, online_cpu_seconds - observation_cpu_seconds)
        selected = selected_measurement(derivation.best, labels, clean, test, settings)
        improved = derivation.best is not None and (best is None or derivation.best_fitness < object_fitness(best, labels))
        if improved:
            best, best_measure = (derivation.best, selected)
        updates = memory.update(derivation, index) if memory is not None else {'admitted': [], 'updated': [], 'evicted': []}
        memory_entries = memory.entries if memory is not None else {}
        ever_memory.update(memory_entries)
        cumulative_merges += derivation.merge_count
        cumulative_evaluations += derivation.external_merge.operator_evaluations
        elapsed = time.perf_counter() - started
        cumulative_seconds += elapsed
        any_symbolic = any_symbolic or selected['symbolic_recovery'] == 1
        any_numeric = any_numeric or selected['test_success'] == 1
        last = observer.trace[-1]
        metrics = {**job, 'derivation_index': index, 'workspace_seed': workspace_seed, 'dataset_id': signature(audit), 'noise_seed': audit['noise_seed'], 'injection_seed': paired_seed(options, job, 'injection', seed_index), 'derivation_seed': derivation_seed, **selected, 'external_merge_count': derivation.external_merge_count, 'internal_merge_count': derivation.internal_merge_count, 'internal_fallback_count': derivation.internal_fallback_count, 'internal_attempts': derivation.internal_merge_count + derivation.internal_fallback_count, 'realized_im_fraction': derivation.internal_merge_count / derivation.merge_count if derivation.merge_count else None, 'esur_in_IM': settings['esur_in_IM'], 'IM_terminate_on_one': settings['IM_terminate_on_one'], 'merges': derivation.merge_count, 'termination': derivation.termination_reason, 'censored': int(derivation.termination_reason in ('max_merge', 'alternative_limit', 'time_limit')), 'alternative_limit': int(derivation.termination_reason == 'alternative_limit'), 'alternative_limit_reason': derivation.alternative_limit_reason, 'replay_pairs': derivation.external_merge.replay_pairs, 'replay_operator_evaluations': derivation.external_merge.replay_operator_evaluations, 'replay_truncated_merges': derivation.external_merge.replay_truncated_merges, 'replay_limit_hits': dict(derivation.external_merge.replay_limit_hits), 'replay_duplicates': derivation.external_merge.replay_duplicates, 'replay_expanded_merges': derivation.external_merge.replay_expanded_merges, 'uncommitted_evaluations': len(observer.pending), 'operator_evaluations': derivation.external_merge.operator_evaluations, 'ic_candidates': derivation.external_merge.ic_candidates, 'ic_decisions': derivation.external_merge.ic_decisions, 'ms_decisions': derivation.external_merge.ms_decisions, 'improvements': observer.improvements, 'last_improvement_merge': observer.last_improvement, 'stagnation_tail': derivation.merge_count - observer.last_improvement, 'evaluated_unique': len(observer.evaluated), 'retained_unique': len(observer.retained), 'evaluated_not_retained_unique': len(observer.evaluated - observer.retained), 'evaluated_target_ever': int(observer.target_key in observer.evaluated), 'retained_target_ever': int(observer.target_key in observer.retained), 'actual_injected': len(injection), 'imports_used': len(observer.used_imports), 'imports_in_best': len(observer.best_imports), 'episode_improved': int(improved), 'memory_size': len(memory_entries), 'memory_ever_retained': len(ever_memory), 'no_legal_merge': int(derivation.termination_reason == 'no_legal_merge'), 'evictions': len(updates['evicted']), 'cumulative_evictions': memory.evictions if memory is not None else 0, 'cumulative_merges': cumulative_merges, 'cumulative_evaluations': cumulative_evaluations, 'search_seconds': search_seconds, 'online_seconds': online_seconds, 'observation_seconds': observation_seconds, 'external_observer_seconds': observer_seconds, 'external_observer_cpu_seconds': observer_cpu_seconds, 'measured_seconds': elapsed, 'cumulative_seconds': cumulative_seconds, 'episode_any_symbolic_solution': int(any_symbolic), 'episode_any_numeric_solution': int(any_numeric), **{'memory_' + k: v for k, v in observer.coverage(set(memory_entries)).items()}, **{'ever_memory_' + k: v for k, v in observer.coverage(ever_memory).items()}, **{k: last[k] for k in last if k.startswith(('workspace_', 'retained_', 'archive_'))}, **{'episode_' + k: v for k, v in (best_measure or selected).items()}}
        metrics['first_retained_target_merge'] = observer.first['retained'].get(observer.expected.get(observer.target_key))
        if observation_mode == 'archive_only':
            metrics['subtarget_first_merge'] = {kind: dict(when) for kind, when in observer.first.items()}
        bb = derivation.building_block_archive
        metrics.update(bb_credit_finalize_seconds=credit_finalize_seconds, bb_credit_finalize_cpu_seconds=credit_finalize_cpu_seconds, workspace_seconds=workspace_seconds, attempted_merges=derivation.attempt_count, fitness_evaluations=derivation.external_merge.fitness_evaluations, internal_candidates_inspected=derivation.internal_candidates_inspected, base_archive_insertions=derivation.base_archive_insertions, base_archive_rejections=derivation.base_archive_rejections, base_archive_replacements=getattr(derivation.archive, 'replacements', None), bb_archive_entries=len(getattr(bb, 'entries', ())) if bb is not None else None, symbolic_identities=SYMBOLIC_EFFORT['identities'] - effort_before['identities'], symbolic_cancel_calls=SYMBOLIC_EFFORT['cancel'] - effort_before['cancel'], symbolic_simplify_calls=SYMBOLIC_EFFORT['simplify'] - effort_before['simplify'], unfactored_domain_guards=SYMBOLIC_EFFORT['unfactored_domain_guards'] - effort_before['unfactored_domain_guards'], archive_seconds=derivation.archive_observation_seconds + sum(derivation.base_archive_seconds.values()) + derivation.best_bb_seconds, symbolic_seconds=derivation.symbolic_observation_seconds, archive_cpu_seconds=derivation.archive_observation_cpu_seconds + sum(derivation.base_archive_cpu_seconds.values()) + derivation.best_bb_cpu_seconds, symbolic_cpu_seconds=derivation.symbolic_observation_cpu_seconds, bb_snapshot_seconds=derivation.archive_observation_seconds-credit_finalize_seconds, bb_snapshot_cpu_seconds=derivation.archive_observation_cpu_seconds-credit_finalize_cpu_seconds, base_archive_seconds_by_policy=dict(derivation.base_archive_seconds), base_archive_cpu_seconds_by_policy=dict(derivation.base_archive_cpu_seconds), best_bb_seconds=derivation.best_bb_seconds, best_bb_cpu_seconds=derivation.best_bb_cpu_seconds, metrics_observation_seconds=derivation.metrics_observation_seconds, metrics_observation_cpu_seconds=derivation.metrics_observation_cpu_seconds, callback_seconds=callback_seconds, callback_cpu_seconds=callback_cpu_seconds, archive_mask=dict(derivation.archive_mask))
        if additions is not None:
            metrics.update(workspace_objects=len(workspace), workspace_seconds=workspace_seconds, added_objects=len(workspace) - settings['variable_atom_count'] - settings['operator_count'], inherited_merged=sorted((inherited_digest(k) for k in observer.merged_imports)), retained_with_inherited=observer.retained_with_imports, **construction_summary(derivation.best))
        if lexicon is not None or additions is not None:
            metrics.update(lexicon_size=len(lexicon) if lexicon is not None else None, inherited_occurrences=len(sampled), inherited_sampled=sorted({inherited_digest(k) for k in sampled}), inherited_used=sorted((inherited_digest(k) for k in observer.used_imports)), inherited_in_best=sorted((inherited_digest(k) for k in observer.best_imports)), best_is_inherited=int(bool(getattr(derivation.best, 'archive_origin', False))))
        if observation_mode == 'archive_only':
            metrics['transition_sequence_hash'] = observer.trajectory_digest.hexdigest()
            step = derivation.merge_count
            if not budget_checkpoints or budget_checkpoints[-1]['merge'] != step:
                point = observer.trace[-1]
                budget_checkpoints.append({'merge': step, 'observed_wall_seconds': time.perf_counter() - started, 'observed_cpu_seconds': time.process_time() - cpu_started, 'operator_evaluations': derivation.external_merge.operator_evaluations, 'fitness_evaluations': derivation.external_merge.fitness_evaluations, 'internal_candidates_inspected': derivation.internal_candidates_inspected, 'ic_decisions': derivation.external_merge.ic_decisions, 'ms_decisions': derivation.external_merge.ms_decisions, 'replay_operator_evaluations': derivation.external_merge.replay_operator_evaluations, 'retained_unique': point.get('retained_unique'), 'evaluated_unique': point.get('evaluated_unique'), 'retained_composite_coverage': point.get('retained_composite_coverage'), 'archive_composite_coverage': point.get('archive_composite_coverage'), 'best_train_rmse': derivation.best_fitness, 'best_clean_rmse': point.get('selected_clean_rmse'), 'best_symbolic_recovery': point.get('selected_symbolic_recovery'), 'evaluated_target_ever': point.get('evaluated_target_ever'), 'retained_target_ever': point.get('retained_target_ever'), **{name: point.get(name) for name in ('workspace_target', 'workspace_component_coverage', 'workspace_composite_coverage', 'workspace_cooccurrence_coverage', 'retained_component_coverage', 'retained_cooccurrence_coverage', 'archive_component_coverage', 'archive_cooccurrence_coverage', 'retained_components_present', 'retained_composites_present', 'retained_component_sets_present', 'archive_components_present', 'archive_composites_present', 'archive_component_sets_present')}})
            metrics['budget_checkpoints'] = budget_checkpoints
        else:
            metrics['trajectory_hash'] = signature([[row.get('left'), row.get('right'), row.get('result')] for row in observer.trace])
        archive_rows = observer.archive_rows(derivation, detail_limit=settings.get('campaign_archive_detail_limit') if observation_mode == 'archive_only' else None, aggregate_only=observation_mode == 'archive_only' and archive_sink is None, identity_sink=identity_sink)
        if archive_sink is not None:
            archive_sink(index, [{key: row.get(key) for key in archive_fields} for row in archive_rows])
        metrics['train_success'] = int(derivation.best_fitness <= settings['perfect_fit_tolerance'])
        for row in observer.archive_metrics:
            prefix = row['archive'] if row['top'] == 'all' else row['archive'] + '_top20'
            metrics.update({prefix + '_' + key: value for key, value in row.items() if key not in ('archive', 'top')})
        if observation_mode == 'archive_only':
            measured = time.perf_counter() - started
            cumulative_seconds += measured - elapsed
            metrics.update(measured_seconds=measured, cumulative_seconds=cumulative_seconds, search_cpu_seconds=search_cpu_seconds, online_cpu_seconds=online_cpu_seconds, observation_cpu_seconds=observation_cpu_seconds, detail_export_seconds=0.0, detail_serialization_seconds=0.0, detail_bytes=0, recorded_nodes=0, recorded_evaluations=0, measured_cpu_seconds=time.process_time() - cpu_started)
            results.append((index, json_safe(metrics), None))
            continue

        def memory_row(key, entry):
            return {'key': key, **{k: v for k, v in entry.items() if k != 'node'}, 'node_ref': observer.records.ref(entry['node'])}
        upserts = [memory_row(key, memory_entries[key]) for key in sorted(set(updates['admitted'] + updates['updated'])) if key in memory_entries]
        evicted = [memory_row(key, entry) for key, entry in (memory.last_evicted.items() if memory is not None else [])]
        final_memory = [memory_row(key, memory_entries[key]) for key in (memory.ranked() if memory is not None else [])] if index == job['derivations'] else None
        for event in injection:
            event['node_ref'] = observer.initial_refs[event['position']]
        export_started = time.perf_counter()
        graph = observer.records.export()
        export_seconds = time.perf_counter() - export_started
        details = {'trace': observer.trace, 'events': observer.events, 'format_version': SCHEMA_VERSION, 'settings': settings, 'dataset_id': signature(audit), 'initial_workspace': observer.initial_workspace, 'initial_workspace_refs': observer.initial_refs, 'workspace_checkpoints': observer.workspace_checkpoints, 'final_workspace_refs': [observer.records.ref(n) for n in derivation.workspace], 'best_ref': observer.records.ref(derivation.best), 'best_evaluation': derivation.best.evaluation if derivation.best is not None else None, 'candidates': list(observer.candidates.values()), 'first_subtargets': observer.first, 'candidate_events': observer.candidate_events, 'choices': observer.choices, 'archives': archive_rows, 'archive_metrics': observer.archive_metrics, 'noise_audit': audit, 'injection': injection, 'memory_updates': updates, 'memory_top20': memory.snapshot() if memory is not None else [], 'memory_upserts': upserts, 'memory_evicted': evicted, 'memory_final': final_memory, **graph, 'transitions': dict(observer.transitions)}
        serialization_started = time.perf_counter()
        blob = packed(details)
        serialization_seconds = time.perf_counter() - serialization_started
        measured = time.perf_counter() - started
        cumulative_seconds += measured - elapsed
        metrics.update(measured_seconds=measured, cumulative_seconds=cumulative_seconds, search_cpu_seconds=search_cpu_seconds, online_cpu_seconds=online_cpu_seconds, observation_cpu_seconds=observation_cpu_seconds, detail_export_seconds=export_seconds, detail_serialization_seconds=serialization_seconds, detail_bytes=len(blob), recorded_nodes=len(graph['nodes']), recorded_evaluations=len(graph['evaluations']), measured_cpu_seconds=time.process_time() - cpu_started)
        results.append((index, json_safe(metrics), blob))
    return results


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value
