import json
import random

from mgp.engine.protocol import stream_seed, noisy_labels
from mgp.experiments import noise as noise_v2
from mgp.experiments.jobs import dataset_cache_key
from mgp.experiments.plan import job_for
from mgp.experiments.plan import signature

FIELDS = ('test_success', 'symbolic_recovery', 'test_rmse', 'test_nrmse', 'train_success', 'train_rmse', 'clean_rmse', 'selected_key',
          'selected_structure_digest', 'expression_operators', 'expression_leaves', 'expression_depth', 'transition_sequence_hash',
          'evaluated_target_ever', 'retained_target_ever', 'merges', 'termination', 'censored', 'ic_decisions', 'ms_decisions',
          'internal_merge_count', 'external_merge_count', 'evaluated_unique', 'retained_unique', 'subtarget_first_merge', 'workspace_seed',
          'derivation_seed', 'noise_seed', 'dataset_id', 'archive_component_coverage', 'archive_cooccurrence_coverage',
          'archive_composite_coverage', 'first_retained_target_merge', 'noise_level', 'test_invalid')


def tasks(study, cases, opts, repetitions, pools, coverage, first=1):
    from mgp.experiments.campaign import digest
    for repetition in range(first, repetitions + 1):
        ordered = list(cases)
        random.Random(stream_seed(opts['master_seed'], study, 'balanced_run_order', repetition)).shuffle(ordered)
        for c in ordered:
            for pool in range(1, pools + 1):
                restart = 100000 + (pool - 1) * 10000 + repetition
                task_settings = dict(opts['settings'], sample_count=c['experimental_training_size'])
                if coverage:
                    task_settings['sample_count'] = opts['settings']['sample_count']
                yield {'run_id': digest([study, c['case_id'], pool, repetition])[:32], 'study': study, 'case': c, 'pool': pool, 'index': repetition,
                       'restart_index': restart, 'options': dict(opts, settings=task_settings), 'target': c['target'],
                       'dataset_replica': c['dataset_replica'], 'noise_level': c['noise_level'], 'training': not coverage}


def nested_training_dataset(task):
    import numpy as np
    from mgp.experiments.jobs import job_settings
    from mgp.engine.problem import random_dataset
    from mgp.archives.diagnostics import noise_audit
    from mgp.engine.fitness import readonly_target
    from mgp.engine.targets import TARGETS as FUNCTION_CATALOG
    job = job_for(task['case'], task['restart_index'])
    config = job_settings(task['options'], job)
    samples = task['case']['experimental_training_size']
    full = dict(config, sample_count=100)
    variables100, clean100 = random_dataset(FUNCTION_CATALOG[task['target']][0], full)
    noise_master = noise_v2.replica_seed(task)
    labels100, seed = noisy_labels(clean100, {**full, 'noise_seed': noise_master, 'fixed_noise_across_repetitions': True}, task['target'], 0)
    variables = {k: readonly_target(np.asarray(v[:samples], dtype=float).copy()) for k, v in variables100.items()}
    clean = readonly_target(np.asarray(clean100[:samples], dtype=float).copy())
    labels = readonly_target(np.asarray(labels100[:samples], dtype=float).copy())
    test = random_dataset(FUNCTION_CATALOG[task['target']][0], config, 'test')
    audit = noise_audit(variables, clean, labels, test, config, seed)
    ref_rms = float(np.sqrt(np.mean(np.square(np.asarray(clean100)[np.isfinite(clean100)]))))
    noise_rms = float(audit['noise_rms'])
    actual_sigma = float(config['noise_level']) * ref_rms
    audit['subset_signal_rms'] = audit['signal_rms']
    audit['reference_signal_rms100'] = ref_rms
    audit['expected_sigma'] = actual_sigma
    audit['noise_to_reference_signal_rms'] = noise_rms / ref_rms if ref_rms else None
    audit['realized_std_over_sigma'] = audit['noise_std'] / actual_sigma if actual_sigma else None
    audit['sigma_reference'] = 'clean_training_100'
    audit.update(noise_protocol=noise_v2.POLICY, config_id=task['case']['config_id'], restart_index=task['restart_index'], noise_master=noise_master,
                 sample_count=samples, reference_noise_RMS100=ref_rms, nested_sampling=True,
                 pairing_key=signature([task['target'], task['case']['config_id'], task['case']['dataset_replica'], task['restart_index']])[:24])
    return job, config, (variables, clean, labels, test, audit)


def execute(task):
    from mgp.experiments.jobs import run_job
    from mgp.archives.codecs import pack_identity_sets
    training = task['training']
    try:
        if training:
            job, task_settings, ds = nested_training_dataset(task)
            cache = {dataset_cache_key(task['options'], job, task_settings): ds}
        else:
            job = job_for(task['case'], task['restart_index'])
            cache, ds = {}, None
        identity = {}
        [(idx, metrics, detail)] = run_job(task['options'], job, cache, observation_mode='archive_only', archive_sink=None, archive_fields=(),
                                           dataset_payload=False, identity_sink=lambda name, keys: identity.update({name: keys}))
        if training and metrics['noise_seed'] != ds[4]['noise_seed']:
            raise RuntimeError('noise seed differs between generated labels and saved metrics')
        small = {k: metrics.get(k) for k in FIELDS}
        small.update(dict.fromkeys(('ic_choices', 'ic_changed', 'ic_equivalent', 'elite_choices', 'elite_changed', 'elite_equivalent'), 0))
        identity = {name: identity.get(name, set()) for name in ('base_fitness', 'bb_snapshot', 'best_bb')}
        small['archive_counts'] = {name: len(keys) for name, keys in identity.items()}
        small['actual_training_size'] = int(task['options']['settings']['sample_count'])
        small['run_id'] = task['run_id']
        small['noise_audit'] = ds[4] if ds else None
        return (task['run_id'], task['case']['case_id'], task['pool'], task['index'], 'ok', json.dumps(small, default=str, allow_nan=False),
                pack_identity_sets(identity), None, None)
    except Exception as exc:
        return task['run_id'], task['case']['case_id'], task['pool'], task['index'], 'error', None, None, None, f'{type(exc).__name__}: {exc}'[:1000]
