import time

from mgp.experiments import campaign as C

ARCHIVES_ON = {'archive_enabled': True, 'building_block_archive_enabled': True, 'best_bb_archive_enabled': True,
               'observational_archive_mask': {'base_fitness': True, 'bb_snapshot': True, 'best_bb': True},
               'archive_similarity': 0.99, 'archive_comparison_method': 'cosine', 'research_wall_limit_seconds': None}
ARCHIVES_OFF = {'archive_enabled': False, 'building_block_archive_enabled': False, 'best_bb_archive_enabled': False,
                'observational_archive_mask': {'base_fitness': False, 'bb_snapshot': False, 'best_bb': False},
                'research_wall_limit_seconds': None}
_OPTIONS = {}


def setup(options):
    _OPTIONS.clear()
    _OPTIONS.update(options)


def measure(task, archives_on, diversity):
    from mgp.experiments.jobs import job_settings, make_job_dataset
    from mgp.experiments.plan import paired_seed
    from mgp.engine.derivation import Derivation
    from mgp.experiments.plan import job_for
    from mgp.engine.problem import make_workspace
    case, rep = task
    job = job_for(case, rep)
    archives = dict(ARCHIVES_ON, base_archive_policies=['symbolic', 'similarity'] if diversity else ['symbolic'])
    task_settings = {**job_settings(_OPTIONS, job), **(archives if archives_on else ARCHIVES_OFF)}
    variables, clean, labels, test, _ = make_job_dataset(_OPTIONS, job, task_settings)
    workspace_seed = paired_seed(_OPTIONS, job, 'workspace', rep)
    derivation_seed = paired_seed(_OPTIONS, job, 'derivation', rep)
    C.clear_caches()
    try:
        wall0, cpu0 = time.perf_counter(), time.process_time()
        workspace = make_workspace(variables, task_settings, workspace_seed)
        d = Derivation(labels, task_settings, derivation_seed, clean_target=clean)
        d.run(workspace)
        c1, w1 = time.process_time(), time.perf_counter()
        if d.building_block_archive is not None:
            d.building_block_archive.finalize_deferred()
        credit, wcredit = time.process_time() - c1, time.perf_counter() - w1
        cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0
        ba = d.base_archive_cpu_seconds.get('symbolic', 0.0)
        div = d.base_archive_cpu_seconds.get('similarity', 0.0)
        rec, fbba = d.archive_observation_cpu_seconds, d.best_bb_cpu_seconds
        return (case['case_id'], rep, 'ok', wall, cpu, cpu - ba - div - rec - credit - fbba, ba, div, rec, credit, fbba,
                d.base_archive_seconds.get('symbolic', 0.0), d.base_archive_seconds.get('similarity', 0.0), d.archive_observation_seconds,
                wcredit, d.best_bb_seconds, d.termination_reason, d.merge_count, d.external_merge.operator_evaluations,
                d.external_merge.replay_operator_evaluations, d.best_fitness, None)
    except Exception as exc:
        return (case['case_id'], rep, 'error') + (None,) * 18 + (f'{type(exc).__name__}: {exc}'[:300],)
