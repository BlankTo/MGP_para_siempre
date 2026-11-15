import json

import numpy as np

from mgp.experiments import campaign as C

ARCH = ('base_fitness', 'bb_snapshot', 'best_bb')
MODE = [None]
OBSERVERS = []


def install():
    from mgp.archives import archive as A, observer as AO, building_blocks as B
    from mgp.engine import derivation as D, merge as M
    from mgp.engine.fitness import object_fitness
    from mgp.engine.objects import SemanticObject

    def negated(current, target):
        f = object_fitness(current, target)
        return -f if MODE[0] else f

    def negated_return(current, target):
        f = object_fitness(current, target)
        return -f if MODE[0] == 'inverted_all' else f

    def best_index(workspace, target):
        selected_index, selected = None, float('inf')
        for index, current in enumerate(workspace):
            if isinstance(current, SemanticObject):
                f = negated(current, target)
                if selected_index is None or f < selected:
                    selected_index, selected = index, f
        return selected_index

    original_quality = B.BuildingBlockArchive.quality

    def quality(self, current):
        if not MODE[0]:
            return original_quality(self, current)
        error = object_fitness(current, self.target)
        if error == 0:
            return 0.0
        if not np.isfinite(error):
            return 1.0
        return 1.0 / (1.0 + self.scale / error)

    class Recording(AO.ArchiveOnlyObserver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            OBSERVERS.append(self)

    M.object_fitness = negated
    D.best_index = best_index
    A.object_fitness = negated
    B.BuildingBlockArchive.quality = quality
    D.object_fitness = negated_return
    AO.ArchiveOnlyObserver = Recording


def pure_noise_labels(task, case, clean):
    from mgp.engine.fitness import readonly_target, rmse
    from mgp.engine.protocol import stream_seed
    finite = np.isfinite(clean)
    scale = rmse(np.zeros(int(finite.sum())), clean[finite])
    seed = stream_seed(task['options']['master_seed'], case['target'], f"fitness_control|{case['config_id']}", task['restart_index'])
    z = np.random.default_rng(seed).standard_normal(int(finite.sum()))
    noisy = np.array(clean, dtype=float)
    noisy[finite] = scale * z
    return readonly_target(noisy)


def derive(task, conditions):
    from mgp.engine.targets import catalog
    from mgp.archives.codecs import ARCHIVE_FIELDS
    from mgp.experiments.jobs import dataset_cache_key, job_settings, make_job_dataset, run_job
    from mgp.experiments.plan import job_for
    digests = catalog(task['case']['target'])[0]
    out, reference = {}, None
    for condition in ('clean',) + tuple(c for c in conditions if c != 'clean'):
        case = dict(task['case'])
        if condition == 'random_ic':
            case['fitness_probability'] = 0.0
        job = job_for(case, task['restart_index'])
        job_settings_ = job_settings(task['options'], job)
        variables, clean, labels, test, audit = make_job_dataset(task['options'], job, job_settings_)
        if condition == 'pure_noise':
            labels = pure_noise_labels(task, case, clean)
        sink = {}
        OBSERVERS.clear()
        C.clear_caches()
        MODE[0] = condition if condition.startswith('inverted') else None
        try:
            [(_, m, _)] = run_job(task['options'], job, {dataset_cache_key(task['options'], job, job_settings_): (variables, clean, labels, test, audit)},
                                  observation_mode='archive_only', dataset_payload=False, archive_fields=ARCHIVE_FIELDS,
                                  identity_sink=lambda name, keys: sink.__setitem__(name, set(keys)))
        finally:
            MODE[0] = None
        evaluated = set(OBSERVERS[-1].evaluated)
        if condition == 'clean':
            reference = evaluated
        union = len(evaluated | reference)
        row = {a: sum(1 << i for d, i in digests.items() if d in sink.get(a, ())) for a in ARCH}
        row.update({a + '_entries': len(sink.get(a, ())) for a in ARCH})
        row['selected'] = next((i for d, i in digests.items() if d in sink.get('selected_best', set())), -1)
        row.update(built=m['evaluated_target_ever'], retained=m['retained_target_ever'], numerical=m['test_success'],
                   symbolic=m['symbolic_recovery'], merges=m['merges'], termination=m['termination'], ic_decisions=m['ic_decisions'],
                   evaluated_unique=len(evaluated), jaccard=len(evaluated & reference) / union if union else 1.0,
                   train_rmse=m.get('train_rmse'), hash=m.get('transition_sequence_hash'),
                   base_fitness_composite=m.get('base_fitness_composite_coverage'), bb_snapshot_composite=m.get('bb_snapshot_composite_coverage'))
        out[condition] = row
    return json.loads(json.dumps(out, default=float))
