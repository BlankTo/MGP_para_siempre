import numpy as np

from mgp.experiments import campaign as C
from mgp.experiments import noise as noise_v2

STATE = {'clean': None, 'stats': None, 'observers': []}


def same_values(a, b):
    va, vb = getattr(a, 'values', None), getattr(b, 'values', None)
    return va is not None and vb is not None and np.allclose(va, vb, rtol=1e-9, atol=1e-12, equal_nan=True)


def mark_first(st, kind):
    key = 'first_' + kind
    if st[key] is None:
        st[key] = STATE['observers'][-1].step + 1


def install():
    from mgp.archives import observer as AO
    from mgp.engine import derivation as D, merge as M
    from mgp.engine.fitness import best_index as orig_best_index, object_fitness
    orig_choose = M.ExternalMerge.choose_original

    def choose_original(self, candidates, weights=None, force_best=False):
        selected = orig_choose(self, candidates, weights, force_best)
        st = STATE['stats']
        if st is not None and len(candidates) > 1:
            clean_choice, bf = None, float('inf')
            for c in candidates:
                f = object_fitness(c, STATE['clean'])
                if clean_choice is None or f < bf:
                    clean_choice, bf = c, f
            st['ic'] += 1
            if clean_choice is not selected:
                kind = 'ic_equiv' if same_values(clean_choice, selected) else 'ic_real'
                st[kind] += 1
                mark_first(st, kind)
        return selected

    def best_index(workspace, target):
        i = orig_best_index(workspace, target)
        st = STATE['stats']
        if st is not None:
            j = orig_best_index(workspace, STATE['clean'])
            st['elite'] += 1
            if i != j:
                kind = 'elite_equiv' if same_values(workspace[i], workspace[j]) else 'elite_real'
                st[kind] += 1
                mark_first(st, kind)
        return i

    class Recording(AO.ArchiveOnlyObserver):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.step_digests = []
            STATE['observers'].append(self)

        def observe(self, derivation):
            super().observe(derivation)
            self.step_digests.append(self.trajectory_digest.hexdigest())

    M.ExternalMerge.choose_original = choose_original
    D.best_index = best_index
    AO.ArchiveOnlyObserver = Recording


def first_divergence(reference, other):
    for index, (a, b) in enumerate(zip(reference, other)):
        if a != b:
            return index
    return None if len(reference) == len(other) else min(len(reference), len(other))


def derive(task, conditions):
    from mgp.experiments.jobs import run_job, dataset_cache_key
    from mgp.archives.codecs import ARCHIVE_FIELDS
    from mgp.engine.fitness import readonly_target, rmse

    def variant(level):
        t = dict(task)
        c = dict(task['case'])
        c['noise_level'] = level
        t['case'] = c
        return noise_v2.paired_dataset(t)

    job0, settings0, ds0 = variant(0.0)
    variables, clean, _, test, audit = ds0
    finite = np.isfinite(clean)
    scale = rmse(np.zeros(int(finite.sum())), clean[finite])
    labels1 = variant(1.0)[2][2]
    z = np.zeros_like(clean)
    z[finite] = (labels1[finite] - clean[finite]) / scale
    out, reference, reference_digests = {}, None, None
    for cond in conditions:
        if cond in (0.0, 0.01, 0.1, 1.0):
            job, job_settings, ds = variant(cond)
        else:
            labels = np.where(finite, (clean if cond == 10.0 else 0.0) + (10.0 if cond == 10.0 else 1.0) * scale * z, clean)
            job, job_settings = job0, settings0
            ds = (variables, clean, readonly_target(np.array(labels, dtype=float)), test, audit)
        stats = {k: 0 for k in ('ic', 'ic_real', 'ic_equiv', 'elite', 'elite_real', 'elite_equiv')}
        stats.update({'first_' + k: None for k in ('ic_real', 'ic_equiv', 'elite_real', 'elite_equiv')})
        STATE['clean'] = clean
        STATE['stats'] = None if cond == 0.0 else stats
        STATE['observers'].clear()
        C.clear_caches()
        [(_, m, _)] = run_job(task['options'], job, {dataset_cache_key(task['options'], job, job_settings): ds}, observation_mode='archive_only',
                              dataset_payload=False, archive_fields=ARCHIVE_FIELDS, identity_sink=lambda n, k: None)
        STATE['stats'] = None
        observer = STATE['observers'][-1]
        evaluated = set(observer.evaluated)
        if cond == 0.0:
            reference, reference_digests = evaluated, observer.step_digests
        union = len(evaluated | reference)
        out[cond] = dict(built=m['evaluated_target_ever'], symbolic=m['symbolic_recovery'], numerical=m['test_success'],
                         hash=m['transition_sequence_hash'], evaluated=len(evaluated), merges=m['merges'],
                         first_divergence=first_divergence(reference_digests, observer.step_digests),
                         jaccard=(len(evaluated & reference) / union) if union else 1.0,
                         contained=(len(evaluated & reference) / len(evaluated)) if evaluated else 1.0, **stats)
    c = task['case']
    return dict(target=task['target'], config_id=c['config_id'], rep=task['index'], esel=c['esel'], esur=c['esur'], strategy=c['merge_strategy'],
                im=c['internal_merge_probability'], size=c['total_objects'], res=out)
