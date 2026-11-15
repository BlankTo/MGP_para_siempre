import numpy as np

COLS = ['atomic_semantic', 'composite_semantic', 'atomic_operator', 'composite_operator', 'incomplete']


def count(task, base, target_name):
    from mgp.engine.targets import TARGETS
    from mgp.engine.derivation import Derivation
    from mgp.experiments.plan import object_counts
    from mgp.engine.objects import MATRIX_COLUMNS
    from mgp.engine.problem import make_training_dataset, make_workspace
    assert list(MATRIX_COLUMNS) == COLS, MATRIX_COLUMNS
    size, ratio, start, seeds = task
    n_o, n_v = object_counts(size, ratio)
    cfg = {**base, 'target': target_name, 'operator_count': n_o, 'variable_atom_count': n_v, 'internal_merge_probability': 0.0, 'esel': 0.0, 'esur': 0.0,
           'merge_strategy': 'standard'}
    variables, target, clean = make_training_dataset(TARGETS[target_name][0], cfg, 1)
    total = np.zeros((size, len(COLS)))
    lengths, terminations = [], {}
    for ws_seed, d_seed in seeds:
        ws = make_workspace(variables, cfg, ws_seed)
        d = Derivation(target, cfg, d_seed, clean)
        d.run(ws)
        a = np.array([d.initial_counts] + d.step_counts, dtype=float)
        if len(a) < size:
            a = np.vstack([a, np.repeat(a[-1:], size - len(a), axis=0)])
        total += a
        lengths.append(len(d.step_counts))
        terminations[d.termination_reason] = terminations.get(d.termination_reason, 0) + 1
    return size, ratio, start, len(seeds), n_o, n_v, total, lengths, terminations
