import numpy as np

from mgp.analysis import data as V
from mgp.analysis.stats import STRATS, present


def frame(path6, target, time='search_and_archives', filters=None, diversity=False):
    t6 = V.where(V.study6(path6), filters)
    t6 = t6[t6.target == target].copy()
    t6['seconds'] = V.study6_seconds(t6, time, diversity)
    return t6


def columns(x, ref):
    cols = [(None, None)]
    for f in ('im', 'size', 'ratio', 'esel', 'esur'):
        cols += [(f, v) for v in V.levels(x, f) if v != ref[f]]
    return cols


def heatmap(x, ref):
    strategies = present(STRATS, x.strategy)
    cols = columns(x, ref)
    M = np.full((len(strategies), len(cols)), np.nan)
    for i, s in enumerate(strategies):
        for j, (f, v) in enumerate(cols):
            cond = {**ref, 'strategy': s, **({f: v} if f else {})}
            m = np.ones(len(x), dtype=bool)
            for k, val in cond.items():
                m &= (x[k] == val).to_numpy()
            if m.any():
                M[i, j] = 1000 * float(np.mean(x.seconds[m]))
    return M, strategies, cols
