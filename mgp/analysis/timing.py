import numpy as np

from mgp.analysis import data as V
from mgp.analysis.stats import STRATS, present

KEYS = list(V.FACTORS)


def frame(path6, target, time='search_and_archives', filters=None):
    t6 = V.where(V.study6(path6), filters)
    t6 = t6[t6.target == target].copy()
    t6['seconds'] = V.study6_seconds(t6, time)
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


def numbers(x, ref):
    ms = x.groupby(KEYS).seconds.mean() * 1000
    M, strategies, cols = heatmap(x, ref)
    out = {'min_ms': float(ms.min()), 'max_ms': float(ms.max())}
    if 'revisit_ic' in strategies and 'standard' in strategies:
        revisit = M[strategies.index('revisit_ic')] / M[strategies.index('standard')] - 1
        out['revisit_slower_pct'] = [float(100 * np.nanmin(revisit)), float(100 * np.nanmax(revisit))]
    if ('size', 256) in cols:
        j = cols.index(('size', 256))
        out['size_256_ratio'] = {s: float(M[i, j] / M[i, 0]) for i, s in enumerate(strategies)}
    pivot = ms.unstack('strategy')
    if 'constructive' in pivot and 'standard' in pivot:
        out['constructive_over_standard_median'] = float((pivot['constructive'] / pivot['standard']).median())
    return out
