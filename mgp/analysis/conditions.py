import statistics

import numpy as np
import pandas as pd

from mgp.analysis import data as V
from mgp.analysis.stats import ARCH, ORDER, TN, measures, pct, present

LEV = {0.0: '0 (reference)', 0.01: '0.01', 0.1: '0.1', 1.0: '1', 10.0: '10', 'pure_noise': 'pure noise'}
CONDS = ('clean', 'pure_noise', 'random_ic', 'inverted', 'inverted_all')
LAB = {'clean': 'noise-free', 'pure_noise': 'pure noise', 'random_ic': 'random IC', 'inverted': 'inverted, best', 'inverted_all': 'inverted, worst'}
COV = ('component', 'cooccurrence', 'composite')


def level(cond):
    if cond in LEV:
        return LEV[cond]
    return f'{cond:g}' if isinstance(cond, float) else str(cond).replace('_', ' ')


def noise_metrics(R, cond):
    X = [r['res'][cond] for r in R]
    ref = [r['res'][0.0] for r in R]
    n = len(X)
    sym = [x['symbolic'] for x in X if x['symbolic'] is not None]
    first = [x['first_divergence'] for x in X if x['first_divergence'] is not None]
    ic = sum(x['ic'] for x in X)
    el = sum(x['elite'] for x in X)
    reference = cond == 0.0
    return {'runs': n, 'built': 100 * sum(x['built'] for x in X) / n, 'num': 100 * sum(x['numerical'] for x in X) / n,
            'sym': 100 * sum(sym) / len(sym) if sym else None, 'overlap': 100 * sum(x['jaccard'] for x in X) / n,
            'identical': 100 * sum(x['hash'] == y['hash'] for x, y in zip(X, ref)) / n,
            'first_median': None if reference or not first else statistics.median(first),
            'ic_changed': None if reference or not ic else 100 * sum(x['ic_real'] + x['ic_equiv'] for x in X) / ic,
            'elite_changed': None if reference or not el else 100 * sum(x['elite_real'] + x['elite_equiv'] for x in X) / el}


def noise_table(res):
    res = [r for r in res if 0.0 in r['res']]
    conds = [c for c in res[0]['res'] if all(c in r['res'] for r in res)]
    out = {'pairs': len(res), 'all': {}, 'target': {}, 'targets': len({r['target'] for r in res})}
    for cond in conds:
        out['all'][level(cond)] = noise_metrics(res, cond)
        for t in ORDER:
            sub = [r for r in res if r['target'] == t]
            if sub:
                out['target'].setdefault(TN[t], {})[level(cond)] = noise_metrics(sub, cond)
    fmt = lambda x: '--' if x is None else f'{x:.1f}'
    lines = []
    for c in conds:
        m = out['all'][level(c)]
        ref = c == 0.0
        lines.append(' & '.join([level(c), fmt(m['identical']), '--' if ref else f"{m['first_median']:.0f}",
                                 '--' if ref or m['ic_changed'] is None else f"{m['ic_changed']:.2f}",
                                 '--' if ref or m['elite_changed'] is None else f"{m['elite_changed']:.2f}", fmt(m['overlap'])]) + ' \\\\')
    return out, lines


def labels_table(frame):
    rows = []
    conds = [c for c in CONDS if all(c in r for r in frame.result)]
    for r in frame.itertuples():
        for cond in conds:
            v = r.result[cond]
            rows.append({'target': r.target, 'case_id': r.case_id, 'rep': r.rep, 'cond': cond, 'built': v['built'], 'num': v['numerical'],
                         'sym': int(v['symbolic'] == 1), 'overlap': v['jaccard'], **{a: v[a] for a in ARCH.values()}})
    X = pd.DataFrame(rows)
    ts = present(ORDER, X.target)
    for a, col in ARCH.items():
        for t in ts:
            sel = (X.target == t).to_numpy()
            m = measures(t, X.loc[sel, col].to_numpy(dtype=np.uint64))
            for g in COV:
                X.loc[sel, f'{a}_{g}'] = 100 * np.asarray(m[g], dtype=float)
    out, lines = {}, []
    for cond in conds:
        y = X[X.cond == cond]
        o = {'derivations': int(len(y)), **{k: float(100 * y[k].mean()) for k in ('built', 'num', 'sym', 'overlap')}}
        for a in ARCH:
            for g in COV:
                v = [y.loc[y.target == t, f'{a}_{g}'].mean() for t in ts]
                v = [x for x in v if np.isfinite(x)]
                o[f'{a}_{g}'] = float(np.mean(v)) if v else None
        out[cond] = o
        lines.append(' & '.join([LAB[cond]] + [pct(o[k]) for k in ('built', 'num', 'sym', 'overlap')]
                                + [pct(o[f'{a}_{g}']) for a in ('BBA', 'FBBA') for g in COV]) + ' \\\\')
    return out, lines


def select(res, filters):
    if not filters:
        return res
    keep = set(V.where(pd.DataFrame([{**{k: r[k] for k in ('target', 'strategy', 'im', 'size', 'esel', 'esur', 'rep')}, 'i': i}
                                     for i, r in enumerate(res)]), filters).i)
    return [r for i, r in enumerate(res) if i in keep]
