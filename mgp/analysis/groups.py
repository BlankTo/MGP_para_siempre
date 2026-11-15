import math
import zlib

import numpy as np

from mgp.analysis import data as V
from mgp.analysis.stats import ORDER, SN, STRATS, TN, measures, present, s2_measures, s2_parts

NS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000]
ARCHS = [('ba', 'base_fitness', 'BA'), ('bb', 'bb_snapshot', 'BBA'), ('fb', 'best_bb', 'FBBA')]
DIVERSITY = ('div', None, 'BAdiv')
KEYS = list(V.FACTORS)


def group_size(path):
    return int(V.meta(path)['params']['group_size'])


def growth(masks_by_group, n_grid, meas):
    u = np.bitwise_or.accumulate(masks_by_group, axis=1)[:, [k - 1 for k in n_grid]]
    vals = meas(u.ravel())
    return {g: (100 * np.asarray(v, dtype=float).reshape(u.shape).mean(axis=0)).tolist() for g, v in vals.items()}


def frame(path, target, filters=None):
    size = group_size(path)
    R2 = V.where(V.study2(path, target), filters)
    n = R2.groupby(['case_id', 'pool']).rep.transform('size')
    return R2[n == size].sort_values(['case_id', 'pool', 'rep']), size


def measure(target):
    single, compo = s2_parts(target)

    def meas(u):
        comp, co, cp = s2_measures(u, single, compo)
        return {'component': comp, 'cooccurrence': co, 'composite': cp}
    return meas


def bounded(masks, meas):
    u = np.asarray(masks, dtype=np.int64)
    vals = meas(u.ravel())
    return {g: (100 * np.asarray(v, dtype=float).reshape(u.shape).mean(axis=0)).tolist() for g, v in vals.items()}


def group_growth(path, target, filters=None, where=None, R2=None, diversity=False, final=None):
    from mgp.analysis import capacity
    if R2 is None:
        R2, size = frame(path, target, filters)
    else:
        size = group_size(path)
    if where is not None:
        R2 = R2[where(R2)]
    meas = measure(target)
    ns = [n for n in NS if n <= size]
    archs = ARCHS + ([DIVERSITY] if diversity else [])
    if final is not None or diversity:
        names = [a for a, _, _ in archs] if final is not None else ['div']
        pm = capacity.prefix_masks(path, target, final, ns, names, keep=sorted(set(R2.case_id)))
    res = {}
    strategies = present(STRATS, R2.strategy)
    for s in strategies:
        y = R2[R2.strategy == s]
        P = len(y) // size
        if not P:
            continue
        for key, col, an in archs:
            if final is None and col is not None:
                curve = growth(y[col].to_numpy(dtype=np.int64).reshape(P, size), ns, meas)
            else:
                curve = bounded([pm[k][key] for k in sorted(set(zip(y.case_id, y.pool))) if k in pm], meas)
            for g, v in curve.items():
                res[f'{s}|{an}|{g}'] = v
            res[f'{s}|{an}|groups'] = P
    return {'n': ns, 'values': res, 'strategies': strategies, 'group_size': size}


def novelty(path, target, pools=None):
    cases = V.case_dicts(path, target)
    nmax = group_size(path)
    by = {}
    for cid, pool, rep, blob in V.study2_evaluated(path, target, pools):
        by.setdefault((cid, pool), []).append((rep, np.frombuffer(zlib.decompress(blob), dtype=np.uint64)))
    curves, sizes = {}, {}
    for (cid, pool), v in by.items():
        if [r for r, _ in v] != list(range(1, nmax + 1)):
            continue
        d = np.concatenate([x for _, x in v])
        r = np.concatenate([np.full(len(x), k, np.int32) for k, x in v])
        o = np.lexsort((r, d))
        d, r = d[o], r[o]
        first = r[np.r_[True, d[1:] != d[:-1]]]
        s = cases[cid]['merge_strategy']
        curves.setdefault(s, []).append(np.bincount(first - 1, minlength=nmax))
        sizes.setdefault(s, []).append(np.array([len(x) for _, x in v]))
    n = np.arange(1, nmax + 1)
    stats = {}
    curves = {s: curves[s] for s in present(STRATS, curves)}
    for s in curves:
        new, size = np.stack(curves[s]).astype(float), np.stack(sizes[s]).astype(float)
        m = new.mean(axis=0)
        k = 9 if nmax > 10 else 0
        slope = np.polyfit(np.log10(n[k:]), np.log10(np.maximum(m[k:], 1e-9)), 1)[0]
        stats[s] = {'groups': len(new), 'first': float(m[0]), 'last100': float(m[-100:].mean()), 'evaluated_mean': float(size.mean()),
                    'distinct_total': float(new.sum(axis=1).mean()), 'decline_per_tenfold': float(100 * (1 - 10 ** slope))}
    return curves, stats


def config_times(path6, target, time='search_and_archives', diversity=False):
    t6 = V.study6(path6)
    t6 = t6[t6.target == target].copy()
    t6['seconds'] = V.study6_seconds(t6, time, diversity)
    tm = t6.groupby(KEYS).seconds.mean()
    return {(s, float(im), int(size), ratio, float(e), float(u)): float(v) for (s, im, size, ratio, e, u), v in tm.items()}


def group_time(path, path6, target, filters=None, time='search_and_archives', R2=None, diversity=False, final=None):
    from mgp.analysis import capacity
    if R2 is None:
        R2, size = frame(path, target, filters)
    else:
        size = group_size(path)
    meas = measure(target)
    T6 = config_times(path6, target, time, diversity)
    NT = sorted({int(round(x)) for x in np.logspace(0, np.log10(size), 40)})
    if final is not None:
        pm = capacity.prefix_masks(path, target, final, NT, ['bb'], keep=sorted(set(R2.case_id)))
    curves = []
    for cid, y in R2.groupby('case_id'):
        P = len(y) // size
        key = (y.strategy.iat[0], float(y.im.iat[0]), int(y['size'].iat[0]), y.ratio.iat[0], float(y.esel.iat[0]), float(y.esur.iat[0]))
        if key not in T6:
            continue
        if final is None:
            g = growth(y['bb_snapshot'].to_numpy(dtype=np.int64).reshape(P, size), NT, meas)
        else:
            g = bounded([pm[k]['bb'] for k in sorted(set(zip(y.case_id, y.pool))) if k in pm], meas)
        curves.append({'cfg': key, 't': max(T6[key], 1e-9), 'groups': P, **g})
    B = np.logspace(-3, 4, 85)
    lnT = np.log(NT)
    goals = ('component', 'cooccurrence', 'composite')
    strategies = present(STRATS, [c['cfg'][0] for c in curves])
    res = {'budgets_s': B.tolist(), 'configurations': len(curves), 'strategies': strategies}
    res['by_config'] = {f'{b:g}s': [{'cfg': list(c['cfg']), 'n': b / c['t'], **{g: float(np.interp(np.log(min(b / c["t"], size)), lnT, c[g]))
                                                                                 for g in goals}} for c in curves if b / c['t'] >= 1] for b in (1.0, 10.0)}
    for s in strategies:
        cs = [c for c in curves if c['cfg'][0] == s]
        stop = max(size * c["t"] for c in cs)
        for g in goals:
            best, arg = [], []
            for b in B:
                vals = []
                for c in cs:
                    k = b / c['t']
                    if k < 1:
                        continue
                    vals.append((float(np.interp(np.log(min(k, size)), lnT, c[g])), c['cfg']))
                if not vals or b > stop:
                    best.append(None)
                    arg.append(None)
                else:
                    v, a = max(vals, key=lambda x: x[0])
                    best.append(v)
                    arg.append(list(a))
            res[f'{s}|{g}'] = best
            res[f'{s}|{g}|config'] = arg
        res[f'{s}|t_ms'] = [1000 * min(c['t'] for c in cs), 1000 * max(c['t'] for c in cs)]
    return res


def recovery7(X):
    X = X.copy()
    X['train'] = X.train_rmse.fillna(math.inf)
    X['test'] = X.test_rmse.fillna(math.inf)
    out = []
    for t in present(ORDER, X.target):
        for s in present(STRATS, X.strategy):
            y = X[(X.target == t) & (X.strategy == s)]
            if not len(y):
                continue
            g = y.groupby('pool')
            best = y.loc[g.train.idxmin()]
            p = y.num.mean()
            out.append({'target': TN[t], 'strategy': SN[s], 'groups': int(g.ngroups), **{k: 100 * float(y[k].mean()) for k in ('built', 'num', 'sym')},
                        **{f'groups_{k}': int(g[k].max().sum()) for k in ('built', 'num', 'sym')}, 'best_train': float(best.train.mean()),
                        'best_test': float(best.test.mean()), 'half_num': None if p == 0 else (1.0 if p >= 0.5 else math.log(2) / -math.log(1 - p))})
    return out


def coverage_growth7(X, target='nguyen_12', permutations=50, seed=20261105, final=None, path7=None, diversity=False):
    from mgp.analysis import capacity
    X = X[X.target == target].sort_values(['strategy', 'pool', 'rep'])

    def meas(u):
        m = measures(target, np.asarray(u, dtype=np.uint64))
        return {g: m[g] for g in ('component', 'cooccurrence', 'composite')}

    rng = np.random.default_rng(seed)
    archs = ARCHS + ([DIVERSITY] if diversity else [])
    res = {}
    size = None
    strategies = present(STRATS, X.strategy)
    for s in strategies:
        y = X[X.strategy == s]
        size = int(y.groupby('pool').size().iloc[0])
        P = y.pool.nunique()
        perms = [rng.permutation(size) for _ in range(permutations)]
        ns = [n for n in NS if n <= size]
        if final is not None or diversity:
            orders = tuple(tuple(int(i) for i in p) for p in perms)
            names = [a for a, _, _ in archs] if final is not None else ['div']
            pm = capacity.prefix_masks(path7, target, final, ns, names, keep=sorted(set(y.case_id)), orders=orders)
        for key, col, an in archs:
            if final is None and col is not None:
                m = y[col].to_numpy(dtype=np.int64).reshape(P, size)
                seqs = np.concatenate([m[:, p] for p in perms], axis=0)
                curve = growth(seqs, ns, meas)
            else:
                curve = bounded([o[key] for k in sorted(set(zip(y.case_id, y.pool))) if k in pm for o in pm[k]], meas)
            for g, v in curve.items():
                res[f'{s}|{an}|{g}'] = v
            res[f'{s}|{an}|groups'] = P
    return {'n': [n for n in NS if size and n <= size], 'values': res, 'strategies': strategies, 'group_size': size}
