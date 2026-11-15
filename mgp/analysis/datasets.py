import numpy as np
from scipy.stats import chi2_contingency, kruskal

from mgp.analysis.stats import ARCH, ORDER, TN, holm, mcnemar, measures, present, signed_rank


def datasets_table(X):
    X = X.copy()
    X['sym'] = (X.symbolic_recovery == 1).astype(int)
    X['comp'] = np.nan
    for t in X.target.unique():
        sel = (X.target == t).to_numpy()
        X.loc[sel, 'comp'] = 100 * np.asarray(measures(t, X.loc[sel, 'bb_snapshot'].to_numpy(dtype=np.uint64))['component'], dtype=float)
    recs, p_rec, p_cov = [], [], []
    for t in ORDER:
        y = X[X.target == t]
        if not len(y):
            continue
        per_set = y.groupby('dataset_replica').test_success.mean() * 100
        rec = {'target': TN[t], 'derivations': int(len(y)), 'sets': int(y.dataset_replica.nunique()),
               'per_config_set': int(len(y) / y.config_id.nunique() / y.dataset_replica.nunique()), 'configs': int(y.config_id.nunique()),
               'num': 100 * y.test_success.mean(), 'sym': 100 * y.sym.mean(), 'set_min': float(per_set.min()), 'set_max': float(per_set.max()),
               'comp': float(y.comp.mean())}
        for cfg, z in y.groupby('config_id'):
            tab = z.groupby('dataset_replica').test_success.agg(['sum', 'size'])
            k = tab['sum'].to_numpy()
            n = tab['size'].to_numpy()
            p_rec.append((TN[t], 1.0 if k.sum() in (0, n.sum()) else float(chi2_contingency(np.c_[k, n - k])[1])))
            groups = [g.comp.to_numpy() for _, g in z.groupby('dataset_replica')]
            allv = np.concatenate(groups)
            p_cov.append((TN[t], 1.0 if np.all(allv == allv[0]) else float(kruskal(*groups).pvalue)))
        recs.append(rec)
    for key, plist in (('rec_effect', p_rec), ('cov_effect', p_cov)):
        adj = holm([p for _, p in plist])
        for r in recs:
            r[key] = int(sum(1 for (t, _), a in zip(plist, adj) if t == r['target'] and a < 0.05))
            r[key + '_raw'] = int(sum(1 for (t, p) in plist if t == r['target'] and p < 0.05))
    return recs


def training_table(T):
    T = T.rename(columns={'training_size': 'n'}).copy()
    T['num'] = T.test_success
    T['sym'] = (T.symbolic_recovery == 1).astype(int)
    T['built'] = T.evaluated_target_ever
    for a, c in ARCH.items():
        T[a] = T[c]
        T[f'{a}_component'] = np.nan
        for t in present(ORDER, T.target):
            sel = (T.target == t).to_numpy()
            T.loc[sel, f'{a}_component'] = 100 * np.asarray(measures(t, T.loc[sel, a].to_numpy(dtype=np.uint64))['component'])
    cols = [('num', 'num'), ('sym', 'sym')] + [(a, f'{a}_component') for a in ARCH]
    key = ['target', 'config_id', 'rep']
    sizes = sorted(T.n.unique(), reverse=True)
    k = T.groupby(key).n.nunique()
    T = T.merge(k[k == len(sizes)].rename('k').reset_index(), on=key).sort_values(key)
    refT = T[T.n == sizes[0]]
    vals, ps = [], {c: [] for c, _ in cols}
    for n in sizes:
        y = T[T.n == n]
        vals.append({'n': int(n), 'derivations': int(len(y)),
                     **{c: float(100 * y[col].mean()) if c in ('num', 'sym') else float(y[col].mean()) for c, col in cols}})
        if n != sizes[0]:
            assert (y[key].to_numpy() == refT[key].to_numpy()).all()
            for c, col in cols:
                ps[c].append(mcnemar(refT[col], y[col]) if c in ('num', 'sym') else signed_rank(refT[col], y[col]))
    sig = {c: [False] + [bool(x) for x in holm(ps[c]) < 0.05] for c, _ in cols}
    return {'values': vals, 'significant': sig}
