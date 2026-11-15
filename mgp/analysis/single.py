import numpy as np
import pandas as pd

from mgp.analysis.data import FACTORS, levels
from mgp.analysis.stats import ARCH, GOALS, ORDER, SN, TN, V2_ORDER, bold, holm, mcnemar, mean_or_nan, measures, num, pct, present, row_label, signed_rank

KEYS = list(FACTORS)
COV = ('component', 'cooccurrence', 'composite')


def lab(s, im):
    return f'{SN[s]}, IM {im:g}'


def complete(frame):
    cells = frame.groupby(KEYS + ['target']).ngroups
    per_rep = frame.groupby('rep').case_id.nunique()
    return frame[frame.rep.isin(per_rep[per_rep == cells].index)].reset_index(drop=True)


def frame(raw):
    S1 = complete(raw)
    S1['sym'] = (S1.symbolic_recovery == 1).astype(int)
    S1['test_e'] = S1.test_rmse.fillna(np.inf)
    S1['train_e'] = S1.train_rmse.fillna(np.inf)
    for a, c in ARCH.items():
        for g in GOALS:
            S1[f'{a}_{g}'] = np.nan
        for t in targets(S1):
            sel = (S1.target == t).to_numpy()
            m = measures(t, S1.loc[sel, c].to_numpy(dtype=np.uint64))
            for g in GOALS:
                S1.loc[sel, f'{a}_{g}'] = 100 * np.asarray(m[g], dtype=float)
    S1['BA_size'], S1['BBA_size'], S1['FBBA_size'] = S1.base_fitness_entries, S1.bb_snapshot_entries, S1.best_bb_entries
    return S1


def targets(df):
    return present(ORDER, df.target)


def cell(df, ref, f=None, v=None):
    cond = {**ref, **({f: v} if f else {})}
    m = np.ones(len(df), dtype=bool)
    for k, val in cond.items():
        m &= (df[k] == val).to_numpy()
    return df[m].sort_values(['target', 'rep'])


def per_target(y, c, kind, ts):
    out = {}
    for t in ts:
        v = y.loc[y.target == t, c].to_numpy(dtype=float)
        if kind == 'rmse':
            out[t] = float(np.median(v))
        else:
            out[t] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    return out


def summary(per, kind):
    v = np.array([x for x in per.values() if np.isfinite(x)])
    if len(v) == 0:
        return np.nan
    return float(100 * v.mean()) if kind == 'rate' else float(v.mean())


def single_table(S1, cols, fmt, rows, ref):
    ts = targets(S1)
    base = cell(S1, ref)
    vals, ps, cons = [], {k: [] for k, _, _ in cols}, {k: [] for k, _, _ in cols}
    ref_per = {k: per_target(base, c, kind, ts) for k, c, kind in cols}
    for f, v in rows:
        y = cell(S1, ref, f, v)
        assert (y.target.to_numpy() == base.target.to_numpy()).all() and (y.rep.to_numpy() == base.rep.to_numpy()).all()
        per = {k: per_target(y, c, kind, ts) for k, c, kind in cols}
        vals.append({k: summary(per[k], kind) for k, _, kind in cols})
        if f is not None:
            for k, c, kind in cols:
                ps[k].append(mcnemar(base[c], y[c]) if kind == 'rate' else signed_rank(base[c], y[c]))
                d = {t: per[k][t] - ref_per[k][t] for t in ts if np.isfinite(per[k][t]) and np.isfinite(ref_per[k][t])}
                signs = [0 if abs(x) < 1e-12 else int(np.sign(x)) for x in d.values()]
                total = vals[-1][k] - vals[0][k]
                direction = 0 if abs(total) < 1e-12 else int(np.sign(total))
                visible = fmt[k](vals[-1][k]) != fmt[k](vals[0][k])
                cons[k].append(bool(direction != 0 and visible and all(x in (0, direction) for x in signs) and direction in signs))
    sig = {k: [False] + [bool(p < 0.05 and c) for p, c in zip(holm(ps[k]), cons[k])] for k, _, _ in cols}
    return vals, sig


def tables(S1, rows, ref):
    out = {}
    cols = [('built', 'evaluated_target_ever', 'rate'), ('num', 'test_success', 'rate'), ('sym', 'sym', 'rate'), ('train', 'train_e', 'rmse'),
            ('test', 'test_e', 'rmse')]
    fmt = {'built': pct, 'num': pct, 'sym': pct, 'train': num, 'test': num}
    v, s = single_table(S1, cols, fmt, rows, ref)
    out['recovery'] = {'values': v, 'bold': s, 'lines': [' & '.join([row_label(ref, f, x)] + [bold(fmt[k](v[i][k]), s[k][i]) for k, _, _ in cols])
                                                        for i, (f, x) in enumerate(rows)]}
    covg = ('component', 'cooccurrence', 'composite', 'highpower')
    for a in ARCH:
        cols = [('size', f'{a}_size', 'cont')] + [(g, f'{a}_{g}', 'cont') for g in covg]
        fmt = {'size': pct, **{g: pct for g in covg}}
        v, s = single_table(S1, cols, fmt, rows, ref)
        out[a] = {'values': v, 'bold': s, 'cells': [[bold(fmt[k](v[i][k]), s[k][i]) for k, _, _ in cols] for i in range(len(rows))]}
    vals = []
    for f, x in rows:
        y = cell(S1, ref, f, x)
        g = []
        for t, z in y.groupby('target'):
            b = z.loc[z.train_e.idxmin()]
            g.append({'built': int(z.evaluated_target_ever.max()), 'num': int(z.test_success.max()), 'sym': int(z.sym.max()),
                      'train': float(b.train_e), 'test': float(b.test_e), 'target': t})
        g = pd.DataFrame(g)
        vals.append({'built': int(g.built.sum()), 'num': int(g.num.sum()), 'sym': int(g.sym.sum()), 'train': float(g.train.mean()),
                     'test': float(g.test.mean())})
    out['groups'] = vals
    out['repetitions'] = int(S1.rep.nunique())
    out['targets'] = len(targets(S1))
    return out


def group_lists(S1, rows, ref):
    out = []
    for f, v in rows:
        cond = {**ref, **({f: v} if f else {})}
        m = np.ones(len(S1), dtype=bool)
        for k, val in cond.items():
            m &= (S1[k] == val).to_numpy()
        g = S1[m].groupby('target').agg(built=('evaluated_target_ever', 'max'), num=('test_success', 'max'), sym=('sym', 'max'))
        out.append({c: [n for n in V2_ORDER if any(TN[t] == n and g.loc[t, c] == 1 for t in g.index)] for c in ('built', 'num', 'sym')})
    return out


def standard_targets(S1, ref):
    std = S1[S1.strategy == ref['strategy']]
    base = std.copy()
    for k, v in ref.items():
        base = base[base[k] == v]
    out = {}
    for t in targets(S1):
        r, a = base[base.target == t], std[std.target == t]
        out[TN[t]] = {**{f'ref_{k}': 100 * float(r[c].mean()) for k, c in (('built', 'evaluated_target_ever'), ('num', 'test_success'), ('sym', 'sym'))},
                      **{f'all_{k}': 100 * float(a[c].mean()) for k, c in (('built', 'evaluated_target_ever'), ('num', 'test_success'), ('sym', 'sym'))}}
    return out


def describe(c):
    return f"({SN[c['strategy']]}, {c['im']:g}, {int(c['size'])}, {c['ratio']}, {c['esel']:g}, {c['esur']:g})"


def best_configurations(S1):
    x = S1.copy()
    x['num'] = 100 * x.test_success
    x['sym_pct'] = 100 * (x.symbolic_recovery == 1)
    cols = ['num', 'sym_pct'] + [f'{a}_{g}' for a in ARCH for g in COV]
    conf = x.groupby(KEYS + ['target'])[cols].mean().reset_index().groupby(KEYS)[cols].mean().reset_index()
    out = {}
    for c in cols:
        best = conf.sort_values(c, ascending=False).iloc[0]
        out[c] = {'value': float(best[c]), 'configuration': describe(best)}
    return out


PANELS = [('nguyen_1', 'composite'), ('nguyen_4', 'component'), ('nguyen_4', 'cooccurrence')]


def ratio_coverage(S1, panels=PANELS):
    res = {'ratios': levels(S1, 'ratio'), 'strategies': levels(S1, 'strategy'), 'im': levels(S1, 'im')}
    for t, g in panels:
        x = S1[S1.target == t]
        if not len(x):
            continue
        m = measures(t, x.bb_snapshot.to_numpy(dtype=np.uint64))[g]
        for s in res['strategies']:
            for im in res['im']:
                res[f'{t}|{g}|{lab(s, im)}'] = [100 * mean_or_nan(m[((x.strategy == s) & (x.im == im) & (x.ratio == r)).to_numpy()])
                                                for r in res['ratios']]
    return res


def load(path, filters=None):
    import hashlib
    from mgp.analysis import data as V
    key = 'study1_frame' + ('' if not filters else '_' + hashlib.sha256(repr(sorted(filters.items())).encode()).hexdigest()[:6])
    return V.cached(key, path, lambda: frame(V.where(V.study1(path), filters)))
