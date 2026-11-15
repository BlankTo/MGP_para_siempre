import math
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.stats import binomtest, wilcoxon

from mgp.analysis.data import FACTORS, levels

TN = {'g1': 'G1', 'g2': 'G2', 'nguyen_1': 'N1', 'nguyen_2': 'N2', 'nguyen_3': 'N3', 'nguyen_4': 'N4', 'nguyen_12': 'N12'}
ORDER = ['g1', 'nguyen_1', 'nguyen_2', 'nguyen_3', 'nguyen_4', 'g2', 'nguyen_12']
V2_ORDER = ['G1', 'G2', 'N1', 'N2', 'N3', 'N4', 'N12']
STRATS = ['standard', 'revisit_ic', 'constructive']
SN = {'standard': 'Standard', 'revisit_ic': 'Revisit IC', 'constructive': 'Constructive'}
ARCH = {'BA': 'base_fitness', 'BBA': 'bb_snapshot', 'FBBA': 'best_bb'}
GOALS = ('component', 'cooccurrence', 'composite', 'highpower')
PAR = {'strategy': 'strategy', 'im': '$p_{IM}$', 'size': '$N_0$', 'ratio': 'ratio', 'esel': '$e_{sel}$', 'esur': '$e_{sur}$'}
SHORT_FMT = {'im': '$p_{{IM}}={}$', 'size': '$N_0={}$', 'ratio': 'ratio {}', 'esel': '$e_{{sel}}={}$', 'esur': '$e_{{sur}}={}$'}


def fmt_val(f, v):
    if f == 'strategy':
        return SN[v]
    if isinstance(v, float):
        return f'{v:g}'
    return str(v)


def row_label(ref, f=None, v=None):
    if f is None:
        return 'reference'
    return f'{PAR[f]}: {fmt_val(f, ref[f])} $\\to$ {fmt_val(f, v)}'


def short(f=None, v=None):
    if f is None:
        return 'reference'
    if f == 'strategy':
        return SN[v]
    return SHORT_FMT[f].format(fmt_val(f, v))


def design_rows(frame, ref):
    missing = [(f, ref[f]) for f in FACTORS if ref[f] not in set(frame[f])]
    if missing:
        raise ValueError(f'reference levels absent {missing}')
    return [(None, None)] + [(f, v) for f in FACTORS for v in levels(frame, f) if v != ref[f]]


def present(order, values):
    values = set(values)
    return [v for v in order if v in values]


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def mcnemar(a, b):
    a, b = np.asarray(a) == 1, np.asarray(b) == 1
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    return 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue


def signed_rank(a, b):
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.all(d == 0):
        return 1.0
    return float(wilcoxon(d, zero_method='wilcox').pvalue)


def num(x, digits=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '--'
    if x == 0:
        return '0'
    if math.isinf(x):
        return '$\\infty$'
    if abs(x) >= 0.01:
        return f'{x:#.{digits}g}' if abs(x) < 1000 else f'{x:,.0f}'
    e = math.floor(math.log10(abs(x)))
    return f'${x / 10 ** e:.1f}\\cdot10^{{{e}}}$'


def pct(x, d=1):
    return '--' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{d}f}'


def bold(s, sig):
    return f'\\textbf{{{s}}}' if sig else s


@lru_cache(maxsize=None)
def pieces(target):
    from mgp.engine.targets import subtarget_specs, lexical_component_ids
    return tuple((i, text) for i, text, _ in subtarget_specs(target, 2)), tuple(sorted(lexical_component_ids(target)))


@lru_cache(maxsize=None)
def cat_info(target):
    items, lexical = pieces(target)
    ids = [i for i, _ in items]
    lex = set(lexical)
    singles = [i for i in ids if bin(i).count('1') == 1]
    comps = [i for i in singles if i not in lex]
    composites = [i for i in ids if bin(i).count('1') > 1]
    groups = [sum(c) for k in range(2, len(comps) + 1) for c in combinations(comps, k)]
    return ids, singles, comps, composites, groups


def coverage(target, masks):
    _, _, comps, composites, groups = cat_info(target)
    m = np.asarray(masks, dtype=np.uint64)
    bit = lambda i: ((m >> np.uint64(i)) & np.uint64(1)).astype(bool)
    comp = np.mean([bit(i) for i in comps], axis=0) if comps else np.full(len(m), np.nan)
    compo = np.mean([bit(i) for i in composites], axis=0)
    avail = np.zeros(len(m), dtype=np.int64)
    for i in comps:
        avail |= bit(i).astype(np.int64) * i
    co = np.mean([(avail & g) == g for g in groups], axis=0) if groups else np.full(len(m), np.nan)
    return comp, co, compo


def measures(target, masks):
    comp, co, compo = coverage(target, masks)
    hp = [i for i, s in pieces(target)[0] if s in ('x1**4', 'x1**5', 'x1**6')]
    m = np.asarray(masks, dtype=np.uint64)
    h = np.mean([((m >> np.uint64(i)) & np.uint64(1)).astype(float) for i in hp], axis=0) if hp else np.full(len(m), np.nan)
    return {'component': comp, 'cooccurrence': co, 'highpower': h, 'composite': compo}


def s2_parts(target):
    items, lexical = pieces(target)
    compo = [i for i, t in items if (' + ' in t or ' - ' in t.strip('- '))]
    single = [i for i, t in items if i not in compo and i not in lexical]
    return single, compo


def s2_measures(u, single, compo):
    u = np.asarray(u, dtype=np.int64)
    bit = lambda i: ((u >> i) & 1).astype(float)
    comp = np.mean([bit(i) for i in single], axis=0)
    sets = [c for k in range(2, len(single) + 1) for c in combinations(single, k)]
    co = np.mean([np.prod([bit(i) for i in c], axis=0) for c in sets], axis=0)
    cmp_ = np.mean([bit(i) for i in compo], axis=0)
    return comp, co, cmp_


def mean_or_nan(v):
    return float(np.nanmean(v)) if not np.all(np.isnan(v)) else np.nan
