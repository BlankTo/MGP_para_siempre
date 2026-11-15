import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import config
from mgp.analysis import data as V
from mgp.analysis.stats import STRATS, measures, present
from mgp.progress import Progress

GOALS = ('component', 'cooccurrence', 'composite')
STATIC = {'ba': ('BA', 'ba_err', 'min'), 'bb': ('BBA', 'bb_score', 'sum'), 'fb': ('FBBA', 'fb_score', 'sum')}
DIVERSITY_THETA = 0.99
DIVERSITY_LABEL = 'BAdiv'
TITLES = {'BA': 'BA', 'BBA': 'BBA', 'FBBA': 'FBBA', DIVERSITY_LABEL: 'BA diversity'}
NEG = -np.inf
ONE = np.uint64(1)
LOADER = []


def ks(capacity_k, single=None):
    if single is not None:
        return tuple(sorted({k for k in capacity_k if k <= single} | {single}))
    k = tuple(capacity_k)
    return k + (2 * k[-1],)


def ksn(capacity_k, single=None):
    return ks(capacity_k, single) + (None,)


def single_of(path):
    return V.meta(path)['identity'].get('campaign', {}).get('settings', {}).get('single_archive_max_size')


def load(blob):
    from mgp.experiments.recording import load as decode
    return (LOADER[0] if LOADER else decode)(blob)


def bits_mask(bits):
    b = np.asarray(bits)
    b = b[b >= 0]
    if not b.size:
        return 0
    return int(np.bitwise_or.reduce(np.left_shift(ONE, b.astype(np.uint64))))


def cover(target, masks):
    v = measures(target, np.asarray(masks, dtype=np.uint64))
    return {g: 100.0 * float(np.nanmean(np.asarray(v[g], dtype=float))) for g in GOALS}


def static_lists(d, a, scol, KSN):
    if a == 'ba':
        stored = {K: np.asarray(l, dtype=int) for K, l in zip(d['ksn'], d['lists'][a])}
        full = stored[None]
        out = []
        for K in KSN:
            if K in stored:
                out.append(stored[K])
            elif K >= len(full):
                out.append(full)
            else:
                raise ValueError(f'BA bounded at K={K} not recorded')
        return out
    entries = np.arange(len(d[a + '_key']))
    ranked = entries[np.lexsort((entries, -d[scol].astype(float)))]
    return [np.sort(ranked if K is None else ranked[:K]) for K in KSN]


def static_group(plist, K, how, bitu, U):
    idx = np.concatenate([p[0] for p in plist])
    sc = np.concatenate([p[1] for p in plist]).astype(float)
    first = np.full(U, np.iinfo(np.int64).max)
    np.minimum.at(first, idx, np.arange(len(idx)))
    if how == 'min':
        val = np.full(U, np.inf)
        np.minimum.at(val, idx, sc)
        key = val
    else:
        val = np.zeros(U)
        np.add.at(val, idx, sc)
        key = -val
    present_ = np.flatnonzero(first < np.iinfo(np.int64).max)
    sel = present_[np.lexsort((first[present_], key[present_]))]
    return bits_mask(bitu[sel if K is None else sel[:K]])


def identities(ders, a):
    uniq, inv = np.unique(np.concatenate([d[a + '_key'] for d in ders]), return_inverse=True)
    bitu = np.full(len(uniq), -1, np.int16)
    bitu[inv] = np.concatenate([d[a + '_bit'] for d in ders])
    starts = np.cumsum([0] + [len(d[a + '_key']) for d in ders])
    return inv, starts, bitu, len(uniq)


def div_inputs(ders):
    inv, starts, bitu, U = identities(ders, 'ba')
    npts = next((d['dir'].shape[1] for d in ders if d['dir'].size), 1)
    dirs = np.zeros((U, npts))
    pats = np.zeros(U, dtype=np.int64)
    for j, d in enumerate(ders):
        uj = inv[starts[j]:starts[j + 1]]
        for row, m in zip(d['dir'], d['dir_idx']):
            nanpat = np.isnan(row)
            dirs[uj[m]] = np.where(nanpat, 0.0, row)
            pats[uj[m]] = hash(nanpat.tobytes())
    return inv, starts, bitu, dirs, pats


def div_lists(d, KSN):
    nm = f'div_{DIVERSITY_THETA}_fit'
    if nm not in d['lists']:
        raise ValueError(f'diversity archive theta={DIVERSITY_THETA} not recorded')
    stored = {K: np.asarray(l, dtype=int) for K, l in zip(d['ksn'], d['lists'][nm])}
    offered = len(d['ba_key'])
    return [stored[K] if K in stored else (stored[None] if K >= offered else None) for K in KSN]


class GroupDiv:

    def __init__(self, theta, K, dirs, pats):
        self.theta, self.K, self.dirs, self.pats = theta, K, dirs, pats
        cap = K if K is not None else 1024
        self.cap, self.n = cap, 0
        self.id = np.full(cap, -1, np.int64)
        self.err = np.full(cap, np.inf)
        self.D = np.zeros((cap, dirs.shape[1]))
        self.P = np.zeros(cap, np.int64)
        self.slot = {}
        self.free = []

    def grow(self):
        c = self.cap * 2
        for name, fill in (('id', -1), ('err', np.inf), ('P', 0)):
            a = getattr(self, name)
            b = np.full(c, fill, a.dtype)
            b[:self.cap] = a
            setattr(self, name, b)
        D = np.zeros((c, self.D.shape[1]))
        D[:self.cap] = self.D
        self.D, self.cap = D, c

    def sims(self, u):
        n = self.n
        s = self.D[:n] @ self.dirs[u]
        s = np.where((self.P[:n] == self.pats[u]) & (self.id[:n] >= 0), s, NEG)
        s[self.id[:n] == u] = 1.0
        return s

    def remove(self, k):
        del self.slot[int(self.id[k])]
        self.id[k] = -1
        self.free.append(k)

    def add(self, u, err):
        if self.free:
            k = self.free.pop()
        else:
            if self.n == self.cap:
                self.grow()
            k = self.n
            self.n += 1
        self.id[k], self.err[k], self.P[k] = u, err, self.pats[u]
        self.D[k] = self.dirs[u]
        self.slot[u] = k

    def offer(self, u, err):
        if u in self.slot and not err < self.err[self.slot[u]]:
            return
        close = np.flatnonzero(self.sims(u) >= self.theta)
        if close.size:
            if np.all(err < self.err[close]):
                for k in close:
                    self.remove(k)
                self.add(u, err)
            return
        if self.K is None or len(self.slot) < self.K:
            self.add(u, err)
            return
        act = np.flatnonzero(self.id[:self.n] >= 0)
        e = self.err[act]
        top = act[e == e.max()]
        worst = top[np.argmax(self.id[top])]
        if err < self.err[worst]:
            self.remove(worst)
            self.add(u, err)

    def members(self):
        return np.fromiter(self.slot.keys(), int, len(self.slot))


def group_div(K, dirs, pats, lists):
    g = GroupDiv(DIVERSITY_THETA, K, dirs, pats)
    for idx, errs in lists:
        for u, err in zip(idx, errs):
            g.offer(int(u), float(err))
    return g.members()


def process(job):
    target, blobs, capacity_k, diversity, single = job
    KSN = ksn(capacity_k, single)
    base = KSN.index(single) if single is not None else -1
    ders = [load(b) for b in blobs]
    n = len(ders)
    res = {p: {} for p in ('one', 'union', 'dinf')}
    for a, (label, scol, how) in STATIC.items():
        inv, starts, bitu, U = identities(ders, a)
        perK = [[None] * n for _ in KSN]
        for j, d in enumerate(ders):
            uj = inv[starts[j]:starts[j + 1]]
            for ki, lst in enumerate(static_lists(d, a, scol, KSN)):
                perK[ki][j] = (uj[lst], d[scol][lst])
        for ki, K in enumerate(KSN):
            kk = 'inf' if K is None else str(K)
            res['one'].setdefault(label, {})[kk] = cover(target, [bits_mask(bitu[p[0]]) for p in perK[ki]])
            res['union'].setdefault(label, {})[kk] = cover(target, [bits_mask(bitu[np.unique(np.concatenate([p[0] for p in perK[ki]]))])])
            res['dinf'].setdefault(label, {})[kk] = cover(target, [static_group(perK[base], K, how, bitu, U)])
    if diversity:
        inv, starts, bitu, dirs, pats = div_inputs(ders)
        per = [div_lists(d, KSN) for d in ders]
        full = [(inv[starts[j]:starts[j + 1]][per[j][base]], ders[j]['ba_err'][per[j][base]]) for j in range(n)]
        for ki, K in enumerate(KSN):
            kk = 'inf' if K is None else str(K)
            if all(p[ki] is not None for p in per):
                lists = [(inv[starts[j]:starts[j + 1]][per[j][ki]], ders[j]['ba_err'][per[j][ki]]) for j in range(n)]
                res['one'].setdefault(DIVERSITY_LABEL, {})[kk] = cover(target, [bits_mask(bitu[idx]) for idx, _ in lists])
                res['union'].setdefault(DIVERSITY_LABEL, {})[kk] = cover(target, [bits_mask(bitu[group_div(None, dirs, pats, lists)])])
            else:
                res['one'].setdefault(DIVERSITY_LABEL, {})[kk] = None
                res['union'].setdefault(DIVERSITY_LABEL, {})[kk] = None
            res['dinf'].setdefault(DIVERSITY_LABEL, {})[kk] = cover(target, [bits_mask(bitu[group_div(K, dirs, pats, full)])])
    return res


def prefix(ders, K, ns, archives, order=None, single=None):
    if order is not None:
        ders = [ders[i] for i in order]
    out = {}
    for a in archives:
        if a == 'div':
            inv, starts, bitu, dirs, pats = div_inputs(ders)
            g = GroupDiv(DIVERSITY_THETA, K, dirs, pats)
            marks, masks = set(ns), []
            for j, d in enumerate(ders):
                full = div_lists(d, (single,))[0]
                for u, err in zip(inv[starts[j]:starts[j + 1]][full], d['ba_err'][full]):
                    g.offer(int(u), float(err))
                if j + 1 in marks:
                    masks.append(bits_mask(bitu[g.members()]))
            out[a] = masks
            continue
        label, scol, how = STATIC[a]
        inv, starts, bitu, U = identities(ders, a)
        plist = []
        for j, d in enumerate(ders):
            lst = static_lists(d, a, scol, (single,))[0]
            plist.append((inv[starts[j]:starts[j + 1]][lst], d[scol][lst]))
        out[a] = [static_group(plist[:n], K, how, bitu, U) for n in ns]
    return out


def _job(args):
    path, target, cid, pool, capacity_k, diversity, single = args
    return (cid, pool), process((target, V.study2_capacity_blobs(path, cid, pool), capacity_k, diversity, single))


def _prefix_job(args):
    path, cid, pool, K, ns, archives, orders, single = args
    ders = [load(b) for b in V.study2_capacity_blobs(path, cid, pool)]
    if orders is None:
        return (cid, pool), prefix(ders, K, ns, archives, None, single)
    return (cid, pool), [prefix(ders, K, ns, archives, order, single) for order in orders]


def complete_groups(path, target, pools=None, keep=None):
    size = int(V.meta(path)['params']['group_size'])
    return [(cid, pool) for cid, pool, n in V.study2_capacity_groups(path, target, pools) if n == size and (keep is None or cid in keep)]


def run_jobs(path, label, fn, jobs, workers=None):
    progress = Progress(f'{Path(path).stem} {label}', len(jobs), max(1, len(jobs) // 20))
    out = {}
    with ProcessPoolExecutor(max_workers=workers or config.WORKERS) as ex:
        for key, res in ex.map(fn, jobs):
            out[key] = res
            progress.step()
    progress.end()
    return out


def results(path, target, pools, keep=None, workers=None, diversity=False):
    capacity_k = V.meta(path)['identity']['capacity_k']
    jobs = [(str(path), target, cid, pool, capacity_k, diversity, single_of(path)) for cid, pool in complete_groups(path, target, pools, keep)]
    return run_jobs(path, 'capacity', _job, jobs, workers)


def prefix_masks(path, target, K, ns, archives, keep=None, orders=None, workers=None):
    key = f'prefix_{target}_{K}_' + hashlib.sha256(repr((tuple(ns), tuple(archives), keep and sorted(keep), orders)).encode()).hexdigest()[:10]

    def build():
        jobs = [(str(path), cid, pool, K, tuple(ns), tuple(archives), orders, single_of(path)) for cid, pool in complete_groups(path, target, None, keep)]
        if not jobs:
            raise ValueError(f'no complete {target} groups with capacity records')
        return run_jobs(path, f'final K={K}', _prefix_job, jobs, workers)
    return V.cached(key, path, build)


def aggregate(res, strategy_of, capacity_k, single=None):
    vals = {}
    KSN = ksn(capacity_k, single)
    for s in present(STRATS, [strategy_of[k[0]] for k in res]):
        keys = [k for k in res if strategy_of[k[0]] == s]
        for part in ('one', 'union', 'dinf'):
            for an in res[keys[0]][part]:
                for g in GOALS:
                    series = []
                    for K in KSN:
                        kk = 'inf' if K is None else str(K)
                        got = [res[k][part][an][kk] for k in keys]
                        series.append(float(np.mean([x[g] for x in got])) if all(x is not None for x in got) else None)
                    vals[f'{s}|{an}|{g}|{part}'] = series
        vals[f'{s}|groups'] = len(keys)
    vals['strategies'] = present(STRATS, [strategy_of[k[0]] for k in res])
    vals['ks'] = list(ks(capacity_k, single))
    return vals


def complete_pools(path, target):
    size = int(V.meta(path)['params']['group_size'])
    return tuple(sorted({pool for _, pool, n in V.study2_capacity_groups(path, target) if n == size}))


def keep_cases(path, target, filters):
    if not filters:
        return None
    cases = V.case_dicts(path, target)
    return set(V.where(pd.DataFrame([{'case_id': c, **{k: v[f] for k, f in V.CONFIG.items()}} for c, v in cases.items()]), filters).case_id)


def values(path, target, pools=None, filters=None, workers=None, diversity=False):
    pools = complete_pools(path, target) if pools is None else tuple(pools)

    def build():
        cases = V.case_dicts(path, target)
        res = results(path, target, pools, keep_cases(path, target, filters), workers, diversity)
        return aggregate(res, {cid: c['merge_strategy'] for cid, c in cases.items()}, V.meta(path)['identity']['capacity_k'], single_of(path))
    key = (f'capacity_{target}_' + '-'.join(map(str, pools)) + ('_div' if diversity else '')
           + ('' if not filters else '_' + hashlib.sha256(repr(sorted(filters.items())).encode()).hexdigest()[:6]))
    return V.cached(key, path, build)


LABELS = {'union': "capacity $K$ of each derivation's archives (group archive unbounded)",
          'dinf': 'capacity $K$ of the group archive (derivation archives unbounded)'}


def panels(vals, part, base, columns):
    from mgp.analysis import style
    from mgp.analysis.stats import SN
    from mgp.analysis.style import GOALS, INK2, LW, MEW, MS, MS_OPEN, W, plt
    style.use('grid')
    k = vals['ks']
    strategies = vals['strategies']
    pos = [float(np.log2(x)) for x in k] + [np.log2(k[-1]) + 1.8]
    fig, axes = plt.subplots(len(strategies), len(columns), figsize=(W, 3.0), sharex=True, sharey=True, squeeze=False)
    for i, s in enumerate(strategies):
        for j, an in enumerate(columns):
            ax = axes[i, j]
            for g, _, c, ls, mk in GOALS:
                v = np.asarray(vals[f'{s}|{an}|{g}|{part}'], dtype=float)
                ax.plot(pos[:-1], v[:-1], color=c, ls=ls, marker=mk, ms=MS, lw=LW)
                ax.plot(pos[-1], v[-1], color=c, marker=mk, ms=MS_OPEN, ls='none', mfc='white', mew=MEW)
            ax.axvline(np.log2(k[-1]) + 0.9, color=INK2, lw=0.5, ls=':')
            ticks = [x for x in (4, 64, 1024) if k[0] <= x <= k[-1]]
            ax.set_xticks([float(np.log2(x)) for x in ticks] + [pos[-1]], [str(x) if x < 1000 else f'{x // 1024}k' for x in ticks] + ['∞'])
            ax.minorticks_off()
            ax.set_ylim(-3, 103)
            ax.set_yticks([0, 50, 100])
            if i == 0:
                ax.set_title(TITLES.get(an, an), fontsize=8.5, fontweight='bold')
        axes[i, 0].set_ylabel(f'{SN[s]}\ncoverage (%)')
    fig.tight_layout(h_pad=0.4, w_pad=0.3, rect=(0, 0.03, 1, 1))
    fig.supxlabel(LABELS[part], y=0.0, va='bottom', fontsize=plt.rcParams['axes.labelsize'])
    style.goal_legend(fig, 0.995)
    style.save(fig, base)
