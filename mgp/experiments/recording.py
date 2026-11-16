import hashlib
import heapq
import io
import json
import math
import zlib

import numpy as np

RULES = ('fit', 'div')
_REC = {}
CONFIG = {'capacity_k': (), 'diversity_thetas': (), 'full': True}


def ksn():
    return tuple(CONFIG['capacity_k']) + (None,)


def names():
    return ['ba', 'bb', 'fb'] + [f'div_{th}_{ru}' for th in CONFIG['diversity_thetas'] for ru in RULES]


def khash(key):
    return int.from_bytes(hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest(), 'little', signed=True)


def direction(values):
    v = np.asarray(values, dtype=float)
    fin = np.isfinite(v)
    d = np.full(v.shape, np.nan)
    vals = v[fin]
    scale = np.max(np.abs(vals)) if vals.size else 0.0
    d[fin] = 0.0
    if scale:
        s = vals / scale
        d[fin] = s / np.linalg.norm(s)
    return d


class TopK:
    __slots__ = ('K', 'fit', 'ver', 'heap', 'n')

    def __init__(self, K):
        self.K, self.fit, self.ver, self.heap, self.n = K, {}, {}, [], 0

    def _push(self, key, f):
        self.n += 1
        self.fit[key], self.ver[key] = f, self.n
        heapq.heappush(self.heap, (-f, -self.n, key))

    def offer(self, key, f):
        cur = self.fit.get(key)
        if cur is not None:
            if f < cur:
                self._push(key, f)
            return
        if self.K is None or len(self.fit) < self.K:
            self._push(key, f)
            return
        while True:
            nf, nn, k = self.heap[0]
            if self.ver.get(k) == -nn:
                break
            heapq.heappop(self.heap)
        if f < -nf:
            heapq.heappop(self.heap)
            del self.fit[k], self.ver[k]
            self._push(key, f)


class Table:

    def __init__(self):
        self.idx, self.keys, self.err, self.dirs, self.vals, self.pat = {}, [], [], [], [], []
        self.S = np.zeros((0, 0))

    def offer(self, key, err, values):
        i = self.idx.get(key)
        if i is not None:
            if err < self.err[i]:
                self.err[i] = err
            return i
        i = len(self.keys)
        self.idx[key] = i
        d = direction(values)
        nanpat = np.isnan(d)
        self.keys.append(key)
        self.err.append(err)
        self.dirs.append(np.where(nanpat, 0.0, d))
        self.vals.append(np.asarray(values, dtype=float))
        self.pat.append(nanpat.tobytes())
        row = np.full(i + 1, -np.inf)
        if i:
            D = np.asarray(self.dirs[:i])
            same = np.fromiter((p == self.pat[i] for p in self.pat[:i]), bool, i)
            row[:i] = np.where(same, D @ self.dirs[i], -np.inf)
            eq = np.fromiter((np.array_equal(v, self.vals[i], equal_nan=True) for v in self.vals[:i]), bool, i)
            row[:i] = np.where(eq, 1.0, row[:i])
        row[i] = 1.0
        S = np.full((i + 1, i + 1), -np.inf)
        S[:i, :i] = self.S
        S[i, :] = row
        S[:, i] = row
        self.S = S
        return i


class DivArchive:
    __slots__ = ('theta', 'rule', 'K', 'mem')

    def __init__(self, theta, rule, K):
        self.theta, self.rule, self.K, self.mem = theta, rule, K, {}

    def offer(self, tab, i, err):
        mem = self.mem
        if mem:
            ids = np.fromiter(mem.keys(), int, len(mem))
            close = ids[tab.S[i, ids] >= self.theta]
        else:
            ids = close = np.zeros(0, int)
        if close.size:
            if all(err < mem[m] for m in close):
                for m in close:
                    del mem[m]
                mem[i] = err
            return
        if self.K is None or len(mem) < self.K:
            mem[i] = err
            return
        if self.rule == 'fit':
            worst = max(mem, key=lambda m: (mem[m], m))
            if err < mem[worst]:
                del mem[worst]
                mem[i] = err
            return
        sub = tab.S[np.ix_(ids, ids)].copy()
        np.fill_diagonal(sub, -np.inf)
        nn = sub.max(axis=1)
        top = np.flatnonzero(nn == nn.max())
        a = top[0]
        b = int(np.argmax(sub[a]))
        r = ids[a] if (mem[ids[a]], ids[a]) >= (mem[ids[b]], ids[b]) else ids[b]
        if tab.S[i, ids].max() < nn.max():
            del mem[r]
            mem[i] = err


def candidates(current, atomic_objects, constants=False):
    from mgp.engine.objects import SemanticList, SemanticObject, underlying_construction
    from mgp.engine.symbolic import symbolic_has_variables
    if isinstance(current, SemanticList):
        for item in current.alternatives:
            yield from candidates(item, atomic_objects, constants)
        return
    if not isinstance(current, SemanticObject):
        return
    if np.any(np.isinf(current.values)) or not np.any(np.isfinite(current.values)):
        return
    if underlying_construction(current).is_atomic and not atomic_objects:
        return
    finite = np.isfinite(current.values)
    values = current.values[finite]
    if not constants and ((len(values) > 1 or np.all(finite)) and np.all(values == values[0]) or not symbolic_has_variables(current)):
        return
    yield current


def install(capacity_k, diversity_thetas, full=True):
    CONFIG.update(capacity_k=tuple(capacity_k), diversity_thetas=tuple(diversity_thetas), full=full)
    from mgp.archives.observer import ArchiveOnlyObserver
    from mgp.engine.derivation import Derivation
    from mgp.engine.fitness import object_fitness
    from mgp.engine.symbolic import symbolic_data
    if getattr(Derivation, '_recording', False):
        return
    observe, finalize, obs_init = Derivation.observe_base_archives, Derivation.finalize_best_archive, ArchiveOnlyObserver.__init__

    def observe_base_archives(self, current):
        observe(self, current)
        if not _REC.get('on') or not CONFIG['full']:
            return
        archive = self.archive
        if not archive.symbolic:
            return
        for c in candidates(current, archive.atomic_objects, archive.constants):
            key = symbolic_data(c)[0]
            f = object_fitness(c, archive.target)
            if not math.isfinite(f):
                f = math.inf
            for sim in _REC['topk']:
                sim.offer(key, f)
            i = _REC['table'].offer(key, f, c.values)
            for a in _REC['div']:
                a.offer(_REC['table'], i, f)

    def finalize_best_archive(self):
        finalize(self)
        _REC['derivation'] = self

    def observer_init(self, *a, **k):
        obs_init(self, *a, **k)
        _REC['observer'] = self

    Derivation.observe_base_archives = observe_base_archives
    Derivation.finalize_best_archive = finalize_best_archive
    ArchiveOnlyObserver.__init__ = observer_init
    Derivation._recording = True


def start():
    _REC.clear()
    if not CONFIG['full']:
        _REC.update(on=True, derivation=None, observer=None)
        return
    _REC.update(on=True, derivation=None, observer=None, topk=[TopK(k) for k in ksn()], table=Table(),
                div=[DivArchive(th, ru, k) for th in CONFIG['diversity_thetas'] for ru in RULES for k in ksn()])


def blob_of(out):
    buf = io.BytesIO()
    np.savez(buf, **out)
    return zlib.compress(buf.getvalue(), 6)


def summary(r):
    m = r['metrics'] or {}
    return dict(merges=m.get('merges'), test_success=m.get('test_success'), symbolic_recovery=m.get('symbolic_recovery'), train_rmse=m.get('train_rmse'),
                test_rmse=m.get('test_rmse'), transition_sequence_hash=m.get('transition_sequence_hash'))


def archives_row(task, r):
    from mgp.engine.fitness import object_fitness
    from mgp.engine.symbolic import symbolic_data
    from mgp.engine.targets import catalog
    digests = catalog(task['target'])[0]
    bit = lambda key: -1 if (i := digests.get(hashlib.blake2b(key.encode('utf-8'), digest_size=16).digest())) is None else i
    d = _REC['derivation']
    keys = [symbolic_data(o)[0] for o in d.archive.objects]
    errs = [object_fitness(o, d.archive.target) for o in d.archive.objects]
    out = {'ba_key': np.array([khash(k) for k in keys], dtype=np.int64), 'ba_err': np.array([e if math.isfinite(e) else math.inf for e in errs], dtype=float),
           'ba_bit': np.array([bit(k) for k in keys], dtype=np.int16)}
    lists = {'ba': list(range(len(keys)))}
    for name, arch in (('bb', d.building_block_archive), ('fb', d.best_bb_archive)):
        entries = list(arch.entries) if arch is not None else []
        out[name + '_key'] = np.array([khash(e['equivalence_key']) for e in entries], dtype=np.int64)
        out[name + '_score'] = np.array([e['score'] for e in entries], dtype=float)
        out[name + '_bit'] = np.array([bit(e['equivalence_key']) for e in entries], dtype=np.int16)
        lists[name] = sorted(range(len(entries)), key=lambda j: (-entries[j]['score'], entries[j]['equivalence_key']))
    flat = lists['ba'] + lists['bb'] + lists['fb']
    out['members'] = np.array(flat, dtype=np.int32)
    out['offsets'] = np.cumsum([0, len(lists['ba']), len(lists['bb']), len(lists['fb'])]).astype(np.int64)
    out['names'] = np.array(['ba', 'bb', 'fb'])
    out['ksn'] = np.array([-1], dtype=np.int64)
    out['dir_idx'] = np.zeros(0, dtype=np.int32)
    out['dir'] = np.zeros((0, len(np.asarray(d.archive.target))), dtype=np.float32)
    return task['run_id'], 'ok', blob_of(out), json.dumps(dict(kind='archives', **summary(r))), None


def capacity_row(task, r):
    if not CONFIG['full']:
        return archives_row(task, r)
    from mgp.engine.targets import catalog
    from mgp.archives.codecs import archive_members
    from mgp.engine.symbolic import symbolic_data
    digests = catalog(task['target'])[0]
    KSN, NAMES = ksn(), names()

    def bit(key):
        i = digests.get(hashlib.blake2b(key.encode('utf-8'), digest_size=16).digest())
        return -1 if i is None else i

    def mask(idx, bits):
        m = 0
        for b in bits[np.asarray(idx, dtype=int)]:
            if b >= 0:
                m |= 1 << int(b)
        return m

    d, tab = _REC['derivation'], _REC['table']
    members = archive_members(r['archive_members'])
    stored = {n: sum(1 << i for dg, i in digests.items() if dg in members[n]) for n in ('base_fitness', 'bb_snapshot', 'best_bb')}
    out, checks, lists = {}, {}, {}
    out['ba_key'] = np.array([khash(k) for k in tab.keys], dtype=np.int64)
    out['ba_err'] = np.array(tab.err, dtype=float)
    out['ba_bit'] = np.array([bit(k) for k in tab.keys], dtype=np.int16)
    bound = d.archive.max_size
    engine_k = next(s for s in _REC['topk'] if s.K == bound)
    checks['ba_equals_archive'] = set(engine_k.fit) == {symbolic_data(o)[0] for o in d.archive.objects}
    lists['ba'] = [[tab.idx[k] for k in s.fit] for s in _REC['topk']]
    checks['ba_stored_mask'] = mask([tab.idx[k] for k in engine_k.fit], out['ba_bit']) == stored['base_fitness']
    similarity = getattr(d, 'base_archives', {}).get('similarity')
    if bound is not None and similarity is not None and similarity.similarity in CONFIG['diversity_thetas']:
        position = (CONFIG['diversity_thetas'].index(similarity.similarity) * len(RULES) + RULES.index('fit')) * len(KSN) + KSN.index(bound)
        checks['div_equals_archive'] = {tab.keys[m] for m in _REC['div'][position].mem} == {symbolic_data(o)[0] for o in similarity.objects}
    for name, arch, st in (('bb', d.building_block_archive, 'bb_snapshot'), ('fb', d.best_bb_archive, 'best_bb')):
        entries = list(getattr(arch, 'unbounded_entries', arch.entries)) if arch is not None else []
        ranked = sorted(range(len(entries)), key=lambda j: (-entries[j]['score'], entries[j]['equivalence_key']))
        out[name + '_key'] = np.array([khash(e['equivalence_key']) for e in entries], dtype=np.int64)
        out[name + '_score'] = np.array([e['score'] for e in entries], dtype=float)
        out[name + '_bit'] = np.array([bit(e['equivalence_key']) for e in entries], dtype=np.int16)
        lists[name] = [ranked[:k] if k else ranked for k in KSN]
        checks[name + '_stored_mask'] = mask(ranked if bound is None else ranked[:bound], out[name + '_bit']) == stored[st]
    divnames = NAMES[3:]
    for j, nm in enumerate(divnames):
        lists[nm] = [list(a.mem.keys()) for a in _REC['div'][j * len(KSN):(j + 1) * len(KSN)]]
    flat, offs = [], [0]
    for nm in NAMES:
        for lst in lists[nm]:
            flat.extend(lst)
            offs.append(len(flat))
    out['members'] = np.array(flat, dtype=np.int32)
    out['offsets'] = np.array(offs, dtype=np.int64)
    out['names'] = np.array(NAMES)
    out['ksn'] = np.array([-1 if k is None else k for k in KSN], dtype=np.int64)
    used = sorted({m for nm in divnames for lst in lists[nm] for m in lst})
    out['dir_idx'] = np.array(used, dtype=np.int32)
    npts = len(np.asarray(d.archive.target))
    out['dir'] = np.array([np.where(np.frombuffer(tab.pat[m], dtype=bool), np.nan, tab.dirs[m]) for m in used],
                          dtype=np.float32).reshape(len(used), npts)
    checks.update(kind='full', sizes={'ba': len(tab.keys), 'bb': len(out['bb_key']), 'fb': len(out['fb_key']), 'dirs': len(used)}, **summary(r))
    return task['run_id'], 'ok', blob_of(out), json.dumps(checks), None


def evaluated_row(task, r):
    keys = _REC['observer'].evaluated
    m = r['metrics']
    d = np.sort(np.fromiter((int.from_bytes(hashlib.blake2b(k.encode('utf-8'), digest_size=16).digest()[:8], 'little') for k in keys),
                            dtype=np.uint64, count=len(keys)))
    status, err = ('ok', None) if len(keys) == m['evaluated_unique'] else ('error', f"evaluated {len(keys)} != {m['evaluated_unique']}")
    return (task['run_id'], status, len(keys), zlib.compress(d.tobytes(), 1), m['transition_sequence_hash'],
            hashlib.sha256(r['archive_members']).hexdigest(), float(m.get('measured_cpu_seconds') or 0.0), err)


def load(blob):
    z = dict(np.load(io.BytesIO(zlib.decompress(blob))))
    names_, ks = [str(x) for x in z['names']], tuple(None if k < 0 else int(k) for k in z['ksn'])
    offs, flat = z['offsets'], z['members']
    lists, p = {}, 0
    for nm in names_:
        lists[nm] = []
        for _ in ks:
            lists[nm].append(flat[offs[p]:offs[p + 1]])
            p += 1
    z['lists'], z['ksn'] = lists, ks
    return z
