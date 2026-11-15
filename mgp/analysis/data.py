import hashlib
import io
import json
import pickle
import sqlite3
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import config
from mgp.progress import Progress

MAGIC = b'MGPMETR2\x00'
ARCHIVES = ('base_fitness', 'bb_snapshot', 'best_bb')
SCALARS = ('test_success', 'symbolic_recovery', 'train_success', 'evaluated_target_ever', 'retained_target_ever', 'first_retained_target_merge',
           'test_nrmse', 'test_rmse', 'train_rmse', 'clean_rmse', 'merges', 'termination', 'operator_evaluations', 'fitness_evaluations',
           'ic_decisions', 'ms_decisions', 'internal_merge_count', 'external_merge_count', 'evaluated_unique', 'retained_unique',
           'expression_depth', 'improvements', 'last_improvement_merge', 'search_cpu_seconds', 'measured_cpu_seconds', 'observation_cpu_seconds',
           'workspace_seed', 'retained_component_coverage', 'retained_cooccurrence_coverage', 'retained_composite_coverage')
PER_ARCHIVE = ('entries', 'component_coverage', 'cooccurrence_coverage', 'composite_coverage', 'clean_nrmse_best', 'symbolic_unique')
CONFIG = {'strategy': 'merge_strategy', 'im': 'internal_merge_probability', 'size': 'total_objects', 'ratio': 'operator_variable_ratio',
          'esel': 'esel', 'esur': 'esur'}
FACTORS = tuple(CONFIG)
LEVEL_ORDER = {'strategy': ['standard', 'revisit_ic', 'constructive'], 'ratio': ['1:4', '1:2', '1:1', '2:1', '4:1'],
               'target': ['g1', 'nguyen_1', 'nguyen_2', 'nguyen_3', 'nguyen_4', 'g2', 'nguyen_12']}


def ro(path):
    return sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro', uri=True, timeout=120)


def meta(path):
    con = ro(path)
    out = {k: v for k, v in con.execute('SELECT key, value FROM meta')}
    con.close()
    for k in ('identity', 'params', 'versions'):
        if k in out:
            out[k] = json.loads(out[k])
    return out


def executions(study):
    found = []
    for path in sorted(config.RUNS.glob(f'{study}__*.sqlite')):
        try:
            found.append((meta(path).get('updated', ''), path))
        except sqlite3.Error:
            continue
    return [p for _, p in sorted(found)]


def resolve(study, name=None):
    if name:
        p = Path(name)
        for cand in (p, config.RUNS / name, config.RUNS / f'{name}.sqlite'):
            if cand.suffix == '.sqlite' and cand.exists():
                return cand
        raise FileNotFoundError(f'no execution {name} in {config.RUNS}')
    found = executions(study)
    if not found:
        raise FileNotFoundError(f'no {study} execution in {config.RUNS}')
    return found[-1]


def stamp(path):
    path = Path(path)
    return max(p.stat().st_mtime for p in (path, Path(str(path) + '-wal')) if p.exists())


def cached(key, path, build):
    folder = config.WORK / 'cache'
    folder.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(f'{Path(path).resolve()}|{key}'.encode()).hexdigest()[:12]
    target = folder / f'{Path(path).stem}__{key}__{tag}.pkl'
    if target.exists() and target.stat().st_mtime >= stamp(path):
        with open(target, 'rb') as fh:
            return pickle.load(fh)
    value = build()
    with open(target, 'wb') as fh:
        pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return value


def where(frame, filters):
    if not filters:
        return frame
    m = np.ones(len(frame), dtype=bool)
    for key, values in filters.items():
        if key not in frame.columns:
            raise KeyError(f'cannot filter on {key}: columns are {list(frame.columns)}')
        col = frame[key]
        vals = [float(v) if pd.api.types.is_float_dtype(col) and not isinstance(v, str) else v for v in values]
        m &= col.isin(vals).to_numpy()
    return frame[m].reset_index(drop=True)


def levels(frame, key):
    present = list(pd.unique(frame[key]))
    order = LEVEL_ORDER.get(key)
    if order:
        return [v for v in order if v in present] + sorted(v for v in present if v not in order)
    return sorted(present)


def metrics(blob):
    return json.loads(zlib.decompress(blob[len(MAGIC):]))


@lru_cache(maxsize=None)
def digests(target):
    from mgp.engine.targets import catalog
    return tuple(catalog(target)[0].items())


@lru_cache(maxsize=None)
def selected_ids(target):
    from mgp.engine.targets import subtarget_specs
    from mgp.engine.symbolic import expression_key
    return {hashlib.sha256(expression_key(text).encode('utf-8')).hexdigest()[:32]: i for i, text, _ in subtarget_specs(target, 2)}


def masks(target, members):
    from mgp.archives.codecs import archive_members
    mem = archive_members(members)
    return {a: np.uint64(sum(1 << i for d, i in digests(target) if d in mem[a])) for a in ARCHIVES}


def case_dicts(path, target=None):
    con = ro(path)
    query = 'SELECT case_id, case_json FROM cases' + (' WHERE target=?' if target else '')
    out = {cid: json.loads(raw) for cid, raw in con.execute(query, (target,) if target else ())}
    con.close()
    return out


def campaign_row(case, pool, rep, m):
    out = {'case_id': case['case_id'], 'config_id': case['config_id'], 'target': case['target'], **{k: case[v] for k, v in CONFIG.items()},
           'dataset_replica': case['dataset_replica'], 'pool': pool, 'rep': rep}
    out.update({k: m.get(k) for k in SCALARS})
    out['selected_nodes'] = (m.get('expression_operators') or 0) + (m.get('expression_leaves') or 0) if m.get('expression_operators') is not None else None
    for archive in ARCHIVES:
        out.update({f'{archive}_{k}': m.get(f'{archive}_{k}') for k in PER_ARCHIVE})
    return out


def as_masks(frame):
    for a in ARCHIVES:
        frame[a] = frame[a].astype(np.uint64)
    return frame


def loading(path, con, query, args=()):
    total = con.execute(f'SELECT COUNT(*) FROM ({query})', args).fetchone()[0]
    return Progress(f'{Path(path).stem} load', total, max(1, total // 10))


def study1(path):
    def build():
        cases = case_dicts(path)
        con = ro(path)
        recs = []
        query = "SELECT case_id, pool, repetition, metrics, archive_members FROM runs WHERE status='ok'"
        progress = loading(path, con, query)
        for cid, pool, rep, blob, members in con.execute(query):
            progress.step()
            case = cases[cid]
            m = metrics(blob)
            r = campaign_row(case, pool, rep, m)
            key = m.get('selected_key_id')
            r['selected'] = -2 if key is None else selected_ids(case['target']).get(key, -1)
            r.update(masks(case['target'], members))
            recs.append(r)
        con.close()
        return as_masks(pd.DataFrame(recs))
    return cached('study1', path, build)


def study2(path, target):
    def build():
        cases = case_dicts(path, target)
        con = ro(path)
        recs = []
        query = ("SELECT r.case_id, r.pool, r.repetition, r.built, r.num, r.sym, r.archive_members FROM runs r JOIN cases c USING(case_id) "
                 "WHERE r.status='ok' AND c.target=?")
        progress = loading(path, con, query, (target,))
        for cid, pool, rep, built, num, sym, members in con.execute(query, (target,)):
            progress.step()
            case = cases[cid]
            r = {'case_id': cid, 'target': case['target'], **{k: case[v] for k, v in CONFIG.items()}, 'pool': pool, 'rep': rep, 'built': built,
                 'num': num, 'sym': int(sym == 1)}
            r.update(masks(case['target'], members))
            recs.append(r)
        con.close()
        return as_masks(pd.DataFrame(recs))
    return cached(f'study2_{target}', path, build)


def study2_evaluated(path, target, pools=None):
    con = ro(path)
    where = '' if pools is None else f" AND r.pool IN ({','.join('?' * len(pools))})"
    rows = con.execute("SELECT r.case_id, r.pool, r.repetition, e.evaluated FROM evaluated e JOIN runs r USING(run_id) JOIN cases c USING(case_id) "
                       f"WHERE e.status='ok' AND c.target=?{where} ORDER BY r.case_id, r.pool, r.repetition", (target, *(pools or ()))).fetchall()
    con.close()
    return rows


def study2_capacity_groups(path, target, pools=None):
    con = ro(path)
    where = '' if pools is None else f" AND r.pool IN ({','.join('?' * len(pools))})"
    keys = con.execute("SELECT r.case_id, r.pool, COUNT(*) FROM capacity k JOIN runs r USING(run_id) JOIN cases c USING(case_id) "
                       f"WHERE k.status='ok' AND c.target=?{where} GROUP BY r.case_id, r.pool", (target, *(pools or ()))).fetchall()
    con.close()
    return keys


def study2_capacity_blobs(path, case_id, pool):
    con = ro(path)
    out = [b for (b,) in con.execute("SELECT k.blob FROM capacity k JOIN runs r USING(run_id) WHERE r.case_id=? AND r.pool=? AND k.status='ok' "
                                     'ORDER BY r.repetition', (case_id, pool))]
    con.close()
    return out


def study3(path):
    def build():
        con = ro(path)
        rows = con.execute("SELECT r.case_id, c.config_id, c.target, c.dataset, r.repetition, r.result FROM runs r JOIN cases c USING(case_id) "
                           "WHERE r.status='ok'").fetchall()
        con.close()
        frame = pd.DataFrame(rows, columns=['case_id', 'config_id', 'target', 'dataset_replica', 'rep', 'result'])
        res = pd.DataFrame([json.loads(r) for r in frame.result])
        return pd.concat([frame.drop(columns='result'), res], axis=1)
    return cached('study3', path, build)


def condition(text):
    try:
        return float(text)
    except ValueError:
        return text


def study4(path):
    def build():
        cases = case_dicts(path)
        con = ro(path)
        cols = [d[1] for d in con.execute('PRAGMA table_info(conditions)')][3:]
        runs = {}
        query = ('SELECT r.run_id, r.case_id, r.repetition, k.* FROM runs r JOIN conditions k USING(run_id) '
                 'ORDER BY r.rowid, k.position')
        for row in con.execute(query):
            run_id, cid, rep, cond = row[0], row[1], row[2], row[4]
            c = cases[cid]
            r = runs.get(run_id)
            if r is None:
                r = runs[run_id] = dict(target=c['target'], config_id=c['config_id'], rep=rep, esel=c['esel'], esur=c['esur'],
                                        strategy=c['merge_strategy'], im=c['internal_merge_probability'], size=c['total_objects'], res={})
            r['res'][condition(cond)] = dict(zip(cols, row[6:]))
        con.close()
        return list(runs.values())
    return cached('study4', path, build)


def study5(path):
    def build():
        con = ro(path)
        rows = con.execute('SELECT r.case_id, c.target, r.repetition, k.condition, k.result FROM runs r JOIN conditions k USING(run_id) '
                           'JOIN cases c USING(case_id) ORDER BY r.rowid, k.position').fetchall()
        con.close()
        out = {}
        for cid, target, rep, cond, result in rows:
            out.setdefault((cid, target, rep), {})[cond] = json.loads(result)
        return pd.DataFrame([(cid, t, rep, res) for (cid, t, rep), res in out.items()], columns=['case_id', 'target', 'rep', 'result'])
    return cached('study5', path, build)


def study6(path):
    def build():
        cases = case_dicts(path)
        con = ro(path)
        runs = pd.read_sql("SELECT * FROM runs WHERE status='ok'", con)
        con.close()
        runs['target'] = runs.case_id.map(lambda c: cases[c]['target'])
        for k, v in CONFIG.items():
            runs[k] = runs.case_id.map(lambda c: cases[c][v])
        return runs
    return cached('study6', path, build)


def study6_seconds(frame, time='search_and_archives', diversity=False):
    if time == 'search':
        return frame.cpu_search
    return frame.cpu_seconds if diversity else frame.cpu_seconds - frame.cpu_div


def study7(path):
    def build():
        from mgp.archives.codecs import unpack_identity_sets
        cases = case_dicts(path)
        con = ro(path)
        recs = []
        query = "SELECT case_id, pool, repetition, metrics_json, archive_ids FROM runs WHERE status='ok'"
        progress = loading(path, con, query)
        for cid, pool, rep, mj, blob in con.execute(query):
            progress.step()
            c, m = cases[cid], json.loads(mj)
            ids = unpack_identity_sets(blob)
            r = {'case_id': cid, 'target': c['target'], 'strategy': c['merge_strategy'], 'pool': pool, 'rep': rep, 'built': m['evaluated_target_ever'],
                 'num': m['test_success'], 'sym': int(m['symbolic_recovery'] == 1), 'train_rmse': m['train_rmse'], 'test_rmse': m['test_rmse']}
            r.update({a: np.uint64(sum(1 << i for d, i in digests(c['target']) if d in ids[a])) for a in ARCHIVES})
            recs.append(r)
        con.close()
        return as_masks(pd.DataFrame(recs))
    return cached('study7', path, build)


def study8(path):
    def build():
        con = ro(path)
        rows = con.execute("SELECT r.run_id, r.case_id, c.config_id, c.target, c.training_size, r.repetition, r.result FROM runs r "
                           "JOIN cases c USING(case_id) WHERE r.status='ok'").fetchall()
        con.close()
        frame = pd.DataFrame(rows, columns=['run_id', 'case_id', 'config_id', 'target', 'training_size', 'rep', 'result'])
        res = pd.DataFrame([json.loads(r) for r in frame.result])
        return pd.concat([frame.drop(columns='result'), res], axis=1)
    return cached('study8', path, build)


def workspace(path):
    con = ro(path)
    rows = con.execute('SELECT size, ratio, start, count, total FROM chunks ORDER BY size, ratio, start').fetchall()
    con.close()
    acc = {}
    for size, ratio, start, count, blob in rows:
        a = acc.setdefault(f'{size}|{ratio}', {'next': 0, 'total': None, 'n': 0})
        if start != a['next']:
            continue
        total = np.load(io.BytesIO(blob))
        a['total'] = np.zeros_like(total) if a['total'] is None else a['total']
        a['total'] += total
        a['n'] += count
        a['next'] = start + count
    return {k: a['total'] / a['n'] for k, a in acc.items()}, {k: a['n'] for k, a in acc.items()}
