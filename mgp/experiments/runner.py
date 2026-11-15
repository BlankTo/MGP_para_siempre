import argparse
import hashlib
import importlib
import json
import platform
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config
from mgp.progress import Progress, duration, log

SCHEMA = 2
SHORT = {'g1': 'G1', 'g2': 'G2', 'nguyen_1': 'N1', 'nguyen_2': 'N2', 'nguyen_3': 'N3', 'nguyen_4': 'N4', 'nguyen_12': 'N12', 'standard': 'std',
         'revisit_ic': 'ric', 'constructive': 'con'}
ABBR = {'targets': 't', 'strategies': 's', 'internal_merge': 'im', 'sizes': 'n', 'ratios': 'r', 'esel': 'es', 'esur': 'eu', 'merge_limit': 'm',
        'datasets': 'd', 'noises': 'z'}
CASES = '''CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY, target TEXT NOT NULL, strategy TEXT, im REAL, size INTEGER, ratio TEXT, esel REAL,
    esur REAL, dataset INTEGER, noise REAL, training_size INTEGER, config_id TEXT, case_json TEXT NOT NULL);'''


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def plain(v):
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def signed(mask):
    mask = int(mask)
    return mask - (1 << 64) if mask >= 1 << 63 else mask


def word(v):
    if isinstance(v, float):
        return f'{v:g}'
    return SHORT.get(v, str(v)).replace(':', 'to')


def token(levels=None, extra=(), engine_settings=None):
    parts = []
    if levels is not None:
        for name, short in ABBR.items():
            v = levels.get(name)
            if v is not None and v != getattr(config, name.upper()):
                parts.append(short + '-'.join(word(x) for x in (v if isinstance(v, (tuple, list)) else (v,))))
    parts += [p for p in extra if p]
    s = engine_settings or {}
    if levels is None or 'revisit_ic' in levels['strategies']:
        budget = s.get('revisit_ic_budget')
        parts.append(f"ric{s.get('revisit_ic_variants')}x{s.get('revisit_ic_pairs')}" + ('' if budget is None else f'b{budget}'))
    if s.get('single_archive_max_size') is not None:
        parts.append(f"sa{s['single_archive_max_size']}")
    text = '_'.join(parts) or 'base'
    return text if len(text) <= 80 else text[:80]


def identity(study, engine_settings=None, campaign=True, **parts):
    out = dict(study=study, schema=SCHEMA, **parts)
    if campaign:
        from mgp.experiments.campaign import campaign as effective
        out.update(campaign=effective(engine_settings))
    return out


def case_row(case):
    get = lambda k: plain(case.get(k))
    return (case['case_id'], case['target'], get('merge_strategy'), get('internal_merge_probability'), get('total_objects'), get('operator_variable_ratio'),
            get('esel'), get('esur'), get('dataset_replica'), get('noise_level'), get('experimental_training_size'), get('config_id'),
            json.dumps(case, sort_keys=True, default=str))


class Execution:

    def __init__(self, study, ident, params, schema, token_text, every=500):
        self.study, self.identity, self.params = study, ident, params
        self.name = f'{study}__{token_text}__{digest(ident)[:8]}'
        self.path = config.RUNS / f'{self.name}.sqlite'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path, timeout=120)
        self.con.execute('PRAGMA journal_mode=WAL')
        self.con.execute('PRAGMA synchronous=NORMAL')
        self.con.executescript(CASES + schema)
        stored = self.con.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if stored is not None and json.loads(stored[0]) != json.loads(canonical(ident)):
            raise RuntimeError(f'identity mismatch in {self.path.name}')
        now = datetime.now(timezone.utc).isoformat(timespec='seconds')
        versions = {'python': platform.python_version()}
        for module in ('numpy', 'pandas', 'scipy', 'sympy', 'flint'):
            try:
                versions[module] = importlib.import_module(module).__version__
            except Exception:
                pass
        self.versions = canonical(versions)
        with self.con:
            for key, value in (('created', now), ('updated', now), ('params', canonical(params)), ('versions', self.versions)):
                self.con.execute('INSERT OR IGNORE INTO meta VALUES (?, ?)', (key, value))
            for key, value in (('name', self.name), ('study', study), ('identity', canonical(ident))):
                self.con.execute('INSERT OR REPLACE INTO meta VALUES (?, ?)', (key, value))
        self.every, self.n = every, 0

    def cases(self, cases):
        with self.con:
            self.con.executemany('INSERT OR IGNORE INTO cases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', [case_row(c) for c in cases])

    def keys(self, query):
        return {tuple(r) if len(r) > 1 else r[0] for r in self.con.execute(query)}

    def tick(self):
        self.n += 1
        if self.n % self.every == 0:
            self.con.commit()

    def close(self):
        if self.n:
            now = datetime.now(timezone.utc).isoformat(timespec='seconds')
            for key, value in (('updated', now), ('params', canonical(self.params)), ('versions', self.versions)):
                self.con.execute('INSERT OR REPLACE INTO meta VALUES (?, ?)', (key, value))
        self.con.commit()
        self.con.close()


def resources(module, workers=None, memory_mb=None):
    return workers or config.value(vars(module), 'WORKERS'), memory_mb or config.value(vars(module), 'MEMORY_MB')


def spread(items, n):
    items = list(items)
    if n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def sample(module, n):
    if hasattr(module, 'sample'):
        return module.sample(n)
    return spread([t for block in module.blocks() for t in block], n)


def run_sample(module, n, workers=None, memory_mb=None):
    workers, memory_mb = resources(module, workers, memory_mb)
    tasks = sample(module, n)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=module.init, initargs=(memory_mb,)) as ex:
        out = list(ex.map(module.execute, tasks))
    log(module.STUDY, 'test', tasks=len(out), workers=workers, elapsed=duration(time.perf_counter() - started))
    return tasks, out


def count(module, done):
    total = pending = 0
    for block in module.blocks():
        total += len(block)
        pending += sum(not done(t) for t in block)
    return total, pending


def run(module, workers=None, memory_mb=None):
    workers, memory_mb = resources(module, workers, memory_mb)
    store = module.open_execution()
    errors = 0
    progress = None
    try:
        done = module.completed(store)
        total, pending = count(module, done)
        log(store.name, 'start', tasks=total, done=total - pending, pending=pending, workers=workers, memory_mb=memory_mb)
        progress = Progress(store.name, pending, config.value(vars(module), 'PROGRESS_EVERY'))
        if pending:
            with ProcessPoolExecutor(max_workers=workers, initializer=module.init, initargs=(memory_mb,)) as ex:
                for block in module.blocks():
                    todo = [t for t in block if not done(t)]
                    for result in ex.map(module.execute, todo, chunksize=getattr(module, 'CHUNK', 2)):
                        errors += not module.store(store, result)
                        store.tick()
                        progress.step(errors=errors)
    finally:
        store.close()
    if progress is not None:
        progress.end(errors=errors, file=store.path.name)
    return errors == 0


def info(module):
    store = module.open_execution()
    try:
        total, pending = count(module, module.completed(store))
        log(store.name, 'info', tasks=total, done=total - pending, pending=pending, file=store.path)
    finally:
        store.close()


def main(path):
    sys.path.insert(0, str(Path(path).resolve().parent))
    module = importlib.import_module(Path(path).stem)
    parser = argparse.ArgumentParser(description=f'{module.STUDY} execution: run, extend, test or inspect')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--memory-mb', type=int, default=None)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--info', action='store_true')
    args = parser.parse_args()
    if args.info:
        info(module)
        return
    if args.test:
        run_sample(module, args.test, args.workers, args.memory_mb)
        return
    sys.exit(0 if run(module, args.workers, args.memory_mb) else 1)
