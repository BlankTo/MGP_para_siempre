import argparse
import json
import sqlite3
from pathlib import Path

import config

RANGES = ('target', 'strategy', 'im', 'size', 'ratio', 'esel', 'esur', 'dataset', 'noise', 'training_size')


def ro(path):
    return sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro', uri=True, timeout=60)


def line(key, value):
    print(f'  {key:<12} {value}')


def describe(path, detail):
    con = ro(path)
    meta = {k: v for k, v in con.execute('SELECT key, value FROM meta')}
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    print(meta.get('name', path.stem))
    line('study', meta.get('study'))
    line('created', meta.get('created'))
    line('updated', meta.get('updated'))
    line('params', meta.get('params'))
    cases = con.execute('SELECT COUNT(*) FROM cases').fetchone()[0]
    if cases:
        cols = [c[1] for c in con.execute('PRAGMA table_info(cases)')]
        line('cases', cases)
        for col in RANGES:
            if col in cols:
                vals = [v for (v,) in con.execute(f'SELECT DISTINCT {col} FROM cases WHERE {col} IS NOT NULL ORDER BY {col}')]
                if vals:
                    line(col, ','.join(str(v) for v in vals))
    if 'runs' in tables:
        cols = [c[1] for c in con.execute('PRAGMA table_info(runs)')]
        line('runs', ' '.join(f'{status}={n}' for status, n in con.execute('SELECT status, COUNT(*) FROM runs GROUP BY status')) or 'none')
        if 'repetition' in cols:
            lo, hi = con.execute("SELECT MIN(repetition), MAX(repetition) FROM runs WHERE status='ok'").fetchone()
            line('repetitions', f'{lo}-{hi}')
            if 'pool' in cols:
                line('pools', ','.join(str(p) for (p,) in con.execute("SELECT DISTINCT pool FROM runs WHERE status='ok' ORDER BY pool")))
    for extra in ('capacity', 'evaluated'):
        if extra in tables:
            line(extra, ' '.join(f'{a}={b}' for a, b in con.execute(f'SELECT status, COUNT(*) FROM {extra} GROUP BY status')) or 'none')
    if 'conditions' in tables:
        rows = con.execute('SELECT condition, COUNT(*) FROM conditions GROUP BY condition ORDER BY MIN(position)').fetchall()
        line('conditions', ' '.join(f'{a}={b}' for a, b in rows) or 'none')
    if 'chunks' in tables:
        rows = con.execute('SELECT size, ratio, SUM(count) FROM chunks GROUP BY size, ratio').fetchall()
        line('derivations', ' '.join(f'{s}|{r}={n}' for s, r, n in rows) or 'none')
    if detail:
        line('identity', json.dumps(json.loads(meta['identity']), sort_keys=True))
    con.close()


def main():
    p = argparse.ArgumentParser(description='saved executions: configuration, tested ranges, progress')
    p.add_argument('study', nargs='?', default='')
    p.add_argument('--detail', action='store_true')
    args = p.parse_args()
    paths = sorted(config.RUNS.glob(f'{args.study}*.sqlite'))
    if not paths:
        print(f'executions none runs={config.RUNS}')
    for path in paths:
        describe(path, args.detail)


if __name__ == '__main__':
    main()
