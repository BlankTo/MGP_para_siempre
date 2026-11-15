import io
import json
import random

import numpy as np

import config
from config import DEFAULT
from mgp.experiments import campaign as C
from mgp.experiments import runner, workspace_counts

STUDY = 'workspace'
TARGET = 'g1'
SIZES = (64, 128, 256)
RATIOS = DEFAULT
MERGE_LIMIT = DEFAULT
SEED = 1
REPETITIONS = 1000
CHUNK_DERIVATIONS = 50
SETTINGS = DEFAULT
WORKERS = DEFAULT
MEMORY_MB = DEFAULT
PROGRESS_EVERY = 100
CHUNK = 1
SCHEMA = '''CREATE TABLE IF NOT EXISTS chunks (size INTEGER NOT NULL, ratio TEXT NOT NULL, start INTEGER NOT NULL, count INTEGER NOT NULL,
    operators INTEGER, variables INTEGER, total BLOB NOT NULL, lengths TEXT NOT NULL, terminations TEXT NOT NULL, PRIMARY KEY (size, ratio, start));'''


def base():
    return {**config.ENGINE, 'protocol_version': 1, 'archive_enabled': False, 'building_block_archive_enabled': False, 'best_bb_archive_enabled': False,
            'sympy_enabled': False, 'max_merge': config.value(globals(), 'MERGE_LIMIT'), 'fitness_probability': 1.0, 'semantic_list_enabled': False,
            **config.value(globals(), 'SETTINGS')}


def blocks():
    out = []
    for size in SIZES:
        for ratio in config.value(globals(), 'RATIOS'):
            gen = random.Random(SEED)
            seeds = [(gen.getrandbits(63), gen.getrandbits(63)) for _ in range(REPETITIONS)]
            out += [(size, ratio, i, seeds[i:i + CHUNK_DERIVATIONS]) for i in range(0, REPETITIONS, CHUNK_DERIVATIONS)]
    yield out


def open_execution():
    ratios = config.value(globals(), 'RATIOS')
    ident = runner.identity(STUDY, campaign=False, engine=base(), target=TARGET, sizes=SIZES, ratios=ratios, seed=SEED, chunk=CHUNK_DERIVATIONS)
    extra = ('t' + runner.word(TARGET), 'n' + '-'.join(str(s) for s in SIZES), f'seed{SEED}')
    return runner.Execution(STUDY, ident, dict(repetitions=REPETITIONS), SCHEMA, '_'.join(extra))


def completed(ex):
    done = ex.keys('SELECT size, ratio, start, count FROM chunks')
    return lambda t: (t[0], t[1], t[2], len(t[3])) in done


def init(memory_mb):
    C.init_worker(memory_mb)


def execute(task):
    return workspace_counts.count(task, base(), TARGET)


def store(ex, result):
    size, ratio, start, count, n_o, n_v, total, lengths, terminations = result
    buf = io.BytesIO()
    np.save(buf, total)
    ex.con.execute('INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?)',
                   (size, ratio, start, count, n_o, n_v, buf.getvalue(), json.dumps(lengths), json.dumps(terminations)))
    return True


if __name__ == '__main__':
    runner.main(__file__)
