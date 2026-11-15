import json

from config import DEFAULT
from mgp.experiments import campaign as C
from mgp.experiments import labels, runner

STUDY = 'study5'
SEED_STREAM = 'campaign2026_study1'
TARGETS = DEFAULT
STRATEGIES = DEFAULT
INTERNAL_MERGE = DEFAULT
SIZES = DEFAULT
RATIOS = DEFAULT
ESEL = DEFAULT
ESUR = DEFAULT
MERGE_LIMIT = DEFAULT
REPETITIONS = 10
CONDITIONS = ('clean', 'random_ic', 'pure_noise', 'inverted', 'inverted_all')
REVISIT_IC_VARIANTS = DEFAULT
REVISIT_IC_PAIRS = DEFAULT
REVISIT_IC_BUDGET = DEFAULT
SINGLE_ARCHIVE_MAX_SIZE = DEFAULT
SETTINGS = DEFAULT
WORKERS = DEFAULT
MEMORY_MB = DEFAULT
PROGRESS_EVERY = DEFAULT
CHUNK = 4
FIELDS = ('built', 'retained', 'numerical', 'symbolic', 'merges', 'termination', 'ic_decisions', 'evaluated_unique', 'jaccard', 'train_rmse', 'hash',
          'selected', 'base_fitness_entries', 'bb_snapshot_entries', 'best_bb_entries', 'base_fitness_composite', 'bb_snapshot_composite')
SCHEMA = '''CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, repetition INTEGER NOT NULL, status TEXT NOT NULL,
    error TEXT);
CREATE TABLE IF NOT EXISTS conditions (run_id TEXT NOT NULL, condition TEXT NOT NULL, position INTEGER NOT NULL, built INTEGER, retained INTEGER,
    numerical INTEGER, symbolic REAL, merges INTEGER, termination TEXT, ic_decisions INTEGER, evaluated_unique INTEGER, jaccard REAL, train_rmse REAL,
    hash TEXT, selected INTEGER, base_fitness_entries INTEGER, bb_snapshot_entries INTEGER, best_bb_entries INTEGER, base_fitness_composite REAL,
    bb_snapshot_composite REAL, ba_mask INTEGER, bba_mask INTEGER, fbba_mask INTEGER, result TEXT NOT NULL, PRIMARY KEY (run_id, condition));'''


def cases():
    return list(C.cells(C.grid(globals()), C.settings(globals())))


def blocks():
    options, cs = C.job_options(C.grid(globals()), C.settings(globals())), cases()
    for rep in range(1, REPETITIONS + 1):
        yield [C.make_task(c, STUDY, 1, rep, options) for c in cs]


def open_execution():
    levels, engine = C.grid(globals()), C.settings(globals())
    ex = runner.Execution(STUDY, runner.identity(STUDY, engine, grid=levels), dict(repetitions=REPETITIONS, conditions=CONDITIONS), SCHEMA,
                          runner.token(levels, (), engine))
    ex.cases(cases())
    return ex


def completed(ex):
    have = {}
    for run_id, cond in ex.con.execute('SELECT run_id, condition FROM conditions'):
        have.setdefault(run_id, set()).add(cond)
    need = set(CONDITIONS)
    return lambda t: need <= have.get(t['run_id'], set())


def init(memory_mb):
    C.init_worker(memory_mb)
    labels.install()


def execute(task):
    return task['run_id'], task['case']['case_id'], task['index'], labels.derive(task, CONDITIONS)


def store(ex, result):
    run_id, case_id, rep, out = result
    ex.con.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)', (run_id, case_id, rep, 'ok', None))
    for position, (cond, v) in enumerate(out.items()):
        ex.con.execute(f'INSERT OR REPLACE INTO conditions VALUES ({",".join("?" * (len(FIELDS) + 7))})',
                       (run_id, cond, position, *[runner.plain(v.get(k)) for k in FIELDS], *[runner.signed(v[a]) for a in labels.ARCH],
                        json.dumps(v)))
    return True


if __name__ == '__main__':
    runner.main(__file__)
