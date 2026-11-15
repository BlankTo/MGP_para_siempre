import traceback

from config import DEFAULT
from mgp.experiments import campaign as C
from mgp.experiments import recording, runner

STUDY = 'study2'
SEED_STREAM = 'campaign2026_study2'
TARGETS = ('nguyen_12', 'nguyen_2')
STRATEGIES = DEFAULT
INTERNAL_MERGE = DEFAULT
SIZES = (64, 128, 256)
RATIOS = ('1:2', '1:1', '2:1')
ESEL = DEFAULT
ESUR = DEFAULT
MERGE_LIMIT = DEFAULT
GROUP_SIZE = 1000
POOLS = 5
CAPACITY_K = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
DIVERSITY_THETAS = (0.99,)
REVISIT_IC_VARIANTS = DEFAULT
REVISIT_IC_PAIRS = DEFAULT
REVISIT_IC_BUDGET = DEFAULT
SINGLE_ARCHIVE_MAX_SIZE = DEFAULT
SETTINGS = DEFAULT
WORKERS = DEFAULT
MEMORY_MB = DEFAULT
PROGRESS_EVERY = DEFAULT
CHUNK = 2
BLOCK = 1440
SCHEMA = C.RUNS_SCHEMA + '''
CREATE TABLE IF NOT EXISTS capacity (run_id TEXT PRIMARY KEY, status TEXT NOT NULL, blob BLOB, checks TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS evaluated (run_id TEXT PRIMARY KEY, status TEXT NOT NULL, n_evaluated INTEGER, evaluated BLOB, transition_hash TEXT,
    archive_hash TEXT, cpu_seconds REAL, error TEXT);'''


def cases():
    return list(C.cells(C.grid(globals()), C.settings(globals())))


def recorded_k():
    single = C.settings(globals())['single_archive_max_size']
    return tuple(sorted({k for k in CAPACITY_K if single is None or k <= single} | ({single} if single is not None else set())))


def pool_tasks(target, pool, cs, options):
    ts = [C.make_task(c, STUDY, pool, index, options) for index in range(1, GROUP_SIZE + 1) for c in cs if c['target'] == target]
    ts.sort(key=lambda t: (t['index'], t['case']['case_id']))
    return ts


def blocks():
    cs, options = cases(), C.job_options(C.grid(globals()), C.settings(globals()))
    for target in C.grid(globals())['targets']:
        for pool in range(1, POOLS + 1):
            ts = pool_tasks(target, pool, cs, options)
            for i in range(0, len(ts), BLOCK):
                yield ts[i:i + BLOCK]


def sample(n):
    cs, options = cases(), C.job_options(C.grid(globals()), C.settings(globals()))
    out = []
    for target in C.grid(globals())['targets']:
        ts = pool_tasks(target, 1, cs, options)
        out += [ts[int(i * len(ts) / n)] for i in range(n)]
    return out


def open_execution():
    levels, engine = C.grid(globals()), C.settings(globals())
    ident = runner.identity(STUDY, engine, grid=levels, capacity_k=CAPACITY_K, diversity_thetas=DIVERSITY_THETAS)
    ex = runner.Execution(STUDY, ident, dict(group_size=GROUP_SIZE, pools=POOLS), SCHEMA, runner.token(levels, (), engine))
    ex.cases(cases())
    return ex


def completed(ex):
    runs = ex.keys("SELECT run_id FROM runs WHERE status='ok'")
    cap = ex.keys("SELECT run_id FROM capacity WHERE status='ok'")
    ev = ex.keys("SELECT run_id FROM evaluated WHERE status='ok'")
    return lambda t: t['run_id'] in runs and t['run_id'] in cap and t['run_id'] in ev


def init(memory_mb):
    C.init_worker(memory_mb)
    recording.install(recorded_k(), DIVERSITY_THETAS)


def execute(task):
    recording.start()
    r = C.execute(task)
    cap_row = ev_row = None
    if r['status'] == 'ok':
        try:
            cap_row = recording.capacity_row(task, r)
            ev_row = recording.evaluated_row(task, r)
        except Exception as exc:
            err = (f'{type(exc).__name__}: {exc} | ' + traceback.format_exc(limit=4))[:2000]
            cap_row = (task['run_id'], 'error', None, None, err)
            ev_row = (task['run_id'], 'error') + (None,) * 5 + (err,)
    return r, cap_row, ev_row


def store(ex, result):
    r, cap_row, ev_row = result
    ok = C.store(ex.con, r)
    if cap_row is not None:
        ex.con.execute('INSERT OR REPLACE INTO capacity VALUES (?,?,?,?,?)', cap_row)
        ok &= cap_row[1] == 'ok'
    if ev_row is not None:
        ex.con.execute('INSERT OR REPLACE INTO evaluated VALUES (?,?,?,?,?,?,?,?)', ev_row)
        ok &= ev_row[1] == 'ok'
    return ok


if __name__ == '__main__':
    runner.main(__file__)
