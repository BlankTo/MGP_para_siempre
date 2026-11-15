import json

from config import DEFAULT
from mgp.experiments import campaign as C
from mgp.experiments import runner

STUDY = 'study3'
SEED_STREAM = 'campaign2026_study3'
TARGETS = DEFAULT
STRATEGIES = DEFAULT
INTERNAL_MERGE = DEFAULT
SIZES = (128, 256)
RATIOS = ('1:1',)
ESEL = (0.5,)
ESUR = (0.5,)
MERGE_LIMIT = DEFAULT
DATASETS = 50
REPETITIONS = 60
REVISIT_IC_VARIANTS = DEFAULT
REVISIT_IC_PAIRS = DEFAULT
REVISIT_IC_BUDGET = DEFAULT
SINGLE_ARCHIVE_MAX_SIZE = DEFAULT
SETTINGS = DEFAULT
WORKERS = DEFAULT
MEMORY_MB = DEFAULT
PROGRESS_EVERY = DEFAULT
CHUNK = 4
BLOCK_REPETITIONS = 5
DATASET_STREAM = 'campaign2026_input_dataset'
ARCHIVES = ('base_fitness', 'bb_snapshot', 'best_bb')
SCHEMA = '''CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, repetition INTEGER NOT NULL, status TEXT NOT NULL,
    built INTEGER, num INTEGER, sym REAL, train_rmse REAL, merges INTEGER, ba_mask INTEGER, bba_mask INTEGER, fbba_mask INTEGER, result TEXT, error TEXT);
CREATE INDEX IF NOT EXISTS runs_case ON runs(case_id, repetition);'''


def cases():
    return list(C.cells(C.grid(globals()), C.settings(globals()), DATASET_STREAM))


def make(case, index, options):
    return {'run_id': C.digest([STUDY, case['case_id'], 1, index])[:32], 'study': STUDY, 'case': case, 'pool': 1, 'index': index,
            'restart_index': index, 'options': options, 'target': case['target'], 'dataset_replica': case['dataset_replica'],
            'noise_level': case['noise_level']}


def blocks():
    options, cs = C.job_options(C.grid(globals()), C.settings(globals())), cases()
    for k in range(1, REPETITIONS + 1, BLOCK_REPETITIONS):
        yield [make(c, i, options) for i in range(k, min(k + BLOCK_REPETITIONS, REPETITIONS + 1)) for c in cs]


def sample(n):
    options, cs = C.job_options(C.grid(globals()), C.settings(globals())), cases()
    return [make(cs[(i * 997) % len(cs)], 1 + (i * REPETITIONS) // n, options) for i in range(n)]


def open_execution():
    levels, engine = C.grid(globals()), C.settings(globals())
    ex = runner.Execution(STUDY, runner.identity(STUDY, engine, grid=levels), dict(repetitions=REPETITIONS), SCHEMA, runner.token(levels, (), engine))
    ex.cases(cases())
    return ex


def completed(ex):
    done = ex.keys("SELECT run_id FROM runs WHERE status='ok'")
    return lambda t: t['run_id'] in done


def init(memory_mb):
    C.init_worker(memory_mb)


def execute(t):
    from mgp.archives.codecs import archive_members
    from mgp.engine.targets import catalog
    r = C.execute(t)
    c = t['case']
    if r['status'] != 'ok':
        return t['run_id'], c['case_id'], t['index'], 'error', None, (r['error'] or '')[:500]
    digests = catalog(t['target'])[0]
    members = archive_members(r['archive_members'])
    m = r['metrics'] or {}
    row = {a: sum(1 << i for d, i in digests.items() if d in members[a]) for a in ARCHIVES}
    row.update({k: m.get(k) for k in ('test_success', 'symbolic_recovery', 'evaluated_target_ever', 'retained_target_ever', 'train_rmse', 'merges')})
    row.update(strategy=c['merge_strategy'], im=c['internal_merge_probability'], size=c['total_objects'])
    return t['run_id'], c['case_id'], t['index'], 'ok', json.dumps(row, default=float), None


def store(ex, result):
    run_id, case_id, rep, status, payload, error = result
    row = json.loads(payload) if payload else {}
    get = lambda k: runner.plain(row.get(k))
    masks = [runner.signed(row[a]) if a in row else None for a in ARCHIVES]
    ex.con.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                   (run_id, case_id, rep, status, get('evaluated_target_ever'), get('test_success'), get('symbolic_recovery'), get('train_rmse'),
                    get('merges'), *masks, payload, error))
    return status == 'ok'


if __name__ == '__main__':
    runner.main(__file__)
