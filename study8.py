import json

from config import DEFAULT
from mgp.experiments import campaign as C
from mgp.experiments import extended as X
from mgp.experiments import runner

STUDY = 'study8'
SEED_STREAM = 'MGP_noise3_training_audited_v3_training'
TARGETS = DEFAULT
STRATEGIES = DEFAULT
DESIGNS = ((128, 0.0, 0.0, 0.0), (128, 0.5, 0.0, 0.5), (256, 0.0, 0.5, 0.0), (256, 0.5, 0.5, 0.5))
RATIO = '1:1'
MERGE_LIMIT = DEFAULT
TRAINING_SIZES = (20, 50, 100)
REPETITIONS = 100
REVISIT_IC_VARIANTS = DEFAULT
REVISIT_IC_PAIRS = DEFAULT
REVISIT_IC_BUDGET = DEFAULT
SINGLE_ARCHIVE_MAX_SIZE = DEFAULT
SETTINGS = DEFAULT
WORKERS = DEFAULT
MEMORY_MB = DEFAULT
PROGRESS_EVERY = DEFAULT
CHUNK = 2
ARCH = ('base_fitness', 'bb_snapshot', 'best_bb')
SCHEMA = '''CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, repetition INTEGER NOT NULL, status TEXT NOT NULL,
    built INTEGER, num INTEGER, sym REAL, train_rmse REAL, test_rmse REAL, ba_mask INTEGER, bba_mask INTEGER, fbba_mask INTEGER, result TEXT, error TEXT);
CREATE INDEX IF NOT EXISTS runs_case ON runs(case_id, repetition);'''


def plan():
    from mgp.experiments.plan import cases as make_cases
    from mgp.experiments.plan import configurations as make_configurations
    from mgp.experiments.plan import signature
    levels = C.grid(globals())
    rows = [dict(total_objects=size, operator_variable_ratio=RATIO, esel=esel, esur=esur, merge_strategy=strategy, internal_merge_probability=im)
            for strategy in levels['strategies'] for size, im, esel, esur in DESIGNS]
    opts = C.options(levels, C.settings(globals()), rows)
    configs, skipped = make_configurations(opts)
    if skipped:
        raise RuntimeError(f'invalid configurations {skipped}')
    cases = []
    for base in make_cases(opts, configs):
        for samples in TRAINING_SIZES:
            case = {k: v for k, v in base.items() if k != 'case_id'}
            case.update(noise_level=0.0, experimental_training_size=samples)
            case['case_id'] = signature(case)[:24]
            cases.append(case)
    return C.job_options(levels, C.settings(globals())), cases


def blocks():
    opts, cases = plan()
    yield list(X.tasks(STUDY, cases, opts, REPETITIONS, 1, False))


def open_execution():
    levels, engine = C.grid(globals()), C.settings(globals())
    ident = runner.identity(STUDY, engine, grid=levels, designs=DESIGNS, ratio=RATIO, training_sizes=TRAINING_SIZES)
    extra = ('ts' + '-'.join(str(n) for n in TRAINING_SIZES),)
    ex = runner.Execution(STUDY, ident, dict(repetitions=REPETITIONS), SCHEMA, runner.token(dict(levels, sizes=None, ratios=None), extra, engine))
    ex.cases(plan()[1])
    return ex


def completed(ex):
    done = ex.keys("SELECT run_id FROM runs WHERE status='ok'")
    return lambda t: t['run_id'] in done


def init(memory_mb):
    C.init_worker(memory_mb)


def execute(t):
    from mgp.archives.codecs import unpack_identity_sets
    from mgp.engine.targets import catalog
    rid, cid, pool, rep, status, metrics, blob, _, err = X.execute(t)
    c = t['case']
    if status != 'ok':
        return rid, cid, rep, 'error', None, err
    m = json.loads(metrics)
    ids = unpack_identity_sets(blob)
    digests = catalog(t['target'])[0]
    row = {a: sum(1 << i for d, i in digests.items() if d in ids[a]) for a in ARCH}
    row['sizes'] = {a: len(ids[a]) for a in ARCH}
    row.update({k: m.get(k) for k in ('test_success', 'symbolic_recovery', 'evaluated_target_ever', 'retained_target_ever', 'train_rmse', 'test_rmse',
                                      'merges', 'transition_sequence_hash', 'selected_key', 'actual_training_size')})
    row.update(strategy=c['merge_strategy'], im=c['internal_merge_probability'], size=c['total_objects'], esel=c['esel'], esur=c['esur'])
    return rid, cid, rep, 'ok', json.dumps(row, default=float), None


def store(ex, result):
    rid, cid, rep, status, payload, err = result
    row = json.loads(payload) if payload else {}
    get = lambda k: runner.plain(row.get(k))
    masks = [runner.signed(row[a]) if a in row else None for a in ARCH]
    ex.con.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                   (rid, cid, rep, status, get('evaluated_target_ever'), get('test_success'), get('symbolic_recovery'), get('train_rmse'),
                    get('test_rmse'), *masks, payload, err))
    return status == 'ok'


if __name__ == '__main__':
    runner.main(__file__)
