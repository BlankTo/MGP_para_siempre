import json
import traceback

import config
from config import DEFAULT
from mgp.experiments import campaign as C
from mgp.experiments import extended as X
from mgp.experiments import recording as R
from mgp.experiments import runner

STUDY = 'study7'
SEED_STREAM = 'MGP_noise3_training_audited_v3_coverage5'
TARGETS = ('g1', 'nguyen_12')
CONFIGURATIONS = {
    'standard': dict(total_objects=128, internal_merge_probability=0.5, operator_variable_ratio='1:1', esel=0.5, esur=0.5),
    'revisit_ic': dict(total_objects=128, internal_merge_probability=0.5, operator_variable_ratio='1:1', esel=0.5, esur=0.5),
    'constructive': dict(total_objects=128, internal_merge_probability=0.5, operator_variable_ratio='1:1', esel=0.0, esur=0.0),
}
MERGE_LIMIT = DEFAULT
GROUPS = 10
GROUP_SIZE = 5000
REVISIT_IC_VARIANTS = DEFAULT
REVISIT_IC_PAIRS = DEFAULT
REVISIT_IC_BUDGET = DEFAULT
SINGLE_ARCHIVE_MAX_SIZE = DEFAULT
FINAL_ARCHIVE_MAX_SIZE = DEFAULT
RECORD_SMALLER_LIMITS = False
CAPACITY_K = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
DIVERSITY_THETAS = (0.99,)
SETTINGS = DEFAULT
WORKERS = DEFAULT
MEMORY_MB = DEFAULT
PROGRESS_EVERY = DEFAULT
CHUNK = 1
BLOCK = 5000
SCHEMA = '''CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, pool INTEGER NOT NULL, repetition INTEGER NOT NULL,
    status TEXT NOT NULL, built INTEGER, num INTEGER, sym REAL, train_rmse REAL, test_rmse REAL, metrics_json TEXT, archive_ids BLOB, error TEXT);
CREATE INDEX IF NOT EXISTS runs_case ON runs(case_id, pool, repetition);
CREATE TABLE IF NOT EXISTS capacity (run_id TEXT PRIMARY KEY, status TEXT NOT NULL, blob BLOB, checks TEXT, error TEXT);'''


def levels():
    return dict(C.grid(globals()), strategies=tuple(sorted(CONFIGURATIONS)), noises=(0.0,), datasets=1)


def recording():
    if RECORD_SMALLER_LIMITS:
        return 'full'
    return 'archives' if config.value(globals(), 'FINAL_ARCHIVE_MAX_SIZE') is not None else None


def recorded_k():
    single = C.settings(globals())['single_archive_max_size']
    return tuple(sorted({k for k in CAPACITY_K if single is None or k <= single} | ({single} if single is not None else set())))


def plan():
    from mgp.experiments.plan import cases as make_cases
    from mgp.experiments.plan import configurations as make_configurations
    from mgp.experiments.plan import signature
    rows = [dict(merge_strategy=strategy, **cfg) for strategy, cfg in sorted(CONFIGURATIONS.items())]
    opts = C.options(levels(), C.settings(globals()), rows)
    configs, skipped = make_configurations(opts)
    if skipped:
        raise RuntimeError(f'invalid configurations {skipped}')
    cases = []
    for base in make_cases(opts, configs):
        case = {k: v for k, v in base.items() if k != 'case_id'}
        case.update(noise_level=0.0, experimental_training_size=100)
        case['case_id'] = signature(case)[:24]
        cases.append(case)
    return C.job_options(levels(), C.settings(globals())), cases


def blocks():
    opts, cases = plan()
    block = []
    for t in X.tasks(STUDY, cases, opts, GROUP_SIZE, GROUPS, True):
        block.append(t)
        if len(block) == BLOCK:
            yield block
            block = []
    if block:
        yield block


def sample(n):
    opts, cases = plan()
    out = []
    for i in range(n):
        rep = 1 + (i * GROUP_SIZE) // n
        ts = list(X.tasks(STUDY, cases, opts, rep, GROUPS, True, first=rep))
        out.append(ts[(i * 7) % len(ts)])
    return out


def open_execution():
    engine = C.settings(globals())
    ident = runner.identity(STUDY, engine, grid=levels(), configurations=CONFIGURATIONS)
    ex = runner.Execution(STUDY, ident, dict(groups=GROUPS, group_size=GROUP_SIZE), SCHEMA, runner.token(levels(), (), engine))
    ex.cases(plan()[1])
    return ex


def completed(ex):
    done = ex.keys("SELECT run_id FROM runs WHERE status='ok'")
    mode = recording()
    kinds = {'full': {'full'}, 'archives': {'full', 'archives'}}.get(mode)
    cap = {rid for rid, checks in ex.con.execute("SELECT run_id, checks FROM capacity WHERE status='ok'") if json.loads(checks).get('kind') in kinds} \
        if mode else None
    return lambda t: t['run_id'] in done and (cap is None or t['run_id'] in cap)


def init(memory_mb):
    C.init_worker(memory_mb)
    mode = recording()
    if mode == 'full':
        R.install(recorded_k(), DIVERSITY_THETAS)
    elif mode == 'archives':
        R.install((), (), full=False)


def execute(task):
    if not recording():
        return X.execute(task), None
    R.start()
    row = X.execute(task)
    cap = None
    if row[4] == 'ok':
        try:
            cap = R.capacity_row(task, {'archive_members': row[6], 'metrics': json.loads(row[5])})
        except Exception as exc:
            cap = (task['run_id'], 'error', None, None, (f'{type(exc).__name__}: {exc} | ' + traceback.format_exc(limit=4))[:2000])
    return row, cap


def store(ex, result):
    row, cap = result
    rid, cid, pool, rep, status, metrics, blob, _, err = row
    m = json.loads(metrics) if metrics else {}
    get = lambda k: runner.plain(m.get(k))
    ex.con.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                   (rid, cid, pool, rep, status, get('evaluated_target_ever'), get('test_success'), get('symbolic_recovery'), get('train_rmse'),
                    get('test_rmse'), metrics, blob, err))
    ok = status == 'ok'
    if cap is not None:
        ex.con.execute('INSERT OR REPLACE INTO capacity VALUES (?,?,?,?,?)', cap)
        ok &= cap[1] == 'ok'
    return ok


if __name__ == '__main__':
    runner.main(__file__)
