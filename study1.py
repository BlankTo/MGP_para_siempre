from config import DEFAULT
from mgp.experiments import campaign as C
from mgp.experiments import runner

STUDY = 'study1'
SEED_STREAM = 'campaign2026_study1'
TARGETS = DEFAULT
STRATEGIES = DEFAULT
INTERNAL_MERGE = DEFAULT
SIZES = DEFAULT
RATIOS = DEFAULT
ESEL = DEFAULT
ESUR = DEFAULT
MERGE_LIMIT = DEFAULT
REPETITIONS = 500
REVISIT_IC_VARIANTS = DEFAULT
REVISIT_IC_PAIRS = DEFAULT
REVISIT_IC_BUDGET = DEFAULT
SINGLE_ARCHIVE_MAX_SIZE = DEFAULT
SETTINGS = DEFAULT
WORKERS = DEFAULT
MEMORY_MB = DEFAULT
PROGRESS_EVERY = DEFAULT
CHUNK = 2


def cases():
    return list(C.cells(C.grid(globals()), C.settings(globals())))


def blocks():
    options, cs = C.job_options(C.grid(globals()), C.settings(globals())), cases()
    for rep in range(1, REPETITIONS + 1):
        yield [C.make_task(c, STUDY, 1, rep, options) for c in cs]


def sample(n):
    options, cs = C.job_options(C.grid(globals()), C.settings(globals())), cases()
    return [C.make_task(cs[(i * 997) % len(cs)], STUDY, 1, 1 + (i * REPETITIONS) // n, options) for i in range(n)]


def open_execution():
    levels, engine = C.grid(globals()), C.settings(globals())
    ex = runner.Execution(STUDY, runner.identity(STUDY, engine, grid=levels), dict(repetitions=REPETITIONS), C.RUNS_SCHEMA, runner.token(levels, (), engine))
    ex.cases(cases())
    return ex


def completed(ex):
    done = ex.keys("SELECT run_id FROM runs WHERE status='ok'")
    return lambda t: t['run_id'] in done


def init(memory_mb):
    C.init_worker(memory_mb)


def execute(task):
    return C.execute(task)


def store(ex, result):
    return C.store(ex.con, result)


if __name__ == '__main__':
    runner.main(__file__)
