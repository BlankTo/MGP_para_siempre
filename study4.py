from config import DEFAULT
from mgp.experiments import campaign as C
from mgp.experiments import noise_search, runner

STUDY = 'study4'
SEED_STREAM = 'campaign2026_study4'
TARGETS = DEFAULT
STRATEGIES = DEFAULT
INTERNAL_MERGE = DEFAULT
SIZES = (128, 256)
RATIOS = ('1:1',)
ESEL = DEFAULT
ESUR = DEFAULT
MERGE_LIMIT = DEFAULT
REPETITIONS = 100
CONDITIONS = (0.0, 0.01, 0.1, 1.0, 10.0, 'pure_noise')
REVISIT_IC_VARIANTS = DEFAULT
REVISIT_IC_PAIRS = DEFAULT
REVISIT_IC_BUDGET = DEFAULT
SINGLE_ARCHIVE_MAX_SIZE = DEFAULT
SETTINGS = DEFAULT
WORKERS = DEFAULT
MEMORY_MB = DEFAULT
PROGRESS_EVERY = DEFAULT
CHUNK = 2
FIELDS = ('built', 'numerical', 'symbolic', 'hash', 'evaluated', 'merges', 'first_divergence', 'jaccard', 'contained', 'ic', 'ic_real', 'ic_equiv',
          'elite', 'elite_real', 'elite_equiv', 'first_ic_real', 'first_ic_equiv', 'first_elite_real', 'first_elite_equiv')
SCHEMA = '''CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, repetition INTEGER NOT NULL, status TEXT NOT NULL,
    error TEXT);
CREATE TABLE IF NOT EXISTS conditions (run_id TEXT NOT NULL, condition TEXT NOT NULL, position INTEGER NOT NULL, built INTEGER, numerical INTEGER,
    symbolic REAL, hash TEXT, evaluated INTEGER, merges INTEGER, first_divergence INTEGER, jaccard REAL, contained REAL, ic INTEGER, ic_real INTEGER,
    ic_equiv INTEGER, elite INTEGER, elite_real INTEGER, elite_equiv INTEGER, first_ic_real INTEGER, first_ic_equiv INTEGER, first_elite_real INTEGER,
    first_elite_equiv INTEGER, PRIMARY KEY (run_id, condition));'''


def cases():
    return [c for c in C.cells(C.grid(globals()), C.settings(globals())) if c['noise_level'] == 0.0]


def blocks():
    options, cs = C.job_options(C.grid(globals()), C.settings(globals())), cases()
    for index in range(1, REPETITIONS + 1):
        yield [{'run_id': C.digest([STUDY, c['case_id'], 1, index])[:32], 'study': STUDY, 'case': c, 'pool': 1, 'index': index,
                'restart_index': index, 'options': options, 'target': c['target'], 'dataset_replica': c['dataset_replica'],
                'noise_level': c['noise_level']} for c in cs]


def open_execution():
    levels, engine = C.grid(globals()), C.settings(globals())
    ex = runner.Execution(STUDY, runner.identity(STUDY, engine, grid=levels), dict(repetitions=REPETITIONS, conditions=CONDITIONS), SCHEMA,
                          runner.token(levels, (), engine))
    ex.cases(cases())
    return ex


def label(cond):
    return cond if isinstance(cond, str) else repr(float(cond))


def completed(ex):
    have = {}
    for run_id, cond in ex.con.execute('SELECT run_id, condition FROM conditions'):
        have.setdefault(run_id, set()).add(cond)
    need = {label(c) for c in CONDITIONS}
    return lambda t: need <= have.get(t['run_id'], set())


def init(memory_mb):
    C.init_worker(memory_mb)
    noise_search.install()


def execute(task):
    return dict(noise_search.derive(task, CONDITIONS), run_id=task['run_id'], case_id=task['case']['case_id'])


def store(ex, r):
    ex.con.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)', (r['run_id'], r['case_id'], r['rep'], 'ok', None))
    for position, (cond, v) in enumerate(r['res'].items()):
        ex.con.execute(f'INSERT OR REPLACE INTO conditions VALUES ({",".join("?" * (3 + len(FIELDS)))})',
                       (r['run_id'], label(cond), position, *[runner.plain(v[k]) for k in FIELDS]))
    return True


if __name__ == '__main__':
    runner.main(__file__)
