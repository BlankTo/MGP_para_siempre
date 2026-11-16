from config import DEFAULT
from mgp.experiments import campaign as C
from mgp.experiments import runner, timing

STUDY = 'study6'
SEED_STREAM = 'MGP_Study6_timing_only_v1'
TARGETS = ('nguyen_12', 'nguyen_2')
STRATEGIES = DEFAULT
INTERNAL_MERGE = DEFAULT
SIZES_AND_RATIOS = (((64, 256), ('1:4', '1:2', '1:1', '2:1', '4:1')), ((128,), ('1:2', '1:1', '2:1')))
ESEL = DEFAULT
ESUR = DEFAULT
MERGE_LIMIT = DEFAULT
REPETITIONS = 300
ARCHIVES = True
DIVERSITY_ARCHIVE = True
REVISIT_IC_VARIANTS = DEFAULT
REVISIT_IC_PAIRS = DEFAULT
REVISIT_IC_BUDGET = DEFAULT
CONSTRUCTIVE_MERGES = DEFAULT
COMPLETE_INTO_HOLE = DEFAULT
SINGLE_ARCHIVE_MAX_SIZE = DEFAULT
ARCHIVE_CONSTANTS = DEFAULT
SETTINGS = DEFAULT
WORKERS = DEFAULT
MEMORY_MB = DEFAULT
PROGRESS_EVERY = DEFAULT
CHUNK = 8
COLS = ('case_id', 'repetition', 'status', 'wall_seconds', 'cpu_seconds', 'cpu_search', 'cpu_ba', 'cpu_div', 'cpu_bba_record', 'cpu_bba_credit',
        'cpu_fbba', 'wall_ba', 'wall_div', 'wall_bba_record', 'wall_bba_credit', 'wall_fbba', 'termination', 'merges', 'operator_evaluations',
        'replay_evaluations', 'best_train_rmse', 'error')
SCHEMA = '''CREATE TABLE IF NOT EXISTS runs (case_id TEXT NOT NULL, repetition INTEGER NOT NULL, status TEXT NOT NULL, wall_seconds REAL,
    cpu_seconds REAL, cpu_search REAL, cpu_ba REAL, cpu_div REAL, cpu_bba_record REAL, cpu_bba_credit REAL, cpu_fbba REAL, wall_ba REAL, wall_div REAL,
    wall_bba_record REAL, wall_bba_credit REAL, wall_fbba REAL, termination TEXT, merges INTEGER, operator_evaluations INTEGER,
    replay_evaluations INTEGER, best_train_rmse REAL, error TEXT, PRIMARY KEY (case_id, repetition));'''


def configurations():
    levels = C.grid(globals())
    return [row for sizes, ratios in SIZES_AND_RATIOS for row in C.selected(dict(levels, sizes=sizes, ratios=ratios))]


def cases(target):
    from mgp.engine.protocol import stream_seed
    from mgp.experiments.plan import GRID_FIELDS, RUN_FACTORS, signature
    from mgp.experiments.plan import configurations as make_configurations
    opts = C.options(C.grid(globals()), C.settings(globals()), configurations())
    configs, _ = make_configurations(opts)
    out = []
    for config in sorted(configs, key=lambda c: tuple(str(c[k]) for k in GRID_FIELDS)):
        case = {k: config[k] for k in RUN_FACTORS}
        for k in ('operator_count', 'variable_atom_count', 'total_objects', 'config_id', 'operator_variable_ratio', 'actual_operator_variable_ratio'):
            case[k] = config[k]
        case.update(stage='confirmation', seed_family=SEED_STREAM, target=target, dataset_seed=stream_seed(opts['master_seed'], target, 'v42_inputs'),
                    dataset_replica=1, repetition=1, noise_level=0.0)
        case['case_id'] = signature(case)[:24]
        out.append(case)
    return out


def blocks():
    for target in C.grid(globals())['targets']:
        cs = cases(target)
        yield [(c, r) for r in range(1, REPETITIONS + 1) for c in cs]


def open_execution():
    levels, engine = C.grid(globals()), C.settings(globals())
    ident = runner.identity(STUDY, engine, grid=levels, sizes_and_ratios=SIZES_AND_RATIOS, archives=ARCHIVES, diversity_archive=DIVERSITY_ARCHIVE)
    extra = ('arch' if ARCHIVES else 'noarch', 'div' if DIVERSITY_ARCHIVE and ARCHIVES else '')
    ex = runner.Execution(STUDY, ident, dict(repetitions=REPETITIONS), SCHEMA, runner.token(dict(levels, sizes=None, ratios=None), extra, engine), every=3120)
    ex.cases([c for t in levels['targets'] for c in cases(t)])
    return ex


def completed(ex):
    done = ex.keys("SELECT case_id, repetition FROM runs WHERE status='ok'")
    return lambda t: (t[0]['case_id'], t[1]) in done


def init(memory_mb):
    C.init_worker(memory_mb)
    timing.setup(C.options(C.grid(globals()), C.settings(globals())))


def execute(task):
    return timing.measure(task, ARCHIVES, DIVERSITY_ARCHIVE)


def store(ex, row):
    ex.con.execute(f'INSERT OR REPLACE INTO runs VALUES ({",".join("?" * len(COLS))})', row)
    return row[2] == 'ok'


if __name__ == '__main__':
    runner.main(__file__)
