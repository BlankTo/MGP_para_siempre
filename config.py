from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'
OUTPUT = ROOT / 'output'
WORK = ROOT / 'work'

DEFAULT = type('Default', (), {'__repr__': lambda self: 'DEFAULT', '__reduce__': lambda self: 'DEFAULT'})()

WORKERS = 4
MEMORY_MB = 5000
PROGRESS_EVERY = 1000
PROGRESS_SECONDS = 600

TARGETS = ('g1', 'g2', 'nguyen_1', 'nguyen_2', 'nguyen_3', 'nguyen_4', 'nguyen_12')
STRATEGIES = ('standard', 'revisit_ic', 'constructive')
INTERNAL_MERGE = (0.0, 0.5)
SIZES = (64, 256)
RATIOS = ('1:4', '1:2', '1:1', '2:1', '4:1')
ESEL = (0.0, 0.5)
ESUR = (0.0, 0.5)
MERGE_LIMIT = 500
DATASETS = 1
NOISES = (0.0,)

REVISIT_IC_VARIANTS = 4
REVISIT_IC_PAIRS = 16
REVISIT_IC_BUDGET = None
CONSTRUCTIVE_MERGES = ('incomplete+semantic', 'operator+semantic', 'operator+operator')
COMPLETE_INTO_HOLE = False

SINGLE_ARCHIVE_MAX_SIZE = None
FINAL_ARCHIVE_MAX_SIZE = None
ARCHIVE_CONSTANTS = False

SETTINGS = {}

ENGINE = {'protocol_version': 2, 'bb_credit_version': 2, 'subtarget_catalog_version': 2, 'test_seed': 2, 'test_sample_count': 1000,
          'base_archive_policies': ['similarity', 'symbolic'], 'archive_comparison_method': 'cosine', 'archive_comparison_parameters': {},
          'seed': None, 'dataset_seed': 1, 'noise_level': 0.0, 'noise_seed': 1, 'semantic_list_enabled': False,
          'building_block_archive_enabled': True, 'best_bb_archive_enabled': True, 'best_bb_alpha': 0.25, 'best_bb_decay': 0.9,
          'building_block_root_weight': 0.25, 'building_block_depth_decay': 0.9, 'target': 'g1', 'sample_count': 100, 'sample_min': None,
          'sample_max': None, 'variable_atom_count': 50, 'operator_count': 20, 'unary_operators_on': False, 'binary_operators_on': True,
          'ternary_operators_on': False, 'fitness_probability': 1.0, 'internal_merge_probability': 0.0, 'esur_in_IM': False,
          'IM_terminate_on_one': True, 'perfect_fit_tolerance': 1e-08, 'max_merge': 1000, 'esel': 0.0, 'esur': 0.0, 'archive_enabled': True,
          'archive_similarity': 0.99, 'archive_atomic_objects': False, 'sympy_enabled': True, 'equivalence_method': 'rmse',
          'equivalence_tolerance': 1e-10, 'equivalence_atol': 1e-10, 'equivalence_rtol': 1e-10}

CAMPAIGN = {'version': 44, 'master_seed': 4201, 'test_stream': 'confirmation', 'skip_esel_esur_one': True,
            'settings': {**ENGINE, 'dataset_sampling': 'latin_hypercube', 'sample_count': 100, 'test_sample_count': 1000, 'max_merge': 500,
                         'fitness_probability': 1.0, 'semantic_list_enabled': False, 'fixed_noise_across_repetitions': True, 'esur_in_IM': False,
                         'IM_terminate_on_one': True, 'research_wall_limit_seconds': None, 'workspace_checkpoint_interval': 64,
                         'count_archive_diagnostics': True, 'defer_building_block_expressions': True, 'defer_building_block_credit': True,
                         'campaign_compact_tie_break': True}}

REFERENCE = dict(strategy='standard', im=0.0, size=64, ratio='1:1', esel=0.0, esur=0.0)
INPUT = None
WHERE = {}
GROUP_TARGET = 'nguyen_12'
TIME = 'search_and_archives'
NOVELTY_POOLS = None
CAPACITY_POOLS = None
CAPACITY_COLUMNS = ('BA', 'BBA', 'FBBA')
DIVERSITY = False


def value(scope, name):
    v = scope.get(name, DEFAULT)
    return globals()[name] if v is DEFAULT else v
