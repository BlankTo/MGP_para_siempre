import hashlib
import json
import random
from fractions import Fraction
from itertools import product

from mgp.engine.derivation import Derivation
from mgp.engine.protocol import stream_seed

FACTORS = ('operator_count', 'variable_atom_count', 'esel', 'esur', 'fitness_probability', 'semantic_list_enabled', 'merge_strategy', 'internal_merge_probability')
SINGLE_STAGES = ('profiles', 'noise_profiles')


def signature(value):
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True, allow_nan=False).iterencode(value):
        digest.update(chunk.encode())
    return digest.hexdigest()


def paired_seed(options, job, purpose, index):
    if options['version'] in (42, 43, 44):
        context = f"v42_{job['seed_family']}_{job['dataset_replica']}_{purpose}"
        return stream_seed(options['master_seed'], job['target'], context, index)
    context = f"v{options['version']}_{job['stage']}_{job['dataset_seed']}_{job['repetition']}_{purpose}"
    return stream_seed(options['master_seed'], job['target'], context, index)


def excluded_elitism(config, options):
    flag = options.get('skip_esel_esur_one', False)
    if type(flag) is not bool:
        raise ValueError('skip_esel_esur_one must be a boolean')
    return flag and config['esel'] == 1 and (config['esur'] == 1)


RUN_FACTORS = (*FACTORS, 'max_merge', 'esur_in_IM', 'IM_terminate_on_one')


def compatible(config):
    if config['merge_strategy'] == 'revisit_ic' and config['semantic_list_enabled']:
        return 'revisit_ic_does_not_support_semantic_lists'
    if config['merge_strategy'] == 'constructive' and config['semantic_list_enabled']:
        return 'constructive_does_not_create_semantic_lists'
    return None


def validate_config(config):
    for name in ('operator_count', 'variable_atom_count', 'max_merge'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(name + ' must be a positive integer')
    Derivation([1.0], config, 0)


GRID_FIELDS = ('total_objects', 'operator_variable_ratio', 'esel', 'esur', 'merge_strategy', 'internal_merge_probability')


def object_counts(total, ratio):
    if type(total) is not int or total < 2:
        raise ValueError('total_objects must be an integer >= 2')
    try:
        a, b = map(int, ratio.split(':'))
        if a < 1 or b < 1:
            raise ValueError()
    except (ValueError, AttributeError):
        raise ValueError(f'invalid operator:variable ratio {ratio!r}') from None
    operators = (2 * total * a + a + b) // (2 * (a + b))
    if not 0 < operators < total:
        raise ValueError(f'ratio {ratio} leaves no operators or no variables')
    return (operators, total - operators)


def configurations(options):
    base = options['settings']
    result, skipped, seen = [], {}, set()
    for row in options['selected_configurations']:
        operators, variables = object_counts(row['total_objects'], row['operator_variable_ratio'])
        config = {**base, **row, 'operator_count': operators, 'variable_atom_count': variables}
        if excluded_elitism(config, options):
            skipped['esel_esur_one'] = skipped.get('esel_esur_one', 0) + 1
            continue
        reason = compatible(config)
        if reason:
            raise ValueError(reason)
        validate_config(config)
        if config['semantic_list_enabled'] or config['fitness_probability'] != 1.0:
            raise ValueError('semantic_list_enabled must be False and fitness_probability 1.0')
        factors = {name: config[name] for name in RUN_FACTORS}
        identifier = signature(factors)[:16]
        if identifier in seen:
            raise ValueError('duplicate configuration after ratio rounding')
        seen.add(identifier)
        actual = Fraction(operators, variables)
        result.append({**factors, 'config_id': identifier, 'total_objects': operators + variables, 'operator_variable_ratio': row['operator_variable_ratio'],
                       'actual_operator_variable_ratio': f'{actual.numerator}:{actual.denominator}'})
    if not result:
        raise ValueError('no configurations')
    return (result, skipped)


def cases(options, configs):
    for target, replica in product(options['targets'], range(1, options['datasets_per_target'] + 1)):
        ordered = list(configs)
        random.Random(stream_seed(options['master_seed'], target, 'v42_case_order', replica)).shuffle(ordered)
        for config, noise in product(ordered, options['noise_levels']):
            case = {**config, 'stage': options['kind'], 'seed_family': options['seed_family'], 'target': target, 'dataset_seed': stream_seed(options['master_seed'], target, 'v42_inputs'), 'dataset_replica': replica, 'repetition': replica, 'noise_level': noise}
            case['case_id'] = signature(case)[:24]
            yield case


def job_for(case, index):
    job = {**case, 'policy': 'observe', 'derivations': 1, 'capacity': None, 'shortlist': 20, 'injected': 0, 'restart_index': index, 'families': [case['stage']], 'settings_overrides': {name: case[name] for name in RUN_FACTORS}}
    job['job_id'] = signature(job)[:24]
    return job
