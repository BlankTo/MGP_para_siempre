import inspect
import random

import numpy as np

from mgp.engine.fitness import readonly_target
from mgp.engine.objects import BINARY_OPERATORS, UNARY_OPERATORS, OperatorObject, SemanticObject
from mgp.engine.objects import target_zero_underflow
from mgp.engine.protocol import current_protocol, noisy_labels, stream_seed
from mgp.engine.targets import TARGET_DOMAINS


def _domain_result(result):
    if isinstance(result, tuple):
        return tuple((_domain_result(item) for item in result))
    if isinstance(result, (np.ndarray, np.floating)) and np.issubdtype(result.dtype, np.floating):
        values = np.asarray(result)
        return np.where(np.isfinite(values), values, np.nan).view(_TargetValues)
    return result


class _TargetValues(np.ndarray):

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == 'at':
            raise ValueError('target functions must not update input samples in place')
        operands = tuple((np.asarray(value) if isinstance(value, _TargetValues) else value for value in inputs))
        outputs = kwargs.get('out', ())
        if outputs:
            kwargs['out'] = tuple((np.asarray(value) if isinstance(value, _TargetValues) else value for value in outputs))
            operands = tuple((value.copy() if isinstance(value, np.ndarray) else value for value in operands))
        result = getattr(ufunc, method)(*operands, **kwargs)
        if np.any(target_zero_underflow(ufunc, method, operands, result, kwargs)):
            raise ValueError('numerical underflow to zero while generating target labels')
        if method == '__call__' and ufunc in (np.power, np.float_power):
            undefined = np.isnan(operands[0]) | np.isnan(operands[1])
            undefined = undefined & kwargs.get('where', True)
            result = np.where(undefined, np.nan, result)
        result = _domain_result(result)
        if outputs:
            results = result if isinstance(result, tuple) else (result,)
            restored = []
            for output, value in zip(outputs, results):
                if output is not None:
                    np.copyto(np.asarray(output), np.asarray(value))
                restored.append(value if output is None else output)
            return tuple(restored) if isinstance(result, tuple) else restored[0]
        return result

    def __array_function__(self, function, types, args, kwargs):
        return _domain_result(super().__array_function__(function, types, args, kwargs))


def evaluate_target(function, variables):
    try:
        with np.errstate(divide='ignore', invalid='ignore', over='raise', under='ignore'):
            inputs = [np.array(value, dtype=float, copy=True).view(_TargetValues) for value in variables.values()]
            values = np.asarray(function(*inputs), dtype=float)
    except FloatingPointError as error:
        raise ValueError('numerical overflow while generating target labels') from error
    count = len(next(iter(variables.values())))
    if values.ndim == 0:
        values = np.full(count, values.item())
    if values.shape != (count,):
        raise ValueError('the target must produce one value per sample')
    return readonly_target(np.where(np.isfinite(values), values, np.nan))


def available_operators(config):
    if config['ternary_operators_on']:
        raise NotImplementedError('ternary operators are not implemented')
    operators = ()
    if config['binary_operators_on']:
        operators += BINARY_OPERATORS
    if config['unary_operators_on']:
        operators += UNARY_OPERATORS
    return operators


def sample_interval(function, config):
    default_min, default_max = TARGET_DOMAINS.get(function, (-20.0, 20.0))
    sample_min = config['sample_min']
    sample_max = config['sample_max']
    if sample_min is None:
        sample_min = default_min
    if sample_max is None:
        sample_max = default_max
    sample_width = sample_max - sample_min
    if not np.isfinite(sample_width) or sample_width <= 0:
        raise ValueError('the sample interval must have a finite positive width')
    return (sample_min, sample_max)


def add_output_noise(clean_target, config, dataset_seed=1):
    level = config.get('noise_level', 0.0)
    if not np.isfinite(level) or level < 0:
        raise ValueError('noise_level must be finite and nonnegative')
    target = np.asarray(clean_target, dtype=float).copy()
    readonly_target(target)
    finite = np.isfinite(target)
    if level:
        noise_seed = config.get('noise_seed', 1)
        if noise_seed is None:
            noise_seed = dataset_seed
        if not isinstance(noise_seed, int) or noise_seed < 0:
            raise ValueError('noise_seed must be a nonnegative integer or None')
        generator = np.random.default_rng(noise_seed)
        scale = float(np.max(np.abs(target[finite])))
        deviation = scale * float(np.sqrt(np.mean((target[finite] / scale) ** 2)) if config.get('protocol_version', 1) >= 2 else np.std(target[finite] / scale)) if scale else 0.0
        target[finite] += generator.normal(0.0, level * deviation, int(finite.sum()))
    if not np.all(np.isfinite(target[finite])):
        raise ValueError('the noise settings produced nonfinite training outputs')
    target.setflags(write=False)
    return target


def make_training_dataset(function, config, seed):
    if current_protocol(config):
        current = {**config, 'dataset_seed': seed}
        variables, clean = random_dataset(function, current)
        labels, noise_seed = noisy_labels(clean, current, function.__name__, config.get('repetition', 1))
        return (variables, labels, clean)
    sample_count = config['sample_count']
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError('sample_count must be a positive integer')
    sample_min, sample_max = sample_interval(function, config)
    if config.get('protocol_version', 1) < 2 and (not sample_min <= 0 <= sample_max):
        raise ValueError('dataset interval must contain zero')
    generator = random.Random(seed)
    variables = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if parameter.kind not in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            raise ValueError('target functions must have explicit positional arguments')
        values = np.empty(sample_count)
        for index in range(sample_count):
            values[index] = generator.uniform(sample_min, sample_max)
        if config.get('protocol_version', 1) < 2:
            values[0] = 0.0
        variables[name] = values
    if not variables:
        raise ValueError('the target function must have at least one input variable')
    target = evaluate_target(function, variables)
    return (variables, add_output_noise(target, config, seed), target)


def make_workspace(variables, config, seed, lexicon=None, additions=None):
    for key in ('variable_atom_count', 'operator_count'):
        if not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(key + ' must be a nonnegative integer')
    operators = available_operators(config)
    if config['variable_atom_count'] and (not variables):
        raise ValueError('the lexicon has no variables')
    if config['operator_count'] and (not operators):
        raise ValueError('operator_count > 0 with no operator group enabled')
    generator = random.Random(seed)
    variable_names = list(variables)
    workspace = []
    inherited = list(lexicon or ())
    for _ in range(config['variable_atom_count']):
        if inherited:
            index = generator.randrange(len(variable_names) + len(inherited))
            if index >= len(variable_names):
                workspace.append(inherited[index - len(variable_names)].occurrence())
                continue
            name = variable_names[index]
        else:
            name = generator.choice(variable_names)
        workspace.append(SemanticObject(name, variables[name]))
    for _ in range(config['operator_count']):
        symbol = generator.choice(operators)
        workspace.append(OperatorObject([symbol]))
    generator.shuffle(workspace)
    if additions is not None:
        extra_generator = random.Random(f'{seed}|additions')
        for node in additions(extra_generator):
            workspace.insert(extra_generator.randrange(len(workspace) + 1), node)
    return workspace


def random_dataset(function, config, purpose='train'):
    count = config['sample_count'] if purpose == 'train' else config.get('test_sample_count', 1000)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('dataset sample counts must be positive integers')
    lower, upper = sample_interval(function, config)
    master = config['dataset_seed'] if purpose == 'train' else config.get('test_seed', 2)
    seed = stream_seed(master, function.__name__, purpose)
    generator = np.random.default_rng(seed)
    sampling = config.get('dataset_sampling', 'iid_uniform')
    if sampling not in ('iid_uniform', 'latin_hypercube'):
        raise ValueError('unknown dataset_sampling')
    variables = {}
    for name in inspect.signature(function).parameters:
        if sampling == 'latin_hypercube':
            unit = (generator.permutation(count) + generator.random(count)) / count
            variables[name] = lower + (upper - lower) * unit
        else:
            variables[name] = generator.uniform(lower, upper, count)
    for values in variables.values():
        values.setflags(write=False)
    return (variables, evaluate_target(function, variables))
