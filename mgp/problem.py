import inspect
import random

import numpy as np

from targets import TARGET_DOMAINS

from .objects import BINARY_OPERATORS, UNARY_OPERATORS, OperatorObject, SemanticObject


def resolve_seed(seed):
    if seed is None:
        return random.SystemRandom().randrange(2**63)

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seeds must be nonnegative integers or None.")

    return seed


def available_operators(config):
    if config["ternary_operators_on"]:
        raise NotImplementedError("Ternary operators are not implemented.")

    operators = ()

    if config["binary_operators_on"]:
        operators += BINARY_OPERATORS

    if config["unary_operators_on"]:
        operators += UNARY_OPERATORS

    return operators


def sample_interval(function, config):
    default_min, default_max = TARGET_DOMAINS.get(function, (-20.0, 20.0))
    sample_min = config["sample_min"]
    sample_max = config["sample_max"]

    if sample_min is None:
        sample_min = default_min

    if sample_max is None:
        sample_max = default_max

    sample_width = sample_max - sample_min

    if not np.isfinite(sample_width) or sample_width <= 0:
        raise ValueError("The sample interval must have a finite positive width.")

    return sample_min, sample_max


def make_dataset(function, config, seed):
    sample_count = config["sample_count"]

    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer.")

    sample_min, sample_max = sample_interval(function, config)
    generator = random.Random(seed)
    variables = {}

    for name, parameter in inspect.signature(function).parameters.items():
        if parameter.kind not in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            raise ValueError("Target functions must have explicit positional arguments.")

        values = np.empty(sample_count)

        for index in range(sample_count):
            values[index] = generator.uniform(sample_min, sample_max)

        variables[name] = values

    if not variables:
        raise ValueError("The target function must have at least one input variable.")

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        target = np.asarray(function(*variables.values()), dtype=float)

    if target.ndim == 0:
        target = np.full(sample_count, target.item())

    if target.shape != (sample_count,) or not np.all(np.isfinite(target)):
        raise ValueError("The target must produce one finite value per sample. Check its sample interval.")

    return variables, target


def make_workspace(variables, config, seed):
    for key in ("variable_atom_count", "operator_count"):
        if not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(key + " must be a nonnegative integer.")

    operators = available_operators(config)

    if config["variable_atom_count"] and not variables:
        raise ValueError("The lexicon has no variables.")

    if config["operator_count"] and not operators:
        raise ValueError("Enable an operator group or set operator_count to 0.")

    generator = random.Random(seed)
    variable_names = list(variables)
    workspace = []

    for _ in range(config["variable_atom_count"]):
        name = generator.choice(variable_names)
        workspace.append(SemanticObject(name, variables[name]))

    for _ in range(config["operator_count"]):
        symbol = generator.choice(operators)
        workspace.append(OperatorObject([symbol]))

    generator.shuffle(workspace)

    return workspace
