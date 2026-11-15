from functools import partial

import numpy as np

from mgp.engine.fitness import rmse
from mgp.engine.symbolic import sympy_equivalent


def cosine_similarity(first, second):
    first, second = (np.asarray(first), np.asarray(second))
    if first.shape != second.shape or np.any(np.isinf(first)) or np.any(np.isinf(second)):
        return None
    if not np.array_equal(np.isnan(first), np.isnan(second)):
        return None
    finite = np.isfinite(first)
    if not np.any(finite):
        return None
    first, second = (first[finite], second[finite])
    first_scale = float(np.max(np.abs(first)))
    second_scale = float(np.max(np.abs(second)))
    if first_scale == 0 or second_scale == 0:
        return None
    first = first / first_scale
    second = second / second_scale
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    return float(np.clip(np.dot(first, second), -1, 1))


def rmse_equivalent(first, second, tolerance=1e-12):
    return rmse(first, second) <= tolerance


def allclose_equivalent(first, second, atol=1e-12, rtol=1e-12):
    first, second = (np.asarray(first), np.asarray(second))
    return bool(first.shape == second.shape and (not np.any(np.isinf(first))) and (not np.any(np.isinf(second))) and np.any(np.isfinite(first)) and np.allclose(first, second, atol=atol, rtol=rtol, equal_nan=True))


def symbolic_enabled(config):
    enabled = config.get('sympy_enabled', config.get('equivalence_method') == 'sympy')
    if not isinstance(enabled, bool):
        raise ValueError('sympy_enabled must be a boolean')
    return enabled


def equivalence_name(config):
    if symbolic_enabled(config):
        return 'sympy'
    name = config.get('equivalence_method', 'rmse')
    return 'rmse' if name == 'sympy' else name


COMPARISON_METHODS = {'cosine': cosine_similarity, 'rmse': rmse}
EQUIVALENCE_METHODS = {'rmse': (rmse_equivalent, ('tolerance',)), 'allclose': (allclose_equivalent, ('atol', 'rtol')), 'sympy': (sympy_equivalent, ())}


def select_comparison(name='cosine', parameters=None):
    if name not in COMPARISON_METHODS:
        raise ValueError('unknown comparison method: ' + name)
    return partial(COMPARISON_METHODS[name], **parameters or {})


def select_equivalence(name='rmse', tolerance=1e-12, atol=1e-12, rtol=1e-12):
    if name not in EQUIVALENCE_METHODS:
        raise ValueError('unknown equivalence method: ' + name)
    parameters = {'tolerance': tolerance, 'atol': atol, 'rtol': rtol}
    for value in parameters.values():
        if not np.isfinite(value) or value < 0:
            raise ValueError('equivalence tolerances must be finite and nonnegative')
    function, names = EQUIVALENCE_METHODS[name]
    selected = {name: parameters[name] for name in names}
    return partial(function, **selected)
