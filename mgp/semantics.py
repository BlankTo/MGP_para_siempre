from functools import partial

import numpy as np

from .fitness import rmse


def cosine_similarity(first, second):
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        return None

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
    return bool(np.allclose(first, second, atol=atol, rtol=rtol))


COMPARISON_METHODS = {"cosine": cosine_similarity, "rmse": rmse}
EQUIVALENCE_METHODS = {
    "rmse": (rmse_equivalent, ("tolerance",)),
    "allclose": (allclose_equivalent, ("atol", "rtol")),
}


def select_comparison(name="cosine", parameters=None):
    if name not in COMPARISON_METHODS:
        raise ValueError("Unknown comparison method: " + name)

    return partial(COMPARISON_METHODS[name], **(parameters or {}))


def select_equivalence(name="rmse", tolerance=1e-12, atol=1e-12, rtol=1e-12):
    if name not in EQUIVALENCE_METHODS:
        raise ValueError("Unknown equivalence method: " + name)

    parameters = {"tolerance": tolerance, "atol": atol, "rtol": rtol}

    for value in parameters.values():
        if not np.isfinite(value) or value < 0:
            raise ValueError("Equivalence tolerances must be finite and nonnegative.")

    function, names = EQUIVALENCE_METHODS[name]
    selected = {name: parameters[name] for name in names}

    return partial(function, **selected)
