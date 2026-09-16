import numpy as np

from .objects import SemanticObject


def readonly_target(target):
    target = np.asarray(target, dtype=float)

    if target.ndim != 1 or target.size == 0 or not np.all(np.isfinite(target)):
        raise ValueError("The target must be a nonempty finite vector.")

    if target.flags.writeable or not target.flags.owndata:
        target = target.copy()

    target.setflags(write=False)

    return target


def rmse(values, target):
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)

    if values.shape != target.shape or values.size == 0:
        return float("inf")

    with np.errstate(over="ignore", invalid="ignore"):
        errors = np.abs(values - target)

    if not np.all(np.isfinite(errors)):
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(target)):
            return float("inf")

        scale = max(float(np.max(np.abs(values))), float(np.max(np.abs(target))))
        errors = values / scale - target / scale

        with np.errstate(over="ignore"):
            return float(scale * np.sqrt(np.mean(errors ** 2)))

    scale = float(np.max(errors))

    if scale == 0:
        return 0.0

    return float(scale * np.sqrt(np.mean((errors / scale) ** 2)))


def nonnegative_mean(values):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return None

    scale = float(np.max(values))

    if scale == 0:
        return 0.0

    return float(scale * np.mean(values / scale))


def object_fitness(current, target):
    if not isinstance(current, SemanticObject):
        return float("inf")

    target = np.asarray(target, dtype=float)
    can_cache = (
        target.flags.owndata
        and not target.flags.writeable
        and current.values.flags.owndata
        and not current.values.flags.writeable
    )

    if (
        can_cache
        and current._fitness_target is target
        and current._fitness_values is current.values
    ):
        return current._fitness_value

    fitness = rmse(current.values, target)

    if can_cache:
        current._fitness_target = target
        current._fitness_values = current.values
        current._fitness_value = fitness
    else:
        current._fitness_target = None
        current._fitness_values = None
        current._fitness_value = None

    return fitness


def best_index(workspace, target):
    selected_index = None
    selected_fitness = float("inf")

    for index, current in enumerate(workspace):
        if isinstance(current, SemanticObject):
            fitness = object_fitness(current, target)

            if selected_index is None or fitness < selected_fitness:
                selected_index = index
                selected_fitness = fitness

    return selected_index
