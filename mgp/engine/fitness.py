import numpy as np

from mgp.engine.objects import SemanticObject


def readonly_target(target):
    target = np.asarray(target, dtype=float)
    if target.ndim != 1 or target.size == 0 or np.any(np.isinf(target)) or (not np.any(np.isfinite(target))):
        raise ValueError('the target needs at least one finite value and no numerical failures')
    if target.flags.writeable or not target.flags.owndata:
        target = target.copy()
    target.setflags(write=False)
    return target


def rmse(values, target):
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    if values.shape != target.shape or values.size == 0:
        return float('inf')
    if np.any(np.isinf(values)) or np.any(np.isinf(target)):
        return float('inf')
    if not np.array_equal(np.isnan(values), np.isnan(target)):
        return float('inf')
    finite = np.isfinite(target)
    if not np.any(finite):
        return float('inf')
    if not np.all(finite):
        values, target = (values[finite], target[finite])
    with np.errstate(over='ignore', invalid='ignore'):
        errors = np.abs(values - target)
    if not np.all(np.isfinite(errors)):
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(target)):
            return float('inf')
        scale = max(float(np.max(np.abs(values))), float(np.max(np.abs(target))))
        errors = values / scale - target / scale
        with np.errstate(over='ignore'):
            return float(scale * np.sqrt(np.mean(errors ** 2)))
    scale = float(np.max(errors))
    if scale == 0:
        return 0.0
    return float(scale * np.sqrt(np.mean((errors / scale) ** 2)))


def object_fitness(current, target):
    if not isinstance(current, SemanticObject):
        return float('inf')
    target = np.asarray(target, dtype=float)
    can_cache = target.flags.owndata and (not target.flags.writeable) and current.values.flags.owndata and (not current.values.flags.writeable)
    if not can_cache or current._fitness_values is not current.values:
        current._fitness_cache = ()
        current._fitness_values = None
    if can_cache:
        for cached_target, fitness in current._fitness_cache:
            if cached_target is target:
                return fitness
    fitness = rmse(current.values, target)
    if can_cache:
        current._fitness_values = current.values
        current._fitness_cache = ((target, fitness),) + current._fitness_cache[:1]
    return fitness


def best_index(workspace, target):
    selected_index = None
    selected_fitness = float('inf')
    for index, current in enumerate(workspace):
        if isinstance(current, SemanticObject):
            fitness = object_fitness(current, target)
            if selected_index is None or fitness < selected_fitness:
                selected_index = index
                selected_fitness = fitness
    return selected_index
