from itertools import product

import numpy as np

from .fitness import rmse
from .objects import apply_operator


DIAGNOSTIC_COLUMNS = [
    "archived_subtarget_count", "archived_monomial_count", "recovered_monomial_count",
    "first_archived_target_step", "best_probe_count", "best_nonfinite_count",
    "best_intermediate_nonfinite_count", "best_origin_nonfinite",
    "best_axis_nonfinite_count", "best_near_zero_nonfinite_count",
    "best_probe_rmse", "best_probe_match", "archive_checked_count",
    "archive_nonfinite_count", "archive_intermediate_nonfinite_count",
    "diagnostic_seconds",
]


def make_probe_dataset(function, names, config):
    count = config["probe_axis_points"]
    near_zero = config["probe_near_zero"]

    if not isinstance(count, int) or count < 2:
        raise ValueError("probe_axis_points must be an integer of at least 2.")

    if not np.isfinite(near_zero) or near_zero <= 0:
        raise ValueError("probe_near_zero must be finite and positive.")

    lower, upper = config["sample_min"], config["sample_max"]
    axis = lower + (np.arange(count) + 0.5) * (upper - lower) / count
    points = {}

    for point in product(axis, repeat=len(names)):
        points.setdefault(point, set()).add("interior")

    for point in product((lower, upper), repeat=len(names)):
        points.setdefault(point, set()).add("boundary")

    points.setdefault((0.0,) * len(names), set()).add("origin")

    for point in product((-near_zero, 0.0, near_zero), repeat=len(names)):
        points.setdefault(point, set()).add("near_zero")

    if len(names) == 2:
        for value in [lower, upper, *axis]:
            points.setdefault((0.0, value), set()).add("axis_zero")
            points.setdefault((value, 0.0), set()).add("axis_zero")

    matrix = np.array(list(points))
    variables = {name: matrix[:, index].copy() for index, name in enumerate(names)}
    target = np.asarray(function(*variables.values()), dtype=float)
    groups = ["|".join(sorted(labels)) for labels in points.values()]
    masks = {}

    for name in ("origin", "axis_zero", "near_zero"):
        masks[name] = np.array([name in labels for labels in points.values()])

    return variables, target, groups, masks


class ProbeEvaluator:
    def __init__(self, variables):
        self.variables = variables
        self.cache = {}

    def evaluate(self, expression):
        pending = [(expression, False)]

        while pending:
            current, ready = pending.pop()
            key = id(current)

            if key in self.cache:
                continue

            if isinstance(current, str):
                if current not in self.variables:
                    raise ValueError("Missing evaluation input: " + current)

                values = self.variables[current]
                self.cache[key] = (values, ~np.isfinite(values))
            elif not ready:
                pending.append((current, True))
                pending.extend((operand, False) for operand in current[1:])
            else:
                operands = [self.cache[id(operand)] for operand in current[1:]]
                values = apply_operator(current[0], *[item[0] for item in operands])
                invalid = ~np.isfinite(values)

                for previous_values, previous_invalid in operands:
                    invalid = invalid | previous_invalid

                self.cache[key] = (values, invalid)

        return self.cache[id(expression)]


def run_diagnostics(derivation, recovery, archive_recovery, probes=None):
    result = dict.fromkeys(DIAGNOSTIC_COLUMNS)
    result["archived_subtarget_count"] = len(archive_recovery.subtarget_steps)
    result["archived_monomial_count"] = sum(
        identifier.bit_count() == 1 for identifier in archive_recovery.subtarget_steps
    )
    result["recovered_monomial_count"] = sum(
        identifier.bit_count() == 1 for identifier in recovery.subtarget_steps
    )
    result["first_archived_target_step"] = archive_recovery.first_target_step

    if probes is None:
        return result

    variables, target, groups, masks = probes
    evaluator = ProbeEvaluator(variables)

    if derivation.best is not None:
        values, invalid = evaluator.evaluate(derivation.best.evaluation)
        nonfinite = ~np.isfinite(values)
        result["best_probe_count"] = len(values)
        result["best_nonfinite_count"] = int(nonfinite.sum())
        result["best_intermediate_nonfinite_count"] = int(invalid.sum())
        result["best_origin_nonfinite"] = int(np.any(nonfinite & masks["origin"]))
        result["best_axis_nonfinite_count"] = int(np.sum(nonfinite & masks["axis_zero"]))
        result["best_near_zero_nonfinite_count"] = int(np.sum(nonfinite & masks["near_zero"]))
        result["best_probe_rmse"] = rmse(values, target)
        result["best_probe_match"] = int(
            not np.any(invalid)
            and result["best_probe_rmse"] <= derivation.config["perfect_fit_tolerance"]
        )

    result["archive_checked_count"] = len(derivation.archive.objects)
    result["archive_nonfinite_count"] = 0
    result["archive_intermediate_nonfinite_count"] = 0

    for current in derivation.archive.objects:
        values, invalid = evaluator.evaluate(current.evaluation)
        result["archive_nonfinite_count"] += int(not np.all(np.isfinite(values)))
        result["archive_intermediate_nonfinite_count"] += int(np.any(invalid))

    return result
