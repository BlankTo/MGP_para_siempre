import hashlib
import math
from itertools import product

import numpy as np

from mgp.engine.fitness import rmse
from mgp.engine.objects import apply_operator
from mgp.engine.problem import evaluate_target, sample_interval


def array_digest(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


def noise_audit(variables, clean, labels, test, settings, seed):
    finite = np.isfinite(clean)
    residual = labels[finite] - clean[finite]
    scale = rmse(np.zeros(int(finite.sum())), clean[finite])
    expected = settings['noise_level'] * scale
    rms = rmse(residual, np.zeros_like(residual))
    return {'dataset_seed': settings['dataset_seed'], 'target': settings['target'], 'noise_level': settings['noise_level'], 'noise_seed': seed, 'samples': len(clean), 'finite_samples': int(finite.sum()), 'signal_rms': scale, 'expected_sigma': expected, 'noise_rms': rms, 'noise_mean': float(np.mean(residual)), 'noise_std': float(np.std(residual)), 'noise_to_signal_rms': rms / scale if scale else None, 'realized_std_over_sigma': float(np.std(residual)) / expected if expected else None, 'changed_fraction': float(np.mean(residual != 0.0)), 'finite_mask_preserved': int(np.array_equal(np.isfinite(labels), finite)), 'input_digest': hashlib.sha256(''.join((array_digest(variables[k]) for k in sorted(variables))).encode()).hexdigest(), 'clean_digest': array_digest(clean), 'labels_digest': array_digest(labels), 'test_digest': hashlib.sha256((''.join((array_digest(test[0][k]) for k in sorted(test[0]))) + array_digest(test[1])).encode()).hexdigest()}


def prediction_diversity(values, signal_rms):
    valid = [np.asarray(value, dtype=float) for value in values if np.all(np.isfinite(value))]
    result = {'valid_prediction_fraction': len(valid) / len(values) if values else None, 'diversity_valid_entries': len(valid), 'semantic_cosine_distance': None, 'prediction_dispersion_nrmse': None}
    if len(valid) < 2:
        return result
    matrix = np.stack(valid)
    amplitude = np.max(np.abs(matrix), axis=1)
    unit = np.zeros_like(matrix)
    nonzero = amplitude > 0
    unit[nonzero] = matrix[nonzero] / amplitude[nonzero, None]
    unit[nonzero] /= np.linalg.norm(unit[nonzero], axis=1, keepdims=True)
    total = unit.sum(axis=0)
    count = len(valid)
    dot_sum = float(total @ total) + int((~nonzero).sum()) ** 2
    result['semantic_cosine_distance'] = max(0.0, min(2.0, 1.0 - (dot_sum - count) / (count * (count - 1))))
    scale = float(amplitude.max())
    if signal_rms > 0:
        dispersion = 0.0 if scale == 0 else float(np.sqrt(np.mean(np.var(matrix / scale, axis=0))))
        result['prediction_dispersion_nrmse'] = dispersion * (scale / signal_rms)
    return result


def archive_quality(rows, values, signal_rms):
    finite_errors = [row['clean_rmse'] / signal_rms for row in rows if signal_rms > 0 and math.isfinite(row['clean_rmse'])]
    return {'symbolic_unique': len({row['key'] for row in rows}), 'shape_profile_count': len({(row['expression_operators'], row['expression_leaves'], row['expression_depth']) for row in rows}), 'clean_nrmse_best': min(finite_errors) if finite_errors else None, 'clean_nrmse_median': float(np.median(finite_errors)) if finite_errors else None, 'clean_quality_valid_entries': len(finite_errors), 'mean_expression_operators': float(np.mean([row['expression_operators'] for row in rows])) if rows else None, 'mean_expression_depth': float(np.mean([row['expression_depth'] for row in rows])) if rows else None, **prediction_diversity(values, signal_rms)}


def make_probe_dataset(function, names, config):
    count = config['probe_axis_points']
    near_zero = config['probe_near_zero']
    if not isinstance(count, int) or count < 2:
        raise ValueError('probe_axis_points must be an integer of at least 2')
    if not np.isfinite(near_zero) or near_zero <= 0:
        raise ValueError('probe_near_zero must be finite and positive')
    lower, upper = sample_interval(function, config)
    axis = lower + (np.arange(count) + 0.5) * (upper - lower) / count
    points = {}
    for point in product(axis, repeat=len(names)):
        points.setdefault(point, set()).add('interior')
    for point in product((lower, upper), repeat=len(names)):
        points.setdefault(point, set()).add('boundary')
    points.setdefault((0.0,) * len(names), set()).add('origin')
    for point in product((-near_zero, 0.0, near_zero), repeat=len(names)):
        points.setdefault(point, set()).add('near_zero')
    if len(names) == 2:
        for value in [lower, upper, *axis]:
            points.setdefault((0.0, value), set()).add('axis_zero')
            points.setdefault((value, 0.0), set()).add('axis_zero')
    matrix = np.array(list(points))
    variables = {name: matrix[:, index].copy() for index, name in enumerate(names)}
    target = evaluate_target(function, variables)
    groups = ['|'.join(sorted(labels)) for labels in points.values()]
    masks = {}
    for name in ('origin', 'axis_zero', 'near_zero'):
        masks[name] = np.array([name in labels for labels in points.values()])
    return (variables, target, groups, masks)


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
                    raise ValueError('missing evaluation input: ' + current)
                values = self.variables[current]
                self.cache[key] = (values, ~np.isfinite(values))
            elif not ready:
                pending.append((current, True))
                pending.extend(((operand, False) for operand in current[1:]))
            else:
                operands = [self.cache[id(operand)] for operand in current[1:]]
                values = apply_operator(current[0], *[item[0] for item in operands])
                invalid = ~np.isfinite(values)
                for previous_values, previous_invalid in operands:
                    invalid = invalid | previous_invalid
                self.cache[key] = (values, invalid)
        return self.cache[id(expression)]
