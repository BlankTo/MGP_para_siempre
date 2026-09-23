from bisect import bisect_right
import csv
import json
import random

import matplotlib.pyplot as plt
import numpy as np

from .fitness import nonnegative_mean, object_fitness, rmse
from .objects import MATRIX_COLUMNS, SemanticObject


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as output:
        json.dump(data, output, indent=2, ensure_ascii=False, allow_nan=False)


def finite_or_none(value):
    if np.isfinite(value):
        return float(value)

    return None


def derivation_row(derivation, number, workspace_seed):
    final_object = derivation.final_object
    best = derivation.best
    final_type = "multiple"
    best_equation = ""
    final_equation = ""
    best_fitness = None
    final_fitness = None

    if not derivation.workspace:
        final_type = "empty"
    elif final_object is not None:
        final_type = final_object.kind
        final_equation = final_object.expression

    if best is not None:
        best_equation = best.expression
        best_fitness = derivation.best_fitness

    if isinstance(final_object, SemanticObject):
        final_fitness = object_fitness(final_object, derivation.target)

    row = {
        "derivation": number,
        "workspace_seed": workspace_seed,
        "derivation_seed": derivation.seed,
        "merge_count": derivation.merge_count,
        "external_merge_count": derivation.external_merge_count,
        "internal_merge_count": derivation.internal_merge_count,
        "internal_fallback_count": derivation.internal_fallback_count,
        "final_workspace_size": len(derivation.workspace),
        "final_type": final_type,
        "termination_reason": derivation.termination_reason,
        "best_equation": best_equation,
        "final_equation": final_equation,
        "best_fitness": best_fitness,
        "final_fitness": final_fitness,
        "found_solution": (
            best is not None
            and derivation.best_fitness <= derivation.config["perfect_fit_tolerance"]
        ),
        "found_exact_zero": best is not None and derivation.best_fitness == 0,
        "archive_size": len(derivation.archive.objects),
    }

    for prefix in ("best", "final"):
        row[prefix + "_mean_pairwise_fitness_distance"] = None
        row[prefix + "_mean_pairwise_value_distance"] = None
        row[prefix + "_value_distance_pairs"] = 0

    return row


class EvolutionStatistics:
    def __init__(self):
        self.contributors = []
        self.means = []
        self.squared_differences = []
        self.normalized_means = []
        self.normalized_squared_differences = []

    def add(self, initial_counts, step_counts):
        matrix = [initial_counts] + step_counts

        for step, counts in enumerate(matrix):
            values = np.asarray(counts, dtype=float)
            workspace_size = np.sum(values)
            proportions = np.zeros(len(MATRIX_COLUMNS))

            if workspace_size > 0:
                proportions = values / workspace_size

            if step == len(self.contributors):
                self.contributors.append(0)
                self.means.append(np.zeros(len(MATRIX_COLUMNS)))
                self.squared_differences.append(np.zeros(len(MATRIX_COLUMNS)))
                self.normalized_means.append(np.zeros(len(MATRIX_COLUMNS)))
                self.normalized_squared_differences.append(np.zeros(len(MATRIX_COLUMNS)))

            self.contributors[step] += 1
            count = self.contributors[step]

            difference = values - self.means[step]
            self.means[step] += difference / count
            self.squared_differences[step] += difference * (values - self.means[step])

            difference = proportions - self.normalized_means[step]
            self.normalized_means[step] += difference / count
            self.normalized_squared_differences[step] += difference * (
                proportions - self.normalized_means[step]
            )

    def arrays(self, normalized=False):
        if normalized:
            means = self.normalized_means
            squared_differences = self.normalized_squared_differences
        else:
            means = self.means
            squared_differences = self.squared_differences

        shape = (len(self.contributors), len(MATRIX_COLUMNS))
        average = np.asarray(means, dtype=float).reshape(shape)
        squared = np.asarray(squared_differences, dtype=float).reshape(shape)
        contributors = np.asarray(self.contributors).reshape(-1, 1)
        variance = np.maximum(0, squared / contributors)

        return average, variance

    def save(self, output_dir):
        for normalized in (False, True):
            prefix = "derivation_"

            if normalized:
                prefix += "normalized_"

            average, variance = self.arrays(normalized)
            matrices = (
                ("average", average),
                ("variance", variance),
                ("standard_deviation", np.sqrt(variance)),
            )

            for name, values in matrices:
                write_csv(output_dir / (prefix + name + "_matrix.csv"), MATRIX_COLUMNS, values)

        write_csv(
            output_dir / "derivation_contributors.csv",
            ["merge", "contributing_derivations"],
            enumerate(self.contributors),
        )


def add_fitness_distances(rows, entries, prefix):
    ordered = []

    for row_index, current in entries:
        fitness = rows[row_index][prefix + "_fitness"]

        if fitness is not None and np.isfinite(fitness):
            ordered.append((fitness, row_index))

    ordered.sort()
    count = len(ordered)

    if count < 2:
        return None

    fitnesses = []

    for fitness, row_index in ordered:
        fitnesses.append(fitness)

    fitnesses = np.asarray(fitnesses)
    scale = fitnesses[-1] - fitnesses[0]

    if scale == 0:
        for fitness, row_index in ordered:
            rows[row_index][prefix + "_mean_pairwise_fitness_distance"] = 0.0

        return 0.0

    scaled = (fitnesses - fitnesses[0]) / scale
    cumulative = np.cumsum(scaled)
    averages = []

    for position in range(count):
        sum_before = 0.0

        if position > 0:
            sum_before = cumulative[position - 1]

        sum_after = cumulative[-1] - cumulative[position]
        distance_before = position * scaled[position] - sum_before
        distance_after = sum_after - (count - position - 1) * scaled[position]
        mean_distance = (distance_before + distance_after) / (count - 1)
        distance = scale * max(0.0, mean_distance)
        row_index = ordered[position][1]

        rows[row_index][prefix + "_mean_pairwise_fitness_distance"] = float(distance)
        averages.append(distance)

    return nonnegative_mean(averages)


def add_value_distances(rows, entries, prefix, max_pairs, seed):
    count = len(entries)
    total_pairs = count * (count - 1) // 2
    evaluated_pairs = total_pairs

    if max_pairs is not None:
        evaluated_pairs = min(total_pairs, max_pairs)

    metadata = {
        "mode": "exact" if evaluated_pairs == total_pairs else "sampled_without_replacement",
        "available_pairs": total_pairs,
        "evaluated_pairs": evaluated_pairs,
        "seed": seed,
        "mean": None,
        "nonfinite_pairs": 0,
    }

    if evaluated_pairs == 0:
        return metadata

    pair_indices = range(total_pairs)

    if evaluated_pairs < total_pairs:
        pair_indices = random.Random(seed).sample(range(total_pairs), evaluated_pairs)

    pair_starts = []
    next_pair = 0

    for first_index in range(count):
        pair_starts.append(next_pair)
        next_pair += count - first_index - 1

    means = np.zeros(count)
    samples = np.zeros(count, dtype=int)
    overall_mean = 0.0

    for number, pair_index in enumerate(pair_indices, start=1):
        first_index = bisect_right(pair_starts, pair_index) - 1
        second_index = first_index + 1 + pair_index - pair_starts[first_index]
        first_values = entries[first_index][1].values
        second_values = entries[second_index][1].values
        distance = rmse(first_values, second_values)

        if not np.isfinite(distance):
            metadata["nonfinite_pairs"] += 1

        for index in (first_index, second_index):
            samples[index] += 1

            if not np.isfinite(distance):
                means[index] = float("inf")
            elif np.isfinite(means[index]):
                means[index] += (distance - means[index]) / samples[index]

        if not np.isfinite(distance):
            overall_mean = float("inf")
        elif np.isfinite(overall_mean):
            overall_mean += (distance - overall_mean) / number

    for position, (row_index, _) in enumerate(entries):
        if samples[position]:
            rows[row_index][prefix + "_mean_pairwise_value_distance"] = finite_or_none(means[position])
            rows[row_index][prefix + "_value_distance_pairs"] = int(samples[position])

    metadata["mean"] = finite_or_none(overall_mean)

    return metadata


def plot_evolution(evolution, normalized=False):
    average, variance = evolution.arrays(normalized)
    merge_axis = np.arange(len(average))
    figure, axis = plt.subplots(figsize=(11, 6))

    for column_index, column in enumerate(MATRIX_COLUMNS):
        mean = average[:, column_index]
        deviation = np.sqrt(variance[:, column_index])
        lower = np.maximum(0, mean - deviation)
        upper = mean + deviation

        if normalized:
            upper = np.minimum(1, upper)

        lines = axis.plot(merge_axis, mean, label=column.replace("_", " "), linewidth=1.7)
        color = lines[0].get_color()
        axis.fill_between(merge_axis, lower, upper, color=color, alpha=0.14)

    axis.set_xlim(0, len(average) - 1)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("MERGE index (0 = initial workspace)")
    axis.set_ylabel("Object count")
    axis.set_title("Workspace composition: mean and standard deviation")

    if normalized:
        axis.set_ylabel("Fraction of current workspace")
        axis.set_ylim(0, 1)

    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()

    return figure


def save_plots(evolution, output_dir):
    if len(evolution.contributors) < 2:
        return []

    plot_names = []

    for normalized in (False, True):
        filename = "derivation_average_standard_deviation.svg"

        if normalized:
            filename = "derivation_normalized_average_standard_deviation.svg"

        figure = plot_evolution(evolution, normalized)
        figure.savefig(output_dir / filename)
        plt.close(figure)
        plot_names.append(filename)

    return plot_names
