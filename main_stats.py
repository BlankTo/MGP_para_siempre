import csv
from datetime import datetime, timezone
from pathlib import Path
import random
import time

import numpy as np

from config import CONFIG
from mgp import Derivation, SemanticObject, make_dataset, make_workspace, resolve_seed
from mgp.fitness import nonnegative_mean
from mgp.objects import MATRIX_COLUMNS
from mgp.problem import sample_interval
from mgp.statistics import (
    EvolutionStatistics,
    add_fitness_distances,
    add_value_distances,
    derivation_row,
    save_plots,
    write_csv,
    write_json,
)
from targets import TARGETS


STATS_CONFIG = {
    **CONFIG,

    "derivations": 10000,
    "output_dir": "stats_output",
    "progress_every": 100,
    "max_value_pairs": 100000,
    "save_plots": True,
}


def summarize_rows(rows):
    final_types = {"semantic": 0, "incomplete": 0, "operator": 0, "multiple": 0, "empty": 0}
    solutions_found = 0
    exact_zero_found = 0
    merge_counts = []
    best_fitnesses = []
    archive_sizes = []

    for row in rows:
        final_types[row["final_type"]] += 1

        if row["found_solution"]:
            solutions_found += 1

        if row["found_exact_zero"]:
            exact_zero_found += 1

        merge_counts.append(row["merge_count"])
        archive_sizes.append(row["archive_size"])

        if row["best_fitness"] is not None and np.isfinite(row["best_fitness"]):
            best_fitnesses.append(row["best_fitness"])

    best_fitness = None

    if best_fitnesses:
        best_fitness = min(best_fitnesses)

    return {
        "derivations": len(rows),
        "solutions_found": solutions_found,
        "exact_zero_found": exact_zero_found,
        "final_types": final_types,
        "mean_merge_count": sum(merge_counts) / len(rows),
        "mean_best_fitness": nonnegative_mean(best_fitnesses),
        "best_fitness": best_fitness,
        "finite_best_count": len(best_fitnesses),
        "mean_archive_size": sum(archive_sizes) / len(rows),
    }


def run_statistics(settings):
    derivation_count = settings["derivations"]
    progress_every = settings["progress_every"]
    max_pairs = settings["max_value_pairs"]

    if not isinstance(derivation_count, int) or derivation_count <= 0:
        raise ValueError("derivations must be a positive integer.")

    if not isinstance(progress_every, int) or progress_every < 0:
        raise ValueError("progress_every must be a nonnegative integer.")

    if max_pairs is not None and (not isinstance(max_pairs, int) or max_pairs < 0):
        raise ValueError("max_value_pairs must be a nonnegative integer or None.")

    started = time.perf_counter()
    config = settings.copy()
    config["seed"] = resolve_seed(settings["seed"])
    config["dataset_seed"] = resolve_seed(settings["dataset_seed"])
    generator = random.Random(config["seed"])

    function, target_expression = TARGETS[config["target"]]
    config["sample_min"], config["sample_max"] = sample_interval(function, config)
    variables, target = make_dataset(function, config, config["dataset_seed"])

    run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(settings["output_dir"]) / run_name
    matrices_dir = output_dir / "derivation_matrices"
    matrices_dir.mkdir(parents=True, exist_ok=False)

    dataset_columns = list(variables)
    dataset_columns.append("target")
    dataset_values = list(variables.values())
    dataset_values.append(target)
    dataset = np.column_stack(dataset_values)
    write_csv(output_dir / "dataset.csv", dataset_columns, dataset)

    evolution = EvolutionStatistics()
    rows = []
    best_entries = []
    final_entries = []

    for number in range(1, derivation_count + 1):
        workspace_seed = generator.getrandbits(63)
        derivation_seed = generator.getrandbits(63)
        workspace = make_workspace(variables, config, workspace_seed)
        derivation = Derivation(target, config, derivation_seed)
        derivation.run(workspace)

        rows.append(derivation_row(derivation, number, workspace_seed))
        evolution.add(derivation.initial_counts, derivation.step_counts)
        write_csv(
            matrices_dir / f"derivation_{number:06d}.csv",
            MATRIX_COLUMNS,
            derivation.step_counts,
        )

        if isinstance(derivation.best, SemanticObject):
            best_entries.append((number - 1, derivation.best))

        if isinstance(derivation.final_object, SemanticObject):
            final_entries.append((number - 1, derivation.final_object))

        if progress_every and (number % progress_every == 0 or number == derivation_count):
            elapsed = time.perf_counter() - started
            remaining = elapsed * (derivation_count - number) / number
            print(
                f"Derivations: {number}/{derivation_count}"
                f" | elapsed {elapsed:.1f}s | ETA {remaining:.1f}s",
                flush=True,
            )

    if progress_every:
        print("Computing diversity and saving statistics...", flush=True)

    diversity = {}

    for prefix, entries in (("best", best_entries), ("final", final_entries)):
        fitness_distance = add_fitness_distances(rows, entries, prefix)
        distance_seed = generator.getrandbits(63)
        value_distances = add_value_distances(rows, entries, prefix, max_pairs, distance_seed)

        diversity[prefix] = {
            "mean_fitness_distance": fitness_distance,
            "value_distances": value_distances,
        }

    evolution.save(output_dir)

    with open(output_dir / "derivation_stats.csv", "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    configuration = {}

    for key in CONFIG:
        configuration[key] = config[key]

    summary = {"target": target_expression, "configuration": configuration}
    summary.update(summarize_rows(rows))
    summary["diversity"] = diversity
    write_json(output_dir / "summary.json", summary)

    if settings["save_plots"]:
        save_plots(evolution, output_dir)

    print("Target:", target_expression)
    print("Derivations:", len(rows), "Solutions:", summary["solutions_found"])
    print("Mean MERGE count:", summary["mean_merge_count"])
    print("Results:", output_dir.resolve())

    return output_dir


if __name__ == "__main__":
    run_statistics(STATS_CONFIG)
