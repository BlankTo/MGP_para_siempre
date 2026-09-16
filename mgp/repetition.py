import random

import numpy as np

from .derivation import Derivation
from .fitness import nonnegative_mean
from .problem import make_dataset, make_workspace, resolve_seed
from targets import TARGETS


def repeat_derivations(config, count, stop_on_solution=False, progress_every=10):
    if not isinstance(count, int) or count <= 0:
        raise ValueError("The derivation count must be a positive integer.")

    if not isinstance(progress_every, int) or progress_every < 0:
        raise ValueError("progress_every must be a nonnegative integer.")

    seed = resolve_seed(config["seed"])
    dataset_seed = resolve_seed(config["dataset_seed"])
    generator = random.Random(seed)

    function, target_expression = TARGETS[config["target"]]
    variables, target = make_dataset(function, config, dataset_seed)

    best = None
    best_fitness = float("inf")
    best_workspace_seed = None
    best_derivation_seed = None
    fitnesses = []
    solutions_found = 0
    exact_matches = 0

    for number in range(1, count + 1):
        workspace_seed = generator.getrandbits(63)
        derivation_seed = generator.getrandbits(63)
        workspace = make_workspace(variables, config, workspace_seed)
        derivation = Derivation(target, config, derivation_seed)
        derivation.run(workspace)

        fitness = derivation.best_fitness

        if np.isfinite(fitness):
            fitnesses.append(fitness)

        if derivation.best is not None and (best is None or fitness < best_fitness):
            best = derivation.best
            best_fitness = fitness
            best_workspace_seed = workspace_seed
            best_derivation_seed = derivation_seed

        found_solution = fitness <= config["perfect_fit_tolerance"]

        if found_solution:
            solutions_found += 1

        if fitness == 0:
            exact_matches += 1

        if progress_every and (number % progress_every == 0 or number == count):
            print(f"Derivations: {number}/{count} | best RMSE: {best_fitness:.8g}", flush=True)

        if stop_on_solution and found_solution:
            break

    return {
        "target": target_expression,
        "seed": seed,
        "dataset_seed": dataset_seed,
        "derivations": number,
        "solutions_found": solutions_found,
        "exact_matches": exact_matches,
        "mean_best_fitness": nonnegative_mean(fitnesses),
        "finite_best_count": len(fitnesses),
        "best": best,
        "best_fitness": best_fitness,
        "best_workspace_seed": best_workspace_seed,
        "best_derivation_seed": best_derivation_seed,
    }


def print_results(result):
    print("\n" + "=" * 80)
    print("Target:", result["target"])
    print("Seed:", result["seed"], "Dataset seed:", result["dataset_seed"])
    print("Derivations:", result["derivations"])
    print("Solutions within tolerance:", result["solutions_found"])
    print("Exact matches (RMSE = 0):", result["exact_matches"])
    print("Mean best RMSE:", result["mean_best_fitness"])
    print("Derivations with finite best RMSE:", result["finite_best_count"])

    if result["best"] is None:
        print("Best object: none")
    else:
        print("Best structure:", result["best"].structure.text())
        print("Best object:", result["best"].expression)
        print("Best RMSE:", result["best_fitness"])
        print("Best workspace seed:", result["best_workspace_seed"])
        print("Best derivation seed:", result["best_derivation_seed"])
