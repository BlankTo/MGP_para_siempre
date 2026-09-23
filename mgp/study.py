import csv
from datetime import datetime, timezone
import hashlib
import inspect
from itertools import product
import json
from pathlib import Path
import platform
import random
import time

import numpy as np

from targets import TARGETS
from .archive import Archive
from .derivation import Derivation
from .diagnostics import make_probe_dataset, run_diagnostics
from .diversity import WorkspaceDiversity, workspace_composition
from .fitness import object_fitness, readonly_target
from .objects import SemanticObject, workspace_text
from .problem import available_operators, make_workspace, resolve_seed
from .recovery import POLYNOMIAL_TERMS, RecoveryTracker, make_subtargets
from .semantics import select_comparison, select_equivalence
from .statistics import write_csv, write_json
from .study_storage import StudyStore
from .study_report import create_report


def parameter_combinations(settings, grid):
    allowed = ("esel", "esur", "variable_atom_count", "operator_count")

    for name, values in grid.items():
        if name not in allowed:
            raise ValueError("Unknown or fixed study parameter: " + name)

        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(name + " needs a nonempty list of values.")

        if len(set(values)) != len(values):
            raise ValueError(name + " contains duplicate values.")

    combinations = []

    for values in product(*grid.values()):
        parameters = dict(zip(grid, values))
        config = {**settings, **parameters}

        if config["internal_merge_probability"] != 0 or config["unary_operators_on"]:
            raise ValueError("This study requires External MERGE and binary operators only.")

        if not config["binary_operators_on"] or config["fitness_probability"] != 1:
            raise ValueError("The study requires binary operators and fitness_probability = 1.")
        Derivation([1.0], config, 0)
        Archive(config["archive_max_size"], config["archive_similarity"], config["archive_atomic_objects"])
        operators = available_operators(config)

        for name in ("variable_atom_count", "operator_count"):
            if not isinstance(config[name], int) or config[name] < 0:
                raise ValueError(name + " must be a nonnegative integer.")

        if config["operator_count"] and not operators:
            raise ValueError("Enable an operator group or set operator_count to 0.")

        combinations.append(parameters)

    return combinations


def seed_plan(seed, count):
    generator = random.Random(seed)
    used = set()
    rows = []

    for repetition in range(1, count + 1):
        trial_seed = generator.getrandbits(63)

        while trial_seed in used:
            trial_seed = generator.getrandbits(63)

        used.add(trial_seed)
        trial_generator = random.Random(trial_seed)
        rows.append((repetition, trial_seed, trial_generator.getrandbits(63), trial_generator.getrandbits(63)))

    return rows


def code_fingerprint(comparison, equivalence):
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()

    for path in sorted(root.glob("*.py")) + [root.parent / "targets.py"]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())

    digest.update(inspect.getsource(comparison.func).encode())
    digest.update(inspect.getsource(equivalence.func).encode())

    return {
        "code_sha256": digest.hexdigest(),
        "python": ".".join(platform.python_version_tuple()[:2]),
        "numpy": np.__version__,
        "comparison": comparison.func.__module__ + "." + comparison.func.__qualname__,
        "equivalence": equivalence.func.__module__ + "." + equivalence.func.__qualname__,
    }


def make_study_dataset(function, config):
    sample_count = config["sample_count"]
    lower = config["sample_min"]
    upper = config["sample_max"]
    names = list(inspect.signature(function).parameters)

    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer.")

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("The study domain must have finite increasing bounds.")

    if len(names) == 1:
        variables = {names[0]: np.linspace(lower, upper, sample_count)}
    elif len(names) == 2:
        side = int(np.sqrt(sample_count))

        if side * side != sample_count:
            raise ValueError("A two-dimensional study dataset needs a square sample_count.")

        axis = np.linspace(lower, upper, side)
        mesh = np.meshgrid(axis, axis, indexing="ij")
        variables = {name: values.ravel() for name, values in zip(names, mesh)}
    else:
        raise ValueError("Study targets must have one or two inputs.")

    target = readonly_target(function(*variables.values()))

    for values in variables.values():
        values.setflags(write=False)

    return variables, target


def load_dataset(path):
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        columns = next(reader)
        matrix = np.array([[float(value) for value in row] for row in reader])

    variables = {}

    for index, name in enumerate(columns[:-1]):
        variables[name] = matrix[:, index].copy()

    return variables, readonly_target(matrix[:, -1])


def dataset_fingerprint(directory, targets, include_probes=False):
    digest = hashlib.sha256()

    for name in targets:
        digest.update((directory / "datasets" / (name + ".csv")).read_bytes())

        if include_probes:
            digest.update((directory / "datasets" / (name + "_probes.csv")).read_bytes())

    return digest.hexdigest()


def collect_derivation(variables, target, config, workspace_seed, derivation_seed, subtargets,
                       comparison, equivalence, save_trace=False):
    workspace = make_workspace(variables, config, workspace_seed)
    derivation = Derivation(target, config, derivation_seed)
    diversity = WorkspaceDiversity(comparison, equivalence, config["histogram_edges"])
    recovery = RecoveryTracker(
        derivation.target, subtargets, config["perfect_fit_tolerance"], config["subtarget_tolerance"],
    )
    recovery.archive = RecoveryTracker(
        derivation.target, subtargets, config["perfect_fit_tolerance"], config["subtarget_tolerance"],
    )
    archive_size = 0
    steps = []

    for current_workspace in derivation.steps(workspace):
        if derivation.last_merge is None:
            for current in current_workspace:
                recovery.observe(current, 0)
        else:
            recovery.observe(derivation.last_merge[2], derivation.merge_count)

        for current in derivation.archive.objects[archive_size:]:
            recovery.archive.observe(current, derivation.merge_count)

        archive_size = len(derivation.archive.objects)
        metrics = workspace_composition(current_workspace)
        size = len(current_workspace)
        metrics["workspace_size"] = size

        for name, count in list(metrics.items()):
            if name != "workspace_size":
                metrics["fraction_" + name] = count / size if size else 0.0

        comparisons, histogram = diversity.measure(current_workspace)
        metrics.update(comparisons)

        for index, count in enumerate(histogram):
            metrics[f"bin_{index:03d}_pairs"] = int(count)

        if save_trace:
            metrics["merge_kind"] = derivation.last_merge_kind or "initial"
            metrics["workspace_structure"] = workspace_text(current_workspace)
            metrics["workspace_evaluated"] = workspace_text(current_workspace, evaluated=True)
            metrics["best_fitness"] = derivation.best_fitness

        steps.append(metrics)

    return derivation, recovery, steps


def run_record(derivation, recovery, run_id, target_name, configuration_id, repetition, duration):
    final = derivation.final_object
    final_type = "multiple"

    if not derivation.workspace:
        final_type = "empty"
    elif final is not None:
        final_type = final.kind

    final_fitness = None

    if isinstance(final, SemanticObject):
        final_fitness = object_fitness(final, derivation.target)

    return {
        "run_id": run_id, "target": target_name,
        "configuration_id": configuration_id, "repetition": repetition,
        "merge_count": derivation.merge_count,
        "internal_merge_count": derivation.internal_merge_count,
        "external_merge_count": derivation.external_merge_count,
        "internal_fallback_count": derivation.internal_fallback_count,
        "termination_reason": derivation.termination_reason,
        "final_type": final_type, "final_workspace_size": len(derivation.workspace),
        "best_equation": derivation.best.expression if derivation.best is not None else None,
        "best_fitness": derivation.best_fitness if derivation.best is not None else None,
        "final_equation": final.expression if final is not None else None,
        "final_fitness": final_fitness,
        "first_target_step": recovery.first_target_step,
        "success": int(recovery.first_target_step is not None),
        "completed": 1,
        "subtarget_count": len(recovery.subtarget_steps),
        "archive_size": len(derivation.archive.objects), "duration_seconds": duration,
    }


def run_study(settings, grid, resume=None, max_runs=None):
    if max_runs is not None and (not isinstance(max_runs, int) or max_runs < 0):
        raise ValueError("max_runs must be a nonnegative integer or None.")

    if resume is not None:
        directory = Path(resume)
        with (directory / "study.json").open(encoding="utf-8") as source:
            manifest = json.load(source)

        config = manifest["configuration"]
        grid = manifest["grid"]

        if manifest.get("schema_version") != 3:
            raise ValueError("Resume this older study with v12. Use main_study_report.py to read its results.")

        if dataset_fingerprint(directory, config["targets"], config["probe_diagnostics"]) != manifest["dataset_sha256"]:
            raise ValueError("The saved datasets have changed.")

        if not (directory / "study.sqlite3").is_file():
            raise ValueError("The study database is missing.")
    else:
        config = settings.copy()
        config["seed"] = resolve_seed(config["seed"])
        config["dataset_seed"] = resolve_seed(config["dataset_seed"])
        config.setdefault("probe_diagnostics", True)
        config.setdefault("probe_axis_points", 9)
        config.setdefault("probe_near_zero", 1e-12)
        config.setdefault("save_report", True)

    comparison = select_comparison(config["comparison_method"], config["comparison_parameters"])
    equivalence = select_equivalence(
        config["equivalence_method"], config["equivalence_tolerance"],
        config["equivalence_atol"], config["equivalence_rtol"],
    )
    fingerprint = code_fingerprint(comparison, equivalence)

    if resume is not None and manifest["implementation"] != fingerprint:
        raise ValueError("Resume requires the saved implementation, Python version and NumPy version.")

    repetitions = config["repetitions"]

    if not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer.")

    if not isinstance(config["progress_every"], int) or config["progress_every"] < 0:
        raise ValueError("progress_every must be a nonnegative integer.")

    for name in ("probe_diagnostics", "save_report"):
        if not isinstance(config[name], bool):
            raise ValueError(name + " must be a boolean.")

    target_names = config["targets"]

    if not isinstance(target_names, list) or not target_names or len(set(target_names)) != len(target_names):
        raise ValueError("targets must be a nonempty list without duplicates.")

    for name in target_names:
        if name not in POLYNOMIAL_TERMS:
            raise ValueError("Define polynomial terms before studying target: " + name)

    bins = len(config["histogram_edges"]) - 1
    WorkspaceDiversity(comparison, equivalence, config["histogram_edges"])

    if not np.isfinite(config["subtarget_tolerance"]) or config["subtarget_tolerance"] < 0:
        raise ValueError("subtarget_tolerance must be finite and nonnegative.")

    if any(not isinstance(number, int) or number < 1 for number in config["trace_run_ids"]):
        raise ValueError("trace_run_ids must contain positive integer run IDs.")
    combinations = parameter_combinations(config, grid)
    total = repetitions * len(target_names) * len(combinations)
    datasets = {}
    subtargets = {}
    probe_datasets = {}

    if resume is None:
        name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        directory = Path(config["output_dir"]) / name
        (directory / "datasets").mkdir(parents=True, exist_ok=False)
        seeds = seed_plan(config["seed"], repetitions)
        target_metadata = {}

        for name in target_names:
            function, expression = TARGETS[name]
            variables, target = make_study_dataset(function, config)
            datasets[name] = (variables, readonly_target(target))
            columns = list(variables) + ["target"]
            matrix = np.column_stack(list(variables.values()) + [target])
            write_csv(directory / "datasets" / (name + ".csv"), columns, matrix)
            subtargets[name] = make_subtargets(name, variables)

            if config["probe_diagnostics"]:
                probes = make_probe_dataset(function, list(variables), config)
                probe_datasets[name] = probes
                probe_variables, probe_target, groups, masks = probes
                columns = list(probe_variables) + ["target", "groups"]
                rows = zip(*probe_variables.values(), probe_target, groups)
                write_csv(directory / "datasets" / (name + "_probes.csv"), columns, rows)

            lower, upper = config["sample_min"], config["sample_max"]
            target_metadata[name] = {"expression": expression, "sample_min": lower, "sample_max": upper}

        manifest = {
            "version": 14, "schema_version": 3, "configuration": config, "grid": grid,
            "targets": target_metadata, "implementation": fingerprint,
            "dataset_sha256": dataset_fingerprint(directory, target_names, config["probe_diagnostics"]),
            "total_runs": total, "order": "repetition, target, configuration",
            "seed_generation": "Random(seed): unique getrandbits(63); Random(repetition_seed): workspace then derivation getrandbits(63)",
            "dataset_generation": "linspace; two-dimensional meshgrid(indexing='ij'); dataset_seed reserved, no random sampling",
            "uniqueness": "First matching representative in workspace order; finite vectors only",
            "acceptance": "Absolute RMSE <= perfect_fit_tolerance; subtargets use subtarget_tolerance",
            "diagnostics": "Best and final archive expressions; midpoint grid, boundaries, origin, coordinate-zero axes and near-zero points. Zero probes may lie outside the training domain. Diagnostic checks never change acceptance.",
            "timing": "duration_seconds includes derivation, workspace statistics and archive recovery tracking; diagnostic_seconds measures additional probe evaluation. Both exclude database writes and plotting.",
        }
        store = StudyStore(directory / "study.sqlite3", bins)

        try:
            store.initialize(combinations, seeds, subtargets, manifest)
            write_json(directory / "study.json", manifest)
        except BaseException:
            store.close()
            raise
    else:
        for name in target_names:
            variables, target = load_dataset(directory / "datasets" / (name + ".csv"))
            datasets[name] = (variables, target)
            subtargets[name] = make_subtargets(name, variables)

        store = StudyStore(directory / "study.sqlite3", bins)
        store.check_manifest(manifest)
        seeds = list(store.connection.execute("SELECT * FROM seeds ORDER BY repetition"))

    if resume is not None and config["probe_diagnostics"]:
        for name in target_names:
            function, expression = TARGETS[name]
            variables, target = datasets[name]
            probes = make_probe_dataset(function, list(variables), config)
            probe_datasets[name] = probes

    print("Study:", directory.resolve(), flush=True)
    print("Targets:", len(target_names), "Configurations:", len(combinations), "Total runs:", total)
    print("Seed:", config["seed"], "Dataset seed:", config["dataset_seed"])
    completed_now = 0
    started = time.perf_counter()

    try:
        first_run = store.next_run_id()
        print("Already saved:", first_run - 1, flush=True)

        for run_id in range(first_run, total + 1):
            if max_runs is not None and completed_now >= max_runs:
                break

            if store.is_completed(run_id):
                continue

            repetition_index, job_index = divmod(run_id - 1, len(target_names) * len(combinations))
            target_index, configuration_index = divmod(job_index, len(combinations))
            name = target_names[target_index]
            repetition, seed, workspace_seed, derivation_seed = seeds[repetition_index]
            current = {**config, **combinations[configuration_index], "target": name}
            variables, target = datasets[name]
            run_started = time.perf_counter()
            save_trace = run_id in config["trace_run_ids"]
            derivation, recovery, steps = collect_derivation(
                variables, target, current, workspace_seed, derivation_seed, subtargets[name],
                comparison, equivalence, save_trace,
            )
            record = run_record(
                derivation, recovery, run_id, name, configuration_index + 1,
                repetition, time.perf_counter() - run_started,
            )
            diagnostic_started = time.perf_counter()
            diagnostics = run_diagnostics(derivation, recovery, recovery.archive, probe_datasets.get(name))
            diagnostics["diagnostic_seconds"] = time.perf_counter() - diagnostic_started
            store.save_run(
                record, steps, recovery.subtarget_steps, save_trace,
                diagnostics, recovery.archive.subtarget_steps,
            )
            completed_now += 1

            if config["progress_every"] and (completed_now % config["progress_every"] == 0 or run_id == total):
                elapsed = time.perf_counter() - started
                remaining = elapsed * (total - run_id) / completed_now
                print(
                    f"Runs: {run_id}/{total} | target {name} | configuration {configuration_index + 1}"
                    f" | elapsed {elapsed:.1f}s | ETA {remaining:.1f}s", flush=True,
                )
    except KeyboardInterrupt:
        print("\nInterrupted. Completed runs are saved; the unfinished run will restart with the same seeds.")
    finally:
        completed = store.next_run_id() - 1
        store.close()

    print(f"Saved runs: {completed}/{total}")
    print(f'Resume: python main_hyperparameters.py --resume "{directory}"')
    print(f'Export: python main_study_export.py "{directory}"')
    print(f'Report: python main_study_report.py "{directory}"')

    if completed == total and config["save_report"]:
        try:
            create_report(directory)
        except KeyboardInterrupt:
            print("Reporting interrupted. All derivations are saved; run main_study_report.py to finish the report.")
        except (OSError, ValueError, RuntimeError) as error:
            print("Runs are saved. Report generation failed:", error)
            print("Run main_study_report.py to retry reporting without repeating derivations.")

    return directory
