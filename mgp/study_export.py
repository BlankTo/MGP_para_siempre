import csv
from itertools import groupby
import json
from math import isfinite, sqrt
from pathlib import Path
import sqlite3
from statistics import pstdev

from config import CONFIG
from .fitness import nonnegative_mean
from .statistics import write_csv, write_json
from .study_storage import step_metrics


def study_connection(directory):
    path = (Path(directory) / "study.sqlite3").resolve()
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row

    return connection


def write_query(connection, path, query, parameters=()):
    cursor = connection.execute(query, parameters)
    columns = [column[0] for column in cursor.description]
    temporary = path.with_suffix(".tmp")
    write_csv(temporary, columns, cursor)
    temporary.replace(path)


def filter_query(target, configuration_id, prefix=""):
    conditions = []
    parameters = []

    if target is not None:
        conditions.append(prefix + "target = ?")
        parameters.append(target)

    if configuration_id is not None:
        conditions.append(prefix + "configuration_id = ?")
        parameters.append(configuration_id)

    if not conditions:
        return "", parameters

    return " WHERE " + " AND ".join(conditions), parameters


def metric_summary(row, name):
    count = int(row[name + "_count"])

    if count == 0:
        return None, None, 0

    mean = row[name + "_mean"]
    variance = max(0.0, row[name + "_m2"] / count)

    return mean, sqrt(variance), count


def export_steps(connection, output, config, where, parameters):
    metrics = step_metrics()
    columns = ["target", "configuration_id", "step", "contributing_derivations"]

    for name in metrics:
        columns += [name + "_mean", name + "_std", name + "_contributors"]

    histogram_columns = [
        "target", "configuration_id", "step", "bin", "comparison_lower", "comparison_upper",
        "pair_count", "total_valid_pairs", "probability", "contributing_derivations",
    ]
    steps_path = output / "step_summary.csv"
    histogram_path = output / "histograms.csv"
    edges = config["histogram_edges"]
    bins = len(edges) - 1
    query = "SELECT * FROM step_totals" + where + " ORDER BY target, configuration_id, step"

    with steps_path.with_suffix(".tmp").open("w", newline="", encoding="utf-8") as step_file:
        with histogram_path.with_suffix(".tmp").open("w", newline="", encoding="utf-8") as histogram_file:
            step_writer = csv.writer(step_file)
            histogram_writer = csv.writer(histogram_file)
            step_writer.writerow(columns)
            histogram_writer.writerow(histogram_columns)

            for row in connection.execute(query, parameters):
                identity = [row["target"], row["configuration_id"], row["step"]]
                values = identity + [int(row["contributors"])]

                for name in metrics:
                    values.extend(metric_summary(row, name))

                step_writer.writerow(values)
                valid_pairs = sum(row[f"bin_{index:03d}_pairs"] for index in range(bins))

                for index in range(bins):
                    count = int(row[f"bin_{index:03d}_pairs"])
                    pooled = count / valid_pairs if valid_pairs else None
                    histogram_writer.writerow(identity + [
                        index, edges[index], edges[index + 1],
                        count, valid_pairs, pooled, int(row["contributors"]),
                    ])

    steps_path.with_suffix(".tmp").replace(steps_path)
    histogram_path.with_suffix(".tmp").replace(histogram_path)


def ranking_rows(connection, where, parameters):
    query = (
        "SELECT target, configuration_id, first_target_step, merge_count, archive_size, "
        "subtarget_count, termination_reason, final_fitness, best_fitness, duration_seconds "
        "FROM runs" + where + " ORDER BY target, configuration_id, repetition"
    )
    cursor = connection.execute(query, parameters)
    rows = []

    for identity, group in groupby(cursor, key=lambda row: (row["target"], row["configuration_id"])):
        count = 0
        successes = 0
        fitnesses = []
        merge_sum = 0
        archive_sum = 0
        subtarget_sum = 0
        limited = 0
        nonfinite_final = 0
        durations = []
        strict_successes = 0
        candidate_successes = 0
        final_successes = 0

        for run in group:
            count += 1
            successes += run["first_target_step"] is not None
            merge_sum += run["merge_count"]
            archive_sum += run["archive_size"]
            subtarget_sum += run["subtarget_count"]
            limited += run["termination_reason"] == "max_merge"
            durations.append(run["duration_seconds"])
            strict_successes += run["best_fitness"] is not None and run["best_fitness"] <= 1e-12
            candidate_successes += run["best_fitness"] is not None and run["best_fitness"] <= 1e-8
            final_successes += run["final_fitness"] is not None and run["final_fitness"] <= 1e-8

            if run["final_fitness"] is not None and not isfinite(run["final_fitness"]):
                nonfinite_final += 1

            if run["best_fitness"] is not None and isfinite(run["best_fitness"]):
                fitnesses.append(run["best_fitness"])

        rows.append({
            "target": identity[0], "configuration_id": identity[1],
            "completed_runs": count, "successful_runs": successes, "success_rate": successes / count,
            "finite_best_count": len(fitnesses), "mean_best_fitness": nonnegative_mean(fitnesses),
            "std_best_fitness": pstdev(fitnesses) if fitnesses else None,
            "best_fitness": min(fitnesses) if fitnesses else None,
            "mean_merge_count": merge_sum / count, "mean_archive_size": archive_sum / count,
            "mean_subtargets_found": subtarget_sum / count,
            "max_merge_count": limited, "nonfinite_final_count": nonfinite_final,
            "strict_successful_runs": strict_successes,
            "best_within_1e_minus_8_runs": candidate_successes,
            "final_within_1e_minus_8_runs": final_successes,
            "total_duration_seconds": sum(durations),
            "mean_duration_seconds": nonnegative_mean(durations),
            "std_duration_seconds": pstdev(durations),
        })

    return sorted(rows, key=ranking_key)


def ranking_key(row):
    mean = row["mean_best_fitness"]

    if mean is None:
        mean = float("inf")

    return (row["target"], -row["success_rate"], -row["finite_best_count"] / row["completed_runs"], mean, row["configuration_id"])


def export_study(directory, target=None, configuration_id=None):
    directory = Path(directory)

    with (directory / "study.json").open(encoding="utf-8") as source:
        manifest = json.load(source)

    config = manifest["configuration"]

    if target is not None and target not in config["targets"]:
        raise ValueError("Unknown target for this study.")

    output = directory / "csv"

    if target is not None:
        output = output / target

    if configuration_id is not None:
        output = output / f"config_{configuration_id:04d}"

    output.mkdir(parents=True, exist_ok=True)
    connection = study_connection(directory)
    where, parameters = filter_query(target, configuration_id)
    run_where, run_parameters = filter_query(target, configuration_id, "r.")

    try:
        connection.execute("BEGIN")
        saved_runs = connection.execute("SELECT next_run_id - 1 FROM progress").fetchone()[0]
        configurations = list(connection.execute("SELECT * FROM configurations ORDER BY configuration_id"))

        if configuration_id is not None and configuration_id not in [row[0] for row in configurations]:
            raise ValueError("Unknown configuration_id for this study.")

        names = [name for name in CONFIG if name in config and name not in ("seed", "dataset_seed", "target")]
        configuration_rows = []

        for row in configurations:
            current = {**config, **json.loads(row["parameters"])}
            configuration_rows.append([row["configuration_id"]] + [current[name] for name in names])

        write_csv(output / "configurations.csv", ["configuration_id"] + names, configuration_rows)
        write_query(connection, output / "seeds.csv", "SELECT * FROM seeds ORDER BY repetition")
        write_query(connection, output / "runs.csv",
                    "SELECT * FROM completed_runs" + where + " ORDER BY run_id", parameters)
        write_query(connection, output / "subtarget_hits.csv",
                    "SELECT r.run_id, r.target, r.configuration_id, r.repetition, h.subtarget_id, h.first_step "
                    "FROM runs r JOIN subtarget_hits h USING (run_id)" + run_where
                    + " ORDER BY r.run_id, h.subtarget_id", run_parameters)
        write_query(connection, output / "subtargets.csv", "SELECT * FROM subtargets ORDER BY target, subtarget_id")

        if manifest.get("schema_version", 0) >= 3:
            write_query(connection, output / "run_diagnostics.csv",
                        "SELECT r.target, r.configuration_id, r.repetition, d.* FROM runs r "
                        "JOIN run_diagnostics d USING (run_id)" + run_where + " ORDER BY r.run_id", run_parameters)
            write_query(connection, output / "archive_subtarget_hits.csv",
                        "SELECT r.target, r.configuration_id, r.repetition, h.* FROM runs r "
                        "JOIN archive_subtarget_hits h USING (run_id)" + run_where
                        + " ORDER BY r.run_id, h.subtarget_id", run_parameters)
        write_query(connection, output / "subtarget_summary.csv", """
            WITH totals AS (
                SELECT target, configuration_id, COUNT(*) AS completed_runs FROM runs
        """ + where + """ GROUP BY target, configuration_id
            ), hits AS (
                SELECT r.target, r.configuration_id, h.subtarget_id, COUNT(*) AS recovered_runs
                FROM runs r JOIN subtarget_hits h USING (run_id)
        """ + run_where + """ GROUP BY r.target, r.configuration_id, h.subtarget_id
            )
            SELECT t.target, t.configuration_id, s.subtarget_id, s.expression,
                   t.completed_runs, COALESCE(h.recovered_runs, 0) AS recovered_runs,
                   1.0 * COALESCE(h.recovered_runs, 0) / t.completed_runs AS recovery_rate
            FROM totals t JOIN subtargets s ON t.target = s.target
            LEFT JOIN hits h ON h.target = t.target AND h.configuration_id = t.configuration_id
                            AND h.subtarget_id = s.subtarget_id
            ORDER BY t.target, t.configuration_id, s.subtarget_id
        """, parameters + run_parameters)
        export_steps(connection, output, config, where, parameters)
        rows = ranking_rows(connection, where, parameters)

        if rows:
            write_csv(output / "ranking.csv", list(rows[0]), [list(row.values()) for row in rows])

        if config["trace_run_ids"]:
            write_query(connection, output / "run_steps.csv",
                        "SELECT r.target, r.configuration_id, r.repetition, s.* FROM runs r "
                        "JOIN run_steps s USING (run_id)" + run_where + " ORDER BY r.run_id, s.step", run_parameters)

        write_json(output / "export.json", {
            "saved_runs_in_snapshot": saved_runs, "planned_runs": manifest["total_runs"],
            "target_filter": target, "configuration_filter": configuration_id,
        })
    finally:
        connection.close()

    print("CSV files:", output.resolve())

    return output
