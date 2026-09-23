from itertools import combinations, groupby
from math import isfinite, sqrt

import numpy as np


PARAMETERS = ("esel", "esur", "variable_atom_count", "operator_count")
METRICS = (
    "success_rate", "probe_clean_success_rate", "normalized_rmse",
    "duration_seconds", "merge_count", "merge_limit_rate",
    "recovered_subtarget_fraction", "archived_subtarget_fraction",
)


def central_configuration(rows):
    middle = {}

    for name in PARAMETERS:
        values = sorted({row[name] for row in rows})
        middle[name] = values[len(values) // 2]

    return min(rows, key=lambda row: (
        sum(row[name] != middle[name] for name in PARAMETERS), row["configuration_id"],
    ))["configuration_id"]


def prepare_aggregate(connection, directory, targets):
    connection.execute("DROP TABLE IF EXISTS temp.report_targets")
    connection.execute("DROP TABLE IF EXISTS temp.report_repetitions")
    connection.execute(
        "CREATE TEMP TABLE report_targets (target TEXT PRIMARY KEY, rms REAL, subtargets INTEGER)"
    )
    scales = []

    for target in targets:
        values = np.loadtxt(directory / "datasets" / (target + ".csv"), delimiter=",", skiprows=1)
        expected = np.atleast_2d(values)[:, -1]
        scale = float(np.sqrt(np.mean(expected ** 2)))
        subtargets = connection.execute(
            "SELECT COUNT(*) FROM subtargets WHERE target = ?", (target,),
        ).fetchone()[0]

        if not isfinite(scale) or scale <= 0:
            raise ValueError("Cross-target normalization requires a positive finite target RMS.")

        scales.append((target, scale, subtargets))

    connection.executemany("INSERT INTO report_targets VALUES (?, ?, ?)", scales)
    configurations = connection.execute("SELECT COUNT(*) FROM configurations").fetchone()[0]
    block_size = len(targets) * configurations
    connection.execute("CREATE TEMP TABLE report_repetitions (repetition INTEGER PRIMARY KEY)")
    connection.execute(
        "INSERT INTO report_repetitions SELECT repetition FROM runs JOIN report_targets USING (target) "
        "GROUP BY repetition HAVING COUNT(*) = ?", (block_size,),
    )
    repetitions = connection.execute("SELECT COUNT(*) FROM report_repetitions").fetchone()[0]
    available = connection.execute(
        "SELECT COUNT(*) FROM runs JOIN report_targets USING (target)"
    ).fetchone()[0]

    return {
        "targets": targets, "configurations": configurations,
        "complete_repetition_blocks": repetitions,
        "included_runs": repetitions * block_size,
        "excluded_incomplete_block_runs": available - repetitions * block_size,
        "target_rms": {target: scale for target, scale, count in scales},
        "weighting": "Equal targets and configurations within each complete repetition seed block",
        "dispersion": "Sample standard deviation between repetition seed blocks; not a confidence interval",
        "normalization": "Best RMSE / RMS of the saved target vector; 1 is the zero-predictor error",
    }


def seed_blocks(connection, names, diagnostics, configuration=None):
    if not set(names).issubset({"configuration_id", *PARAMETERS}):
        raise ValueError("Unsupported aggregation column.")

    columns = ", ".join("c." + name for name in names)
    probe = "NULL"
    archived = "NULL"
    join = ""

    if diagnostics:
        join = " LEFT JOIN run_diagnostics d USING (run_id) "
        probe = (
            "CASE WHEN COUNT(d.best_probe_count) = COUNT(*) "
            "THEN AVG(r.success = 1 AND d.best_probe_match = 1) END"
        )
        archived = "AVG(1.0 * d.archived_subtarget_count / NULLIF(t.subtargets, 0))"

    where = ""
    parameters = (float("inf"),)

    if configuration is not None:
        where = " WHERE r.configuration_id = ? "
        parameters += (configuration,)

    query = """
        SELECT """ + columns + """, r.repetition, COUNT(*) AS runs, SUM(r.success) AS successful_runs,
               AVG(r.success) AS success_rate,
               """ + probe + """ AS probe_clean_success_rate,
               CASE WHEN SUM(r.best_fitness IS NOT NULL AND ABS(r.best_fitness) < ?) = COUNT(*)
                    THEN AVG(r.best_fitness / t.rms) END AS normalized_rmse,
               AVG(r.duration_seconds) AS duration_seconds,
               AVG(r.merge_count) AS merge_count,
               AVG(r.termination_reason = 'max_merge') AS merge_limit_rate,
               AVG(1.0 * r.subtarget_count / NULLIF(t.subtargets, 0)) AS recovered_subtarget_fraction,
               """ + archived + """ AS archived_subtarget_fraction
        FROM runs r JOIN configurations c USING (configuration_id)
        JOIN report_targets t USING (target) JOIN report_repetitions USING (repetition)
        """ + join + where + " GROUP BY " + columns + ", r.repetition ORDER BY " + columns + ", r.repetition"

    return connection.execute(query, parameters)


def summarize_blocks(blocks, names, baseline=None):
    rows = []

    for identity, group in groupby(blocks, key=lambda row: tuple(row[name] for name in names)):
        totals = {name: [0, 0.0, 0.0] for name in METRICS}

        if baseline is not None:
            totals["paired_success_difference"] = [0, 0.0, 0.0]

        row = dict(zip(names, identity))
        row["repetition_blocks"] = 0
        row["completed_runs"] = 0
        row["successful_runs"] = 0

        for block in group:
            row["repetition_blocks"] += 1
            row["completed_runs"] += block["runs"]
            row["successful_runs"] += block["successful_runs"]

            for metric, moments in totals.items():
                if metric == "paired_success_difference":
                    value = block["success_rate"] - baseline[block["repetition"]]
                else:
                    value = block[metric]

                if value is None or not isfinite(value):
                    continue

                moments[0] += 1
                difference = value - moments[1]
                moments[1] += difference / moments[0]
                moments[2] += difference * (value - moments[1])

        for metric, (count, mean, squared) in totals.items():
            row[metric + "_mean"] = mean if count else None
            row[metric + "_seed_std"] = sqrt(max(0, squared) / (count - 1)) if count > 1 else None
            row[metric + "_seed_count"] = count

        row["success_rate_mean"] = row["successful_runs"] / row["completed_runs"]

        rows.append(row)

    return rows


def aggregate_results(connection, diagnostics, baseline):
    reference = {
        row["repetition"]: row["success_rate"]
        for row in seed_blocks(connection, ["configuration_id"], diagnostics, baseline)
    }
    configurations = summarize_blocks(
        seed_blocks(connection, ["configuration_id"], diagnostics), ["configuration_id"], reference,
    )
    parameters = []

    for name in PARAMETERS:
        for row in summarize_blocks(seed_blocks(connection, [name], diagnostics), [name]):
            parameters.append({"parameter": name, "value": row.pop(name), **row})

    rules = summarize_blocks(seed_blocks(connection, ["esel", "esur"], diagnostics), ["esel", "esur"])
    sizes = summarize_blocks(
        seed_blocks(connection, ["variable_atom_count", "operator_count"], diagnostics),
        ["variable_atom_count", "operator_count"],
    )

    return configurations, parameters, rules, sizes


def pairwise_aggregates(connection, diagnostics):
    """Balanced summaries for all six hyperparameter pairs."""
    groups = {}
    for first, second in combinations(PARAMETERS, 2):
        names = [first, second]
        groups[first, second] = summarize_blocks(seed_blocks(connection, names, diagnostics), names)
    return groups


def workspace_marginals(connection, target, comparison_method="cosine"):
    """Pool saved per-run means by their actual contributor counts at each step."""
    rows = []
    for name in PARAMETERS:
        query = """
            SELECT c.""" + name + """ AS value, s.step,
                   SUM(s.contributors) AS workspace_runs,
                   SUM((s.fraction_AS_mean + s.fraction_CS_mean) * s.contributors)
                       / SUM(s.contributors) AS semantic_fraction,
                   SUM(s.mean_comparison_count) AS comparison_runs,
                   SUM(s.mean_comparison_mean * s.mean_comparison_count)
                       / NULLIF(SUM(s.mean_comparison_count), 0) AS mean_pair_comparison,
                   COUNT(*) AS configurations
            FROM step_totals s JOIN configurations c USING (configuration_id)
            WHERE s.target = ? GROUP BY c.""" + name + ", s.step ORDER BY c." + name + ", s.step"
        for record in connection.execute(query, (target,)):
            row = {"target": target, "parameter": name, **dict(record)}
            value = row["mean_pair_comparison"]
            if comparison_method == "cosine" and value is not None:
                # Old incremental sums can drift slightly outside cosine's range.
                if abs(value) > 1 + 1e-8:
                    raise ValueError("Saved cosine mean is outside [-1, 1] beyond rounding tolerance.")
                row["mean_pair_comparison"] = min(1.0, max(-1.0, value))
            rows.append(row)
    return rows
