from collections import defaultdict
from html import escape
from itertools import combinations
import json
from math import sqrt
from pathlib import Path

from .statistics import write_csv, write_json
from .study_aggregate import (
    PARAMETERS, aggregate_results, central_configuration, pairwise_aggregates,
    prepare_aggregate, workspace_marginals,
)
from .study_export import export_steps, ranking_rows, study_connection, write_query
from .study_plots import (
    plot_archive, plot_coverage, plot_heatmaps, plot_overview,
    plot_pair_heatmaps, plot_runtime, plot_workspace_marginals,
)
from .study_overview import (
    pareto_frontier, plot_global_effects, plot_global_heatmaps,
    plot_global_rankings, plot_global_subtargets, plot_subtarget_surface,
)


def save_rows(path, rows):
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        write_csv(temporary, list(rows[0]), [list(row.values()) for row in rows])
        temporary.replace(path)


def wilson_interval(successes, count):
    if count == 0:
        return None, None

    probability = successes / count
    factor = 1 + 1.96 ** 2 / count
    center = (probability + 1.96 ** 2 / (2 * count)) / factor
    radius = 1.96 * sqrt(probability * (1 - probability) / count + 1.96 ** 2 / (4 * count ** 2)) / factor

    return max(0.0, center - radius), min(1.0, center + radius)


def paired_comparisons(connection, target, baseline):
    query = """
        SELECT r.configuration_id, COUNT(*) AS paired_runs,
               SUM(r.success = 1 AND b.success = 0) AS configuration_only,
               SUM(r.success = 0 AND b.success = 1) AS baseline_only,
               AVG(r.success - b.success) AS success_difference,
               AVG(r.duration_seconds - b.duration_seconds) AS duration_difference_seconds,
               AVG(r.duration_seconds / NULLIF(b.duration_seconds, 0)) AS mean_paired_duration_ratio
        FROM runs r JOIN runs b ON r.target = b.target AND r.repetition = b.repetition
        WHERE r.target = ? AND b.configuration_id = ?
        GROUP BY r.configuration_id ORDER BY r.configuration_id
    """
    rows = []

    for item in connection.execute(query, (target, baseline)):
        row = {"target": target, "baseline_configuration_id": baseline, **dict(item)}
        count = row["paired_runs"]
        discordance = (row["configuration_only"] + row["baseline_only"]) / count
        variance = max(0.0, discordance - row["success_difference"] ** 2)
        row["paired_standard_error"] = sqrt(variance / (count - 1)) if count > 1 else None
        rows.append(row)

    return rows


def marginal_rows(rows, first, second):
    groups = defaultdict(list)

    for row in rows:
        groups[row[first], row[second]].append(row)

    result = []

    for identity, group in sorted(groups.items()):
        count = sum(row["completed_runs"] for row in group)
        successes = sum(row["successful_runs"] for row in group)
        result.append({
            first: identity[0], second: identity[1], "completed_runs": count,
            "successful_runs": successes, "success_rate": successes / count,
            "mean_duration_seconds": sum(row["total_duration_seconds"] for row in group) / count,
            "mean_merge_count": sum(row["mean_merge_count"] * row["completed_runs"] for row in group) / count,
        })

    return result


def parameter_effects(rows):
    result = []

    for name in ("esel", "esur", "variable_atom_count", "operator_count"):
        for value in sorted({row[name] for row in rows}):
            selected = [row for row in rows if row[name] == value]
            count = sum(row["completed_runs"] for row in selected)
            success = sum(row["successful_runs"] for row in selected)
            result.append({
                "parameter": name, "value": value, "completed_runs": count,
                "successful_runs": success, "success_rate": success / count,
                "mean_duration_seconds": sum(row["total_duration_seconds"] for row in selected) / count,
                "total_duration_seconds": sum(row["total_duration_seconds"] for row in selected),
                "mean_merge_count": sum(row["mean_merge_count"] * row["completed_runs"] for row in selected) / count,
                "merge_limit_runs": sum(row["max_merge_count"] for row in selected),
            })

    return result


def subtarget_summary(connection, target, has_diagnostics):
    count = connection.execute("SELECT COUNT(*) FROM runs WHERE target = ?", (target,)).fetchone()[0]
    recovered = dict(connection.execute(
        "SELECT h.subtarget_id, COUNT(*) FROM subtarget_hits h JOIN runs r USING (run_id) "
        "WHERE r.target = ? GROUP BY h.subtarget_id", (target,),
    ))
    archived = {}

    if has_diagnostics:
        archived = dict(connection.execute(
            "SELECT h.subtarget_id, COUNT(*) FROM archive_subtarget_hits h JOIN runs r USING (run_id) "
            "WHERE r.target = ? GROUP BY h.subtarget_id", (target,),
        ))

    rows = []

    for identifier, expression in connection.execute(
        "SELECT subtarget_id, expression FROM subtargets WHERE target = ? ORDER BY subtarget_id", (target,),
    ):
        found = recovered.get(identifier, 0)
        retained = archived.get(identifier, 0) if has_diagnostics else None
        rows.append({
            "target": target, "subtarget_id": identifier, "expression": expression,
            "monomial_count": identifier.bit_count(), "completed_runs": count,
            "recovered_runs": found, "archived_runs": retained,
            "recovery_rate": found / count,
            "archive_rate": retained / count if retained is not None else None,
            "retention_given_recovery": retained / found if retained is not None and found else None,
        })

    return rows


def coverage_rows(connection, target, table):
    query = """
        WITH hits AS (
            SELECT h.run_id, COUNT(*) AS components
            FROM """ + table + """ h JOIN runs r USING (run_id)
            WHERE r.target = ? AND (h.subtarget_id & (h.subtarget_id - 1)) = 0
            GROUP BY h.run_id
        )
        SELECT COALESCE(h.components, 0) AS monomials, COUNT(*) AS runs,
               SUM(r.success) AS successful_runs
        FROM runs r LEFT JOIN hits h USING (run_id)
        WHERE r.target = ? GROUP BY COALESCE(h.components, 0) ORDER BY monomials
    """

    return [dict(row) for row in connection.execute(query, (target, target))]


def diagnostic_summary(connection, target):
    query = """
        SELECT r.configuration_id, COUNT(*) AS diagnostic_runs,
               SUM(d.archived_subtarget_count) AS archived_subtarget_count,
               SUM(d.archived_monomial_count) AS archived_monomial_count,
               SUM(d.recovered_monomial_count) AS recovered_monomial_count,
               SUM(d.first_archived_target_step IS NOT NULL) AS target_archived_runs,
               COUNT(d.best_probe_count) AS best_checked_count,
               SUM(d.best_nonfinite_count > 0) AS best_nonfinite_runs,
               SUM(d.best_origin_nonfinite) AS best_origin_nonfinite_runs,
               SUM(d.best_axis_nonfinite_count > 0) AS best_axis_nonfinite_runs,
               SUM(d.best_near_zero_nonfinite_count > 0) AS best_near_zero_nonfinite_runs,
               SUM(d.best_intermediate_nonfinite_count > 0) AS best_intermediate_nonfinite_runs,
               SUM(r.success = 1 AND d.best_probe_count IS NOT NULL) AS successful_checked_count,
               SUM(r.success = 1 AND d.best_nonfinite_count > 0) AS successful_nonfinite_runs,
               SUM(r.success = 1 AND d.best_intermediate_nonfinite_count > 0) AS successful_intermediate_nonfinite_runs,
               SUM(r.success = 1 AND d.best_probe_match = 1) AS successful_probe_match_runs,
               SUM(d.archive_checked_count) AS archive_checked_count,
               SUM(d.archive_nonfinite_count) AS archive_nonfinite_count,
               SUM(d.archive_intermediate_nonfinite_count) AS archive_intermediate_nonfinite_count,
               SUM(d.diagnostic_seconds) AS diagnostic_seconds
        FROM runs r JOIN run_diagnostics d USING (run_id)
        WHERE r.target = ? GROUP BY r.configuration_id ORDER BY r.configuration_id
    """

    return [dict(row) for row in connection.execute(query, (target,))]


def selected_configurations(rows, baseline, requested):
    available = {row["configuration_id"] for row in rows}

    if requested:
        if not set(requested).issubset(available):
            raise ValueError("A selected configuration has no completed runs for this target.")

        return list(dict.fromkeys(requested))

    baseline_row = next(row for row in rows if row["configuration_id"] == baseline)
    matching = [row for row in rows if all(
        row[name] == baseline_row[name] for name in ("variable_atom_count", "operator_count")
    )]
    least_elite = min(matching, key=lambda row: (row["esur"], row["esel"]))
    persistent = max(matching, key=lambda row: (row["esur"], row["esel"]))

    return list(dict.fromkeys([least_elite["configuration_id"], baseline, persistent["configuration_id"]]))


def target_overview(targets, ranking, diagnostics, scales):
    result = []

    for target in targets:
        rows = [row for row in ranking if row["target"] == target]

        if not rows:
            continue

        checked = diagnostics[target]
        probe_count = sum(row["best_checked_count"] for row in checked)
        count = sum(row["completed_runs"] for row in rows)
        finite = sum(row["finite_best_count"] for row in rows)
        error_sum = sum((row["mean_best_fitness"] or 0) * row["finite_best_count"] for row in rows)
        result.append({
            "target": target, "runs": count,
            "successes": sum(row["successful_runs"] for row in rows),
            "successes_at_1e_minus_12": sum(row["strict_successful_runs"] for row in rows),
            "single_final_matches_at_1e_minus_8": sum(row["final_within_1e_minus_8_runs"] for row in rows),
            "probe_clean_successes": sum(row["successful_probe_match_runs"] or 0 for row in checked) if probe_count else None,
            "accepted_best_nonfinite": sum(row["successful_nonfinite_runs"] or 0 for row in checked) if probe_count else None,
            "target_archived": sum(row["target_archived_runs"] for row in checked) if checked else None,
            "mean_subtargets_recovered": sum(row["mean_subtargets_found"] * row["completed_runs"] for row in rows) / count,
            "mean_subtargets_archived": sum(row["archived_subtarget_count"] for row in checked) / count if checked else None,
            "target_rms": scales[target], "finite_best_runs": finite,
            "mean_normalized_rmse": error_sum / finite / scales[target] if finite else None,
            "total_duration_seconds": sum(row["total_duration_seconds"] for row in rows),
            "diagnostic_seconds": sum(row["diagnostic_seconds"] or 0 for row in checked) if checked else None,
            "merge_limit_runs": sum(row["max_merge_count"] for row in rows),
        })

    return result


def create_global_report(connection, directory, output, targets, ranking, diagnostics, configuration_map, baseline):
    metadata = prepare_aggregate(connection, directory, targets)
    overview = target_overview(targets, ranking, diagnostics, metadata["target_rms"])
    save_rows(output / "data" / "target_summary.csv", overview)
    figures = []
    folder = output / "overview"
    folder.mkdir(exist_ok=True)
    data = output / "data" / "overview"
    data.mkdir(parents=True, exist_ok=True)
    reference = baseline if baseline is not None else central_configuration(list(configuration_map.values()))
    metadata["baseline_configuration_id"] = reference
    write_json(output / "metadata" / "aggregation.json", metadata)

    if not metadata["complete_repetition_blocks"]:
        return figures, metadata, []

    has_diagnostics = any(diagnostics.values())
    configurations, effects, rules, sizes = aggregate_results(connection, has_diagnostics, reference)

    for row in configurations:
        settings = configuration_map[row["configuration_id"]]

        for name in ("esel", "esur", "variable_atom_count", "operator_count"):
            row[name] = settings[name]

    save_rows(data / "configuration_summary.csv", configurations)
    save_rows(data / "parameter_effects.csv", effects)
    save_rows(data / "rule_marginals.csv", rules)
    save_rows(data / "size_marginals.csv", sizes)
    save_rows(data / "cost_success_frontier.csv", pareto_frontier(configurations))
    figures += plot_global_heatmaps(rules, sizes, folder)
    figures += plot_global_effects(effects, folder)

    if metadata["excluded_incomplete_block_runs"]:
        ranking = ranking_rows(connection,
            " WHERE repetition IN (SELECT repetition FROM report_repetitions) "
            "AND target IN (SELECT target FROM report_targets)", (),
        )

    figures += plot_global_rankings(configurations, ranking, targets, reference, folder)
    figures += plot_global_subtargets(effects, folder)
    groups = pairwise_aggregates(connection, has_diagnostics)
    save_rows(data / "hyperparameter_pairs.csv", pair_rows(groups))
    figures += plot_pair_heatmaps("Across targets", groups, (
        ("success_rate_mean", "Mean target success", "Success (%)", True),
        ("normalized_rmse_mean", "Normalized error", "RMSE / target RMS", False),
        ("duration_seconds_mean", "Runtime", "Mean seconds / run", False),
    ), folder, "06_global_hyperparameter_pairs")
    figures += plot_subtarget_surface(sizes, folder)
    runtime = [{
        "parameter": row["parameter"], "value": row["value"],
        "mean_duration_seconds": row["duration_seconds_mean"],
        "total_duration_seconds": row["duration_seconds_mean"] * row["completed_runs"],
        "mean_merge_count": row["merge_count_mean"],
    } for row in effects]
    figures += plot_runtime("Across targets", runtime, folder, "08_global_runtime")

    return figures, metadata, overview_findings(configurations, effects, reference)


def overview_findings(rows, effects, baseline):
    lines = [
        "### Observed marginal associations", "",
        "| Parameter | Level with highest mean success | Mean success | Mean seconds/run |",
        "| --- | --- | --- | --- |",
    ]

    for name in ("esel", "esur", "variable_atom_count", "operator_count"):
        levels = [row for row in effects if row["parameter"] == name]
        best = max(levels, key=lambda row: row["success_rate_mean"])
        lines.append(
            f"| {name} | {best['value']:g} | {100 * best['success_rate_mean']:.3f}% "
            f"| {best['duration_seconds_mean']:.4f} |"
        )

    lines += [
        "", "Each row averages over the other parameters. These levels do not automatically form "
        "the best joint configuration. The target matrix shows which functions drive the average.", "",
        "| Configuration | esel | esur | Variables | Operators | Successes / runs | Mean target success | Mean seconds/run |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    best_rows = sorted(rows, key=lambda row: (-row["success_rate_mean"], row["configuration_id"]))[:5]

    for row in best_rows:
        lines.append(
            f"| C{row['configuration_id']} | {row['esel']:g} | {row['esur']:g} "
            f"| {row['variable_atom_count']} | {row['operator_count']} "
            f"| {row['successful_runs']}/{row['completed_runs']} "
            f"| {100 * row['success_rate_mean']:.3f}% | {row['duration_seconds_mean']:.4f} |"
        )

    lines += [
        "", f"These are observed rankings. C{baseline} is the reference for paired differences. "
        "Read the saved seed-block dispersion and per-target breakdown before interpreting a small "
        "difference as a reliable improvement. Validate selected settings on new seeds.", "",
    ]

    return lines


def write_gallery(output, figures):
    groups = defaultdict(list)

    for path in figures:
        relative = path.relative_to(output)
        group = relative.parent.as_posix() if relative.parent.as_posix() != "." else "Target overview"
        groups[group].append(relative)

    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1"><title>MGP study figures</title>',
        '<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1400px;margin:30px auto;padding:0 24px;'
        'color:#203040;background:#f5f7fa}a{color:#176096}nav a{display:inline-block;margin:0 18px 10px 0}'
        'figure{margin:24px 0;padding:16px;background:white;border:1px solid #d8dfe5;border-radius:8px}'
        'img{width:100%;height:auto}figcaption{margin-top:10px}summary{font-size:1.3rem;cursor:pointer;'
        'padding:12px 0}details{margin-bottom:16px}h1{margin-bottom:0}</style>',
        '<h1>MGP study figures</h1><p>Open a section below. Click any image for its scalable SVG version.</p>',
        '<nav><a href="docs/report.md">Findings and methods</a><a href="docs/PLOT_GUIDE.md">Complete plot guide</a>'
        '<a href="data/target_summary.csv">Exact target results</a></nav>',
    ]

    for group, paths in groups.items():
        opened = " open" if group in ("Target overview", "overview") else ""
        parts.append("<details" + opened + "><summary>" + escape(group.replace("_", " ")) + "</summary>")

        for path in paths:
            label = path.stem.replace("_", " ")
            source = escape(path.as_posix(), quote=True)
            vector = escape(path.with_suffix(".svg").as_posix(), quote=True)
            parts.append(
                f'<figure><a href="{vector}"><img loading="lazy" src="{source}" alt="{escape(label)}"></a>'
                f'<figcaption>{escape(label)} · <a href="{vector}">SVG</a></figcaption></figure>'
            )

        parts.append("</details>")

    parts.append("</html>")
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")


def target_findings(target, rows, pairs, subtargets, diagnostics):
    count = sum(row["completed_runs"] for row in rows)
    success = sum(row["successful_runs"] for row in rows)
    strongest = max(rows, key=lambda row: (row["success_rate"], -row["configuration_id"]))
    fastest = min(rows, key=lambda row: row["mean_duration_seconds"])
    slowest = max(rows, key=lambda row: row["mean_duration_seconds"])
    lower, upper = wilson_interval(strongest["successful_runs"], strongest["completed_runs"])
    lines = [
        f"## {target}", "",
        f"Recorded recoveries: **{success:,}/{count:,} ({100 * success / count:.2f}%)**.", "",
        f"Highest observed rate: configuration {strongest['configuration_id']}, "
        f"{strongest['successful_runs']}/{strongest['completed_runs']} successes. "
        f"Its individual 95% Wilson interval is {100 * lower:.1f}% to {100 * upper:.1f}%. "
        "This interval does not correct for selecting the best configuration after examining the grid.", "",
        f"Mean recorded duration ranges from {fastest['mean_duration_seconds']:.6g} s "
        f"(configuration {fastest['configuration_id']}) to {slowest['mean_duration_seconds']:.6g} s "
        f"(configuration {slowest['configuration_id']}).", "",
    ]
    strict = sum(row["strict_successful_runs"] for row in rows)
    relaxed = sum(row["best_within_1e_minus_8_runs"] for row in rows)
    lines += [
        f"Saved best fitness passes 1e-12 in {strict} runs and 1e-8 in {relaxed} runs "
        f"({relaxed - strict} additional numerical matches at the new threshold).", "",
        "| Parameter | Lowest observed marginal rate | Highest observed marginal rate |",
        "| --- | --- | --- |",
    ]
    effects = parameter_effects(rows)

    for name in ("esel", "esur", "variable_atom_count", "operator_count"):
        levels = [row for row in effects if row["parameter"] == name]
        lowest = min(levels, key=lambda row: row["success_rate"])
        highest = max(levels, key=lambda row: row["success_rate"])
        lines.append(
            f"| {name} | {lowest['value']:g}: {100 * lowest['success_rate']:.2f}% "
            f"| {highest['value']:g}: {100 * highest['success_rate']:.2f}% |"
        )

    lines += ["", "These are pooled marginal associations; interactions and shared repetition seeds matter.", ""]
    most_survival = max((row for row in effects if row["parameter"] == "esur"), key=lambda row: row["value"])
    duration = sum(row["total_duration_seconds"] for row in rows)

    if duration:
        lines += [
            f"ESUR={most_survival['value']:g} accounts for "
            f"{100 * most_survival['total_duration_seconds'] / duration:.1f}% of recorded time. "
            f"{most_survival['merge_limit_runs']}/{most_survival['completed_runs']} of its runs "
            "reach the MERGE limit.", "",
        ]

    if pairs:
        difference = max(pairs, key=lambda row: row["success_difference"])
        lines += [
            f"Largest observed gain over baseline {difference['baseline_configuration_id']}: "
            f"configuration {difference['configuration_id']}, {100 * difference['success_difference']:+.2f} "
            f"percentage points across {difference['paired_runs']} paired seeds "
            f"({difference['configuration_only']} wins, {difference['baseline_only']} losses).", "",
        ]

    if diagnostics:
        checked = sum(row["successful_checked_count"] or 0 for row in diagnostics)
        bad = sum(row["successful_nonfinite_runs"] or 0 for row in diagnostics)
        intermediate = sum(row["successful_intermediate_nonfinite_runs"] or 0 for row in diagnostics)
        retained = sum(row["archived_subtarget_count"] for row in diagnostics)
        lines += [
            f"Among {checked} accepted best expressions checked on the probe set, {bad} have "
            f"nonfinite outputs and {intermediate} have at least one nonfinite intermediate operation. "
            "These counts can overlap.", "",
            f"Mean distinct subtargets retained per run: {retained / count:.3f} out of {len(subtargets)}. "
            "Admission is measured directly, not inferred from discovery.", "",
        ]

    return lines



def pair_rows(groups):
    rows = []
    for (first, second), group in groups.items():
        for item in group:
            row = {"first_parameter": first, "second_parameter": second,
                   "first_value": item[first], "second_value": item[second]}
            row.update({key: value for key, value in item.items() if key not in (first, second)})
            rows.append(row)
    return rows


def clean_previous_report(output, targets):
    """Remove only retired report figures and migrate known old report tables."""
    retired = [output / "12_target_outcomes",
               output / "overview" / "02_global_parameter_effects",
               output / "overview" / "04_global_cost_tradeoff"]
    for target in targets:
        for stem in ("03_success_differences", "04_first_recovery", "05_workspace_dynamics",
                     "06_pair_distributions", "07_archived_components", "09_domain_diagnostics",
                     "10_workspace_composition"):
            retired.append(output / target / stem)
    for stem in retired:
        for suffix in (".png", ".svg"):
            stem.with_suffix(suffix).unlink(missing_ok=True)
    tables = ("paired_comparisons", "rule_marginals", "size_marginals", "parameter_effects",
              "subtargets", "recovered_monomial_coverage", "archived_monomial_coverage",
              "diagnostics", "selected_configurations", "step_summary", "histograms", "first_recovery")
    moves = [(output / (name + ".csv"), output / "data" / (name + ".csv"))
             for name in ("configuration_summary", "target_summary")]
    for target in targets:
        moves += [(output / target / (name + ".csv"), output / "data" / target / (name + ".csv"))
                  for name in tables]
    moves += [(output / "overview" / (name + ".csv"), output / "data" / "overview" / (name + ".csv"))
              for name in ("configuration_summary", "parameter_effects", "rule_marginals",
                           "size_marginals", "cost_success_frontier")]
    moves += [(output / name, output / "docs" / name) for name in ("report.md", "PLOT_GUIDE.md")]
    moves += [(output / name, output / "metadata" / name) for name in ("report.json", "study_metadata.json")]
    moves.append((output / "overview" / "aggregation.json", output / "metadata" / "aggregation.json"))
    for old, new in moves:
        if old.is_file():
            new.parent.mkdir(parents=True, exist_ok=True)
            if new.exists():
                old.unlink()
            else:
                old.replace(new)

def create_report(directory, target=None, baseline=None, configurations=None, output=None):
    directory = Path(directory)
    manifest = json.loads((directory / "study.json").read_text(encoding="utf-8"))
    settings = manifest["configuration"]
    targets = [target] if target is not None else settings["targets"]

    if not set(targets).issubset(settings["targets"]):
        raise ValueError("Unknown target for this study.")

    connection = study_connection(directory)
    if output is None:
        output = directory / "report"

        if target is not None:
            output = output / target

    output = Path(output)

    output.mkdir(parents=True, exist_ok=True)
    for subfolder in ("data", "metadata", "docs"):
        (output / subfolder).mkdir(exist_ok=True)
    figures = []
    findings = []
    selections = {}
    diagnostics_by_target = {}

    try:
        connection.execute("BEGIN")
        completed = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        configuration_map = {row["configuration_id"]: dict(row) for row in connection.execute("SELECT * FROM configurations")}

        if baseline is not None and baseline not in configuration_map:
            raise ValueError("Unknown baseline configuration.")
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        has_diagnostics = "run_diagnostics" in tables
        where, parameters = (" WHERE target = ?", (target,)) if target is not None else ("", ())
        ranking = ranking_rows(connection, where, parameters)
        print(f"Reporting {completed:,} completed runs across {len(targets)} target(s).", flush=True)

        for row in ranking:
            parameters = configuration_map[row["configuration_id"]]

            for name in ("esel", "esur", "variable_atom_count", "operator_count"):
                row[name] = parameters[name]

            row["wilson_lower"], row["wilson_upper"] = wilson_interval(row["successful_runs"], row["completed_runs"])

        save_rows(output / "data" / "configuration_summary.csv", ranking)

        if ranking:
            figures += plot_overview(ranking, output)

        for name in targets:
            rows = [row for row in ranking if row["target"] == name]

            if not rows:
                continue

            print("Generating figures:", name, flush=True)

            available = {row["configuration_id"] for row in rows}
            reference = baseline if baseline is not None else central_configuration(rows)

            if reference not in available:
                raise ValueError("The baseline has no completed runs for target " + name)

            folder = output / name
            folder.mkdir(exist_ok=True)
            data = output / "data" / name
            data.mkdir(parents=True, exist_ok=True)
            pairs = paired_comparisons(connection, name, reference)
            save_rows(data / "paired_comparisons.csv", pairs)
            rules = marginal_rows(rows, "esel", "esur")
            sizes = marginal_rows(rows, "variable_atom_count", "operator_count")
            save_rows(data / "rule_marginals.csv", rules)
            save_rows(data / "size_marginals.csv", sizes)
            effects = parameter_effects(rows)
            save_rows(data / "parameter_effects.csv", effects)
            subtargets = subtarget_summary(connection, name, has_diagnostics)
            save_rows(data / "subtargets.csv", subtargets)
            coverage = coverage_rows(connection, name, "subtarget_hits")
            save_rows(data / "recovered_monomial_coverage.csv", coverage)
            diagnostics = diagnostic_summary(connection, name) if has_diagnostics else []
            diagnostics_by_target[name] = diagnostics
            save_rows(data / "diagnostics.csv", diagnostics)
            selected = selected_configurations(rows, reference, configurations)
            selections[name] = {"baseline": reference, "detail_csv_configurations": selected}
            save_rows(data / "selected_configurations.csv", [
                {key: configuration_map[identifier][key] for key in (
                    "configuration_id", "esel", "esur", "variable_atom_count", "operator_count",
                )} for identifier in selected
            ])
            placeholders = ", ".join("?" for identifier in selected)
            export_steps(connection, data, settings,
                " WHERE target = ? AND configuration_id IN (" + placeholders + ")", [name] + selected,
            )
            write_query(connection, data / "first_recovery.csv",
                "SELECT configuration_id, first_target_step, COUNT(*) AS runs FROM runs "
                "WHERE target = ? GROUP BY configuration_id, first_target_step "
                "ORDER BY configuration_id, first_target_step", (name,),
            )
            figures += plot_heatmaps(name, rules, sizes, folder)
            figures += plot_runtime(name.replace("nguyen_", "N").upper(), effects, folder, "02_runtime")
            figures += plot_coverage(name, coverage, folder)
            figures += plot_archive(name, subtargets, folder)
            groups = {(first, second): marginal_rows(rows, first, second)
                      for first, second in combinations(PARAMETERS, 2)}
            save_rows(data / "hyperparameter_pairs.csv", pair_rows(groups))
            figures += plot_pair_heatmaps(name.replace("nguyen_", "N").upper(), groups, (
                ("success_rate", "Target success", "Success (%)", True),
                ("mean_duration_seconds", "Runtime", "Mean seconds / run", False),
            ), folder, "12_hyperparameter_pairs")
            evolution = workspace_marginals(connection, name, settings["comparison_method"])
            save_rows(data / "workspace_hyperparameter_curves.csv", evolution)
            figures += plot_workspace_marginals(name, evolution, folder,
                "semantic_fraction", "10_semantic_fraction")
            figures += plot_workspace_marginals(name, evolution, folder,
                "mean_pair_comparison", "11_semantic_similarity", settings["comparison_method"])

            if diagnostics:
                archived = coverage_rows(connection, name, "archive_subtarget_hits")
                save_rows(data / "archived_monomial_coverage.csv", archived)

            findings += target_findings(name, rows, pairs, subtargets, diagnostics)

        print("Generating the cross-target overview.", flush=True)
        for name in targets:
            diagnostics_by_target.setdefault(name, [])

        overview_figures, aggregate_metadata, global_findings = create_global_report(
            connection, directory, output, targets, ranking, diagnostics_by_target, configuration_map, baseline,
        )
        figures = overview_figures + figures
    finally:
        connection.close()

    text = [
        "# MGP study report", "",
        "Start with `../00_overview.png`, then `../overview/01_global_hyperparameters.png`. "
        "See [PLOT_GUIDE.md](PLOT_GUIDE.md) for every figure, its denominator and its source table.", "",
        f"Snapshot: **{completed:,}/{manifest['total_runs']:,} completed derivations**.", "",
        f"Recorded target acceptance: absolute RMSE <= {settings['perfect_fit_tolerance']:.6g}. "
        f"Subtarget acceptance: absolute RMSE <= {settings['subtarget_tolerance']:.6g}.", "",
        "Each configuration summary uses its completed runs as the denominator. Paired differences "
        "use only common repetition seeds within one target. Marginal heatmaps pool completed runs "
        "over the other two parameters; unbalanced partial studies can confound these comparisons. "
        "Repeated configurations under one seed are not independent repetitions.", "",
        "The threshold figure compares best RMSE <= 1e-12 and <= 1e-8. "
        "These are retrospective checks of saved best fitness, not new recovery times "
        "or proofs of algebraic equality. Equivalence for diversity remains independently configured.", "",
        "Recorded duration includes the derivation and its online statistics, including archive recovery "
        "tracking. Probe time is saved separately. Disk writes, setup and reporting are excluded. "
        "Timings depend on the machine and background load; paired ratios are descriptive observations.", "",
        "Workspace curves group all configurations by one hyperparameter at a time. Each point "
        "averages equally over the runs still observed at that step, including step 0; terminated runs "
        "are not carried forward. Semantic fraction is (AS + CS) / workspace size in each run. "
        "Mean pair comparison averages only defined per-run means, not all pooled pairs. "
        "Contributor counts are saved in data/TARGET/workspace_hyperparameter_curves.csv. "
        "The selected configuration IDs affect only the detailed step and histogram CSVs.", "",
        "Subtargets use unsigned monomials and exclude the complete unsigned combination. Recovery "
        "means an expression materialized; archive retention means it passed actual archive admission. "
        "The current archive never evicts, so admitted subtargets remain at run end. Finding individual "
        "parts at different steps does not imply that they coexisted or were combined.", "",
        "Probe checks cover saved best expressions and final archive entries on deterministic extra "
        "points, including zero, coordinate-zero axes, near zero and boundaries. They do not change "
        "fitness, MERGE or acceptance. A finite output can still involve invalid intermediate operations. "
        "Probe failure rates estimate problems on these specific points, not their probability over the "
        "whole continuous domain. Passing probes does not prove domain-wide validity or equality.", "",
    ]
    text += [
        "## Across-target overview", "",
        f"Comparable seed blocks: **{aggregate_metadata['complete_repetition_blocks']}**; "
        f"included runs: {aggregate_metadata['included_runs']:,}; completed runs in incomplete blocks: "
        f"{aggregate_metadata['excluded_incomplete_block_runs']:,} (excluded from the overview).", "",
        "Each target has equal weight. The same repetition seed remains one block across all targets "
        "and configurations. Error is divided by the RMS of that target's saved values, so a zero "
        "predictor has normalized error 1. This reporting normalization never changes RMSE fitness or "
        "the static acceptance threshold. Bands describe variation between seed blocks, not confidence "
        "intervals. Raw RMSE values from different targets are not averaged together.", "",
        "A block's normalized error is undefined if any required best fitness is nonfinite or missing; "
        "metric-specific seed counts are saved. Probe-clean success is undefined when required probe "
        "checks are missing. Per-target figures include all completed runs; the balanced overview may "
        "therefore use fewer runs in a partial study.", "",
    ]

    if not aggregate_metadata["complete_repetition_blocks"]:
        text += ["No complete common repetition block is available yet; cross-target comparison figures are omitted.", ""]

    if not has_diagnostics:
        text += [
            "**This older study has no probe or archive-subtarget records.** Those figures cannot be "
            "reconstructed from discovery counts or scalar fitness alone. Re-run with v13 to collect them.", "",
        ]

    text += global_findings + findings
    text += ["## Figures", ""]

    for path in figures:
        text += [f"![{path.stem.replace('_', ' ')}](../{path.relative_to(output).as_posix()})", ""]

    text += [
        "## Suggested interpretation", "",
        "Start with success counts and uncertainty, then inspect paired differences and runtime. "
        "Use workspace dynamics to form hypotheses about those differences. Check numerical sensitivity, "
        "probe failures and archive retention before calling a configuration reliable. Validate promising "
        "settings on additional independent seeds; these exploratory rankings are not confirmatory tests.", "",
        "CSV files retain exact counts and denominators. Figures are exported as PNG and SVG. "
        "Use main_study_export.py for detailed per-run records and step aggregates.", "",
    ]
    temporary = output / "docs" / "report.tmp"
    temporary.write_text("\n".join(text), encoding="utf-8")
    temporary.replace(output / "docs" / "report.md")
    guide = Path(__file__).resolve().parent.parent / "PLOT_GUIDE.md"
    (output / "docs" / "PLOT_GUIDE.md").write_text(guide.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(output / "metadata" / "study_metadata.json", manifest)
    clean_previous_report(output, targets)
    write_gallery(output, figures)
    write_json(output / "metadata" / "report.json", {
        "report_version": 15,
        "completed_runs": completed, "planned_runs": manifest["total_runs"],
        "targets": targets, "resolved_configurations": selections,
        "cross_target": aggregate_metadata, "has_diagnostics": has_diagnostics,
        "figures": [path.relative_to(output).as_posix() for path in figures],
    })
    print("Report:", (output / "index.html").resolve())

    return output
