import matplotlib.pyplot as plt
import numpy as np

from .recovery import POLYNOMIAL_TERMS


PARAMETER_LABELS = {
    "esel": "Elite selection (esel)", "esur": "Elite survival (esur)",
    "variable_atom_count": "Variable count", "operator_count": "Operator count",
}


def target_label(name):
    return name.replace("nguyen_", "N").upper()


def configuration_label(row):
    return (
        f"C{row['configuration_id']}: select={row['esel']:g}, survive={row['esur']:g}, "
        f"variables={row['variable_atom_count']}, operators={row['operator_count']}"
    )


def save_figure(figure, folder, name):
    path = folder / (name + ".png")

    try:
        figure.savefig(path, dpi=170, bbox_inches="tight")
        figure.savefig(folder / (name + ".svg"), bbox_inches="tight")
    finally:
        plt.close(figure)

    return [path]


def heatmap(axis, rows, first, second, metric, title, label, percent=False, maximum=None):
    horizontal = sorted({row[second] for row in rows})
    vertical = sorted({row[first] for row in rows})
    values = np.full((len(vertical), len(horizontal)), np.nan)

    for row in rows:
        if row[metric] is not None:
            values[vertical.index(row[first]), horizontal.index(row[second])] = row[metric]

    if percent:
        values *= 100

    if maximum is None:
        finite = values[np.isfinite(values)]
        maximum = max(float(finite.max()), 1e-12) if finite.size else 1

    maximum = max(maximum, 1e-12)
    picture = axis.imshow(values, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=maximum)
    axis.set_xticks(range(len(horizontal)), [f"{value:g}" for value in horizontal])
    axis.set_yticks(range(len(vertical)), [f"{value:g}" for value in vertical])
    axis.set_xlabel(PARAMETER_LABELS[second])
    axis.set_ylabel(PARAMETER_LABELS[first])
    axis.set_title(title)

    for row_index, column_index in np.ndindex(values.shape):
        value = values[row_index, column_index]

        if np.isfinite(value):
            text = f"{value:.2f}" if percent else f"{value:.2g}"
            color = "white" if value < 0.5 * maximum else "black"
            axis.text(column_index, row_index, text, ha="center", va="center", color=color, fontsize=8)
        else:
            axis.text(column_index, row_index, "N/A", ha="center", va="center", fontsize=8)

    axis.figure.colorbar(picture, ax=axis, label=label, shrink=0.85)


def plot_heatmaps(target, rules, sizes, folder):
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    title = target_label(target)
    maximum = max(100 * row["success_rate"] for row in rules + sizes) or 1
    runtime = max(row["mean_duration_seconds"] for row in rules + sizes) or 1
    heatmap(axes[0, 0], rules, "esel", "esur", "success_rate", "Success by selection / survival", "Success (%)", True, maximum)
    heatmap(axes[0, 1], sizes, "variable_atom_count", "operator_count", "success_rate", "Success by initial workspace", "Success (%)", True, maximum)
    heatmap(axes[1, 0], rules, "esel", "esur", "mean_duration_seconds", "Runtime by selection / survival", "Mean seconds / run", maximum=runtime)
    heatmap(axes[1, 1], sizes, "variable_atom_count", "operator_count", "mean_duration_seconds", "Runtime by initial workspace", "Mean seconds / run", maximum=runtime)
    figure.suptitle(title + " | Other parameters pooled; color scales apply within this target", fontsize=13)

    return save_figure(figure, folder, "01_hyperparameters")



def plot_overview(rows, folder):
    targets = list(dict.fromkeys(row["target"] for row in rows))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    positions = np.arange(len(targets))

    for offset, key, label in (
        (-0.18, "strict_successful_runs", "Best RMSE ≤ 1e-12"),
        (0.18, "best_within_1e_minus_8_runs", "Best RMSE ≤ 1e-8"),
    ):
        counts = [sum(row[key] for row in rows if row["target"] == target) for target in targets]
        totals = [sum(row["completed_runs"] for row in rows if row["target"] == target) for target in targets]
        rates = [100 * count / total for count, total in zip(counts, totals)]
        bars = axes[0].bar(positions + offset, rates, width=0.34, label=label)
        axes[0].bar_label(bars, labels=[str(count) for count in counts], rotation=90, fontsize=8, padding=3)

    for offset, key, label in (
        (-0.18, "best_within_1e_minus_8_runs", "Best at any step"),
        (0.18, "final_within_1e_minus_8_runs", "Single final semantic object"),
    ):
        counts = [sum(row[key] for row in rows if row["target"] == target) for target in targets]
        bars = axes[1].bar(positions + offset, counts, width=0.34, label=label)
        axes[1].bar_label(bars, rotation=90, fontsize=8, padding=3)

    for axis in axes:
        axis.set_xticks(positions, [target_label(name) for name in targets])
        axis.legend(fontsize=8, loc="upper right")
        axis.set_ylim(bottom=0)
        axis.margins(y=0.25)

    axes[0].set_ylabel("Runs (%)")
    axes[0].set_title("Acceptance thresholds · labels: run counts")
    axes[1].set_ylabel("Runs with RMSE ≤ 1e-8")
    axes[1].set_title("Best during derivation and final object")
    figure.suptitle("Study overview", fontsize=14)
    return save_figure(figure, folder, "00_overview")


def plot_pair_heatmaps(label, groups, metrics, folder, filename):
    """One row per pair; each metric keeps a common color scale."""
    figure, axes = plt.subplots(
        len(groups), len(metrics), figsize=(6 * len(metrics), 3.9 * len(groups)),
        squeeze=False, layout="constrained",
    )
    for column, (metric, title, unit, percent) in enumerate(metrics):
        values = [row[metric] for rows in groups.values() for row in rows if row[metric] is not None]
        maximum = max(values, default=1) * (100 if percent else 1) or 1
        for panels, ((first, second), rows) in zip(axes, groups.items()):
            heatmap(panels[column], rows, first, second, metric, title, unit, percent, maximum)
    figure.suptitle(label + " | All hyperparameter pairs; other values averaged", fontsize=14)
    return save_figure(figure, folder, filename)


def plot_runtime(label, effects, folder, filename):
    figure = plt.figure(figsize=(14, 8), layout="constrained")
    grid = figure.add_gridspec(2, 6)
    length_axis = figure.add_subplot(grid[0, :2])
    axes = [figure.add_subplot(grid[0, 2:4]), figure.add_subplot(grid[0, 4:]),
            figure.add_subplot(grid[1, :3]), figure.add_subplot(grid[1, 3:])]
    survival = [row for row in effects if row["parameter"] == "esur"]
    values = [row["value"] for row in survival]
    length_axis.plot(values, [row["mean_merge_count"] for row in survival], marker="o", color="#2563a6")
    length_axis.set_xticks(values, [f"{value:g}" for value in values])
    length_axis.set_xlabel(PARAMETER_LABELS["esur"])
    length_axis.set_ylabel("Mean MERGEs / run")
    length_axis.set_ylim(bottom=0)
    length_axis.set_title("Derivation length")
    length_axis.grid(alpha=0.15)

    for axis, (name, title) in zip(axes, PARAMETER_LABELS.items()):
        rows = [row for row in effects if row["parameter"] == name]
        total_time = sum(row["total_duration_seconds"] for row in rows)
        durations = [row["mean_duration_seconds"] for row in rows]
        shares = [100 * row["total_duration_seconds"] / total_time if total_time else 0 for row in rows]
        bars = axis.bar(range(len(rows)), durations, color="#2563a6")
        axis.bar_label(bars, labels=[f"{value:.1f}%" for value in shares], fontsize=8, padding=3)
        axis.set_xticks(range(len(rows)), [f"{row['value']:g}" for row in rows])
        axis.set_xlabel(title)
        axis.set_ylabel("Mean seconds / run")
        axis.set_title("Runtime by " + title.lower())
        axis.margins(y=0.18)
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    figure.suptitle(label + " | Runtime; bar labels: share of total time", fontsize=14)
    return save_figure(figure, folder, filename)


def plot_coverage(target, rows, folder, source="recovered"):
    count = len(POLYNOMIAL_TERMS[target])
    successes = np.zeros(count + 1, dtype=int)
    failures = np.zeros(count + 1, dtype=int)
    for row in rows:
        successes[row["monomials"]] = row["successful_runs"]
        failures[row["monomials"]] = row["runs"] - row["successful_runs"]

    figure, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    axis.bar(range(count + 1), failures, label="Target not recovered", color="#aebdcc")
    axis.bar(range(count + 1), successes, bottom=failures, label="Target recovered", color="#278276")
    maximum = max(1, int(np.max(successes + failures)))
    for position, (success, failure) in enumerate(zip(successes, failures)):
        if failure:
            if failure >= maximum * 0.065:
                axis.text(position, failure / 2, str(failure), ha="center", va="center", fontsize=9, color="#203040")
            else:
                offset = (-28, 12) if success else (0, 18)
                axis.annotate(str(failure), (position, failure / 2), xytext=offset,
                              textcoords="offset points", ha="center", color="#203040", fontsize=9,
                              arrowprops={"arrowstyle": "-", "color": "#203040", "lw": 0.8})
        if success:
            center = failure + success / 2
            if success >= maximum * 0.065:
                axis.text(position, center, str(success), ha="center", va="center", fontsize=9, color="white")
            else:
                # Very small segments need an external count, with a leader to that segment.
                axis.annotate(str(success), (position, center), xytext=(0, 18),
                              textcoords="offset points", ha="center", color="#17665b", fontsize=9,
                              arrowprops={"arrowstyle": "-", "color": "#17665b", "lw": 0.8})
    axis.set_ylim(0, maximum * 1.18)
    axis.set_xticks(range(count + 1))
    axis.set_xlabel("Distinct unsigned monomials " + source)
    axis.set_ylabel("Runs")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
    axis.set_title(target_label(target) + " | Recovered components; labels: run counts")
    return save_figure(figure, folder, "07_" + source + "_components")


def plot_archive(target, subtargets, folder):
    monomials = [row for row in subtargets if row["monomial_count"] == 1]
    figure, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    positions = np.arange(len(monomials))
    axis.bar(positions - 0.18, [100 * row["recovery_rate"] for row in monomials], 0.35, label="Recovered")
    if any(row["archive_rate"] is not None for row in monomials):
        axis.bar(positions + 0.18, [100 * row["archive_rate"] if row["archive_rate"] is not None else np.nan
                                  for row in monomials], 0.35, label="Retained in archive")
    axis.set_xticks(positions, [row["expression"] for row in monomials], rotation=15)
    axis.set_ylabel("Runs (%)")
    axis.set_ylim(0, 100)
    axis.set_xlabel("Unsigned monomial")
    axis.legend(fontsize=9)
    axis.set_title(target_label(target) + " | Archive retention")
    return save_figure(figure, folder, "08_archive_retention")


def plot_workspace_marginals(target, rows, folder, metric, filename, method="cosine"):
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    fraction = metric == "semantic_fraction"
    for axis, (name, title) in zip(axes.flat, PARAMETER_LABELS.items()):
        selected = [row for row in rows if row["parameter"] == name]
        levels = sorted({row["value"] for row in selected})
        colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.86, len(levels)))
        for value, color in zip(levels, colors):
            observations = [row for row in selected if row["value"] == value]
            values = [row[metric] if row[metric] is not None else np.nan for row in observations]
            axis.plot([row["step"] for row in observations], values, label=f"{value:g}", color=color,
                      marker="o" if len(observations) == 1 else None, linewidth=1.6)
        axis.set_xlim(0, max(1, max((row["step"] for row in selected), default=0)))
        if fraction:
            axis.set_ylim(0, 1)
        elif method == "cosine":
            axis.set_ylim(-1, 1)
        else:
            axis.set_ylim(bottom=0)
        axis.set_xlabel("MERGE step")
        axis.set_ylabel("Mean semantic fraction" if fraction else "Mean pair comparison (" + method + ")")
        axis.set_title(title)
        axis.legend(title="Value", fontsize=8, title_fontsize=8, ncol=min(5, len(levels)))
        axis.grid(alpha=0.2)
    title = "Semantic fraction (AS + CS)" if fraction else "Mean pair comparison"
    figure.suptitle(target_label(target) + " | " + title + "; runs observed at each step", fontsize=14)
    return save_figure(figure, folder, filename)
