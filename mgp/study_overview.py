import matplotlib.pyplot as plt
import numpy as np

from .study_plots import PARAMETER_LABELS, configuration_label, heatmap, save_figure, target_label


def plot_global_heatmaps(rules, sizes, folder):
    figure, axes = plt.subplots(3, 2, figsize=(12, 12), layout="constrained")
    metrics = (
        ("success_rate_mean", "Mean target success", "Success (%)", True),
        ("normalized_rmse_mean", "Mean normalized error", "RMSE / target RMS (lower is better)", False),
        ("duration_seconds_mean", "Mean recorded runtime", "Seconds / run", False),
    )

    for panels, (metric, title, label, percent) in zip(axes, metrics):
        values = [row[metric] for row in rules + sizes if row[metric] is not None]
        maximum = max(values, default=1) * (100 if percent else 1) or 1
        heatmap(panels[0], rules, "esel", "esur", metric, title + ": selection / survival", label, percent, maximum)
        heatmap(panels[1], sizes, "variable_atom_count", "operator_count", metric, title + ": workspace size", label, percent, maximum)

    figure.suptitle("Across targets | Equal target weights; other hyperparameters averaged", fontsize=13)

    return save_figure(figure, folder, "01_global_hyperparameters")


def plot_global_effects(rows, folder):
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained")
    metrics = (
        ("success_rate", "Mean target success (%)", 100),
        ("duration_seconds", "Mean recorded seconds / run", 1),
    )

    for column, (name, title) in enumerate(PARAMETER_LABELS.items()):
        selected = [row for row in rows if row["parameter"] == name]
        values = [row["value"] for row in selected]

        for panels, (metric, label, scale) in zip(axes, metrics):
            axis = panels[column]
            means = np.array([
                row[metric + "_mean"] if row[metric + "_mean"] is not None else np.nan for row in selected
            ]) * scale
            deviation = np.array([
                row[metric + "_seed_std"] if row[metric + "_seed_std"] is not None else np.nan for row in selected
            ]) * scale
            mean_label = "Target recovered" if metric == "success_rate" else "Mean"
            axis.plot(values, means, marker="o", color="#2563a6", label=mean_label)
            axis.fill_between(values, means - deviation, means + deviation, alpha=0.2, label="±1 SD between seeds")

            axis.set_xlabel(title)
            axis.set_ylabel(label)
            axis.set_xticks(values, [f"{value:g}" for value in values])
            axis.set_ylim(bottom=0)
            axis.grid(alpha=0.2)

        axes[0, column].set_title(title)

    axes[0, 0].legend(fontsize=8)
    count = rows[0]["repetition_blocks"]
    figure.suptitle(f"Across targets | {count} paired seed blocks", fontsize=13)

    return save_figure(figure, folder, "02_global_hyperparameter_effects")


def plot_global_rankings(rows, ranking, targets, baseline, folder):
    selected = sorted(rows, key=lambda row: (
        -row["success_rate_mean"],
        row["normalized_rmse_mean"] if row["normalized_rmse_mean"] is not None else float("inf"),
        row["configuration_id"],
    ))[:12]

    if baseline not in [row["configuration_id"] for row in selected]:
        selected.append(next(row for row in rows if row["configuration_id"] == baseline))

    values = np.full((len(selected), len(targets)), np.nan)
    index = {(row["target"], row["configuration_id"]): row for row in ranking}
    figure, axis = plt.subplots(figsize=(12, max(5, len(selected) * 0.48)), layout="constrained")

    for position, row in enumerate(selected):
        for column, target in enumerate(targets):
            record = index[target, row["configuration_id"]]
            values[position, column] = 100 * record["success_rate"]
            axis.text(
                column, position, f"{record['successful_runs']}/{record['completed_runs']}",
                ha="center", va="center", fontsize=8,
                color="white" if values[position, column] < 50 else "black",
            )

    picture = axis.imshow(values, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(targets)), [target_label(name) for name in targets])
    axis.set_yticks(range(len(selected)), [configuration_label(row) for row in selected], fontsize=8)
    axis.set_title("Success by target (labels = successes / runs)")
    figure.colorbar(picture, ax=axis, label="Success (%)", shrink=0.7)
    figure.suptitle("Across targets | Highest observed success rates and reference", fontsize=13)

    return save_figure(figure, folder, "03_global_success_differences")


def pareto_frontier(rows):
    best = -1
    frontier = []
    ordered = sorted(rows, key=lambda row: (
        row["duration_seconds_mean"], -row["success_rate_mean"], row["configuration_id"],
    ))

    for row in ordered:
        if row["success_rate_mean"] > best:
            frontier.append(row)
            best = row["success_rate_mean"]

    return frontier




def plot_global_subtargets(rows, folder):
    figure, axes = plt.subplots(1, 4, figsize=(15, 4.5), layout="constrained")
    maximum = max(
        row["recovered_subtarget_fraction_mean"] or 0 for row in rows
    ) * 100

    for axis, (name, label) in zip(axes, PARAMETER_LABELS.items()):
        selected = [row for row in rows if row["parameter"] == name]
        values = [row["value"] for row in selected]

        for metric, title, color in (
            ("recovered_subtarget_fraction", "Materialized", "#2563a6"),
            ("archived_subtarget_fraction", "Retained in archive", "#278276"),
        ):
            means = [
                100 * row[metric + "_mean"] if row[metric + "_mean"] is not None else np.nan for row in selected
            ]
            if np.isfinite(means).any():
                axis.plot(values, means, marker="o", label=title, color=color)

        axis.set_xlabel(label)
        axis.set_xticks(values, [f"{value:g}" for value in values])
        axis.set_ylim(0, max(1, maximum * 1.2))
        axis.set_ylabel("Mean subtarget coverage (%)")
        axis.grid(alpha=0.2)
        axis.set_title(label)

    axes[0].legend(fontsize=8)
    figure.suptitle("Across targets | Subtarget coverage; each target normalized by its own number of subtargets", fontsize=12)

    return save_figure(figure, folder, "05_global_subtarget_coverage")




def plot_subtarget_surface(rows, folder):
    variables = sorted({row["variable_atom_count"] for row in rows})
    operators = sorted({row["operator_count"] for row in rows})
    x, y = np.meshgrid(variables, operators)
    z = np.full(x.shape, np.nan)
    for row in rows:
        value = row["recovered_subtarget_fraction_mean"]
        if value is not None:
            z[operators.index(row["operator_count"]), variables.index(row["variable_atom_count"])] = 100 * value
    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    if len(variables) > 1 and len(operators) > 1:
        surface = axis.plot_surface(x, y, z, cmap="viridis", edgecolor="#ffffff", linewidth=0.5,
                                    alpha=0.85, vmin=0, vmax=max(1, np.nanmax(z)))
        figure.colorbar(surface, ax=axis, shrink=0.55, pad=0.12, label="Materialized subtargets (%)")
    axis.scatter(x.ravel(), y.ravel(), z.ravel(), color="#203040", s=20, depthshade=False)
    axis.set_xlabel("Variable count", labelpad=12)
    axis.set_ylabel("Operator count", labelpad=12)
    # A figure-level label stays inside the exported bounding box in 3D.
    figure.text(0.02, 0.5, "Mean coverage (%)", rotation=90, va="center")
    axis.set_xticks(variables)
    axis.set_yticks(operators)
    axis.set_zlim(0, max(1, float(np.nanmax(z)) * 1.15))
    axis.view_init(elev=28, azim=-135)
    axis.set_title("Across targets | Materialized subtarget coverage\nESEL and ESUR averaged; equal target weights", pad=18)
    figure.subplots_adjust(left=0.04, right=0.85, bottom=0.05, top=0.91)
    return save_figure(figure, folder, "07_global_subtarget_coverage_3d")
