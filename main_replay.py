import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from main_visual import WorkspaceViewer
from mgp import Derivation, make_workspace
from mgp.study import load_dataset
from mgp.study_export import study_connection
from targets import TARGETS


def replay_inputs(directory, run_id):
    directory = Path(directory)

    with (directory / "study.json").open(encoding="utf-8") as source:
        manifest = json.load(source)

    connection = study_connection(directory)

    try:
        row = connection.execute(
            "SELECT r.*, c.parameters, s.workspace_seed, s.derivation_seed FROM runs r "
            "JOIN configurations c USING (configuration_id) JOIN seeds s USING (repetition) "
            "WHERE r.run_id = ?", (run_id,),
        ).fetchone()

        if row is None:
            raise ValueError("This run_id has not been saved.")

        record = dict(row)
    finally:
        connection.close()

    config = {**manifest["configuration"], **json.loads(record["parameters"]), "target": record["target"]}
    variables, target = load_dataset(directory / "datasets" / (record["target"] + ".csv"))
    workspace = make_workspace(variables, config, record["workspace_seed"])
    derivation = Derivation(target, config, record["derivation_seed"])

    return derivation, workspace, record


def main():
    parser = argparse.ArgumentParser(description="Replay a saved run with its exact dataset, configuration and seeds.")
    parser.add_argument("directory")
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--visual", action="store_true")
    arguments = parser.parse_args()
    derivation, workspace, record = replay_inputs(arguments.directory, arguments.run)
    print("Saved best:", record["best_equation"], "RMSE:", record["best_fitness"])
    print("Workspace seed:", record["workspace_seed"], "Derivation seed:", record["derivation_seed"])

    if arguments.visual:
        expression = TARGETS[record["target"]][1]
        viewer = WorkspaceViewer(derivation, workspace, expression)
        plt.show(block=True)
        viewer.steps.close()
    else:
        derivation.run(workspace, verbose=True)
        print("\n" + "=" * 80)
        print("Best:", derivation.best.expression if derivation.best is not None else None)
        print("Best RMSE:", derivation.best_fitness)
        print("Termination:", derivation.termination_reason)


if __name__ == "__main__":
    main()
