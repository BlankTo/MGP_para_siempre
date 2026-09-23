import random
import textwrap

import matplotlib.pyplot as plt

from config import CONFIG
from mgp import Derivation, make_dataset, make_workspace, resolve_seed
from mgp.objects import workspace_text
from targets import TARGETS


VISUAL_CONFIG = {
    **CONFIG,

    "target": "addition", "unary_operators_on": False, "seed": 7086469724043868593, "dataset_seed": 6863861622991049563,
    #"target": "square_plus", "unary_operators_on": False,
    #"target": "nguyen_1", "unary_operators_on": False,
    #"target": "nguyen_8", "unary_operators_on": True,

    "internal_merge_probability": 0.0,

    "esel": 0.0, #0.5,
    "esur": 0.0, # best left at 0.0 in this script to avoid visual explosion

    "variable_atom_count": 20,
    "operator_count": 10,
}


def layout_workspace(workspace):
    nodes = []
    pending = []
    next_leaf = 0

    for current in reversed(workspace):
        pending.append((current.structure, None))

    while pending:
        tree, parent_index = pending.pop()
        node_index = len(nodes)
        node = {"tree": tree, "parent": parent_index, "children": [], "x": 0, "y": 0}
        nodes.append(node)

        if parent_index is not None:
            nodes[parent_index]["children"].append(node_index)

        if tree.children:
            for child in reversed(tree.children):
                pending.append((child, node_index))
        else:
            node["x"] = next_leaf
            next_leaf += 1

    for node in reversed(nodes):
        if node["children"]:
            left = nodes[node["children"][0]]
            right = nodes[node["children"][1]]
            node["x"] = (left["x"] + right["x"]) / 2
            node["y"] = max(left["y"], right["y"]) + 1

    return nodes


class WorkspaceViewer:
    def __init__(self, derivation, workspace, target_expression):
        self.derivation = derivation
        self.target_expression = target_expression
        self.steps = derivation.steps(workspace)
        self.workspace = next(self.steps)
        self.space_pressed = False

        self.figure, self.axis = plt.subplots(figsize=(16, 9))
        self.figure.canvas.manager.set_window_title("MGP workspace")
        self.figure.subplots_adjust(left=0.03, right=0.97, bottom=0.18, top=0.85)
        self.status = self.figure.text(0.03, 0.12, "", fontsize=11)
        self.figure.text(
            0.03, 0.07,
            "Click inside the window. SPACE: next attempt. Use the toolbar to zoom, pan or save.",
            fontsize=10,
        )
        self.figure.text(
            0.03, 0.035,
            "Yellow: atoms. Blue: MERGE labels. Green: latest result.",
            fontsize=10,
        )

        self.figure.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.figure.canvas.mpl_connect("key_release_event", self.on_key_release)
        self.figure.canvas.mpl_connect("close_event", self.on_close)
        self.draw()

    def on_key_press(self, event):
        if event.key not in (" ", "space") or self.space_pressed:
            return

        self.space_pressed = True

        if self.derivation.termination_reason:
            return

        self.workspace = next(self.steps)
        self.draw()

    def on_key_release(self, event):
        if event.key in (" ", "space"):
            self.space_pressed = False

    def on_close(self, event):
        self.steps.close()

    def draw(self):
        self.axis.clear()
        self.axis.set_axis_off()
        nodes = layout_workspace(self.workspace)
        latest_tree = None

        if self.derivation.last_merge is not None:
            result = self.derivation.last_merge[2]
            latest_tree = result.structure

        for node in nodes:
            if node["parent"] is not None:
                parent = nodes[node["parent"]]
                self.axis.plot(
                    [node["x"], parent["x"]], [node["y"], parent["y"]],
                    color="#64748b", linewidth=1.4, zorder=1,
                )

        for node in nodes:
            color = "#fef3c7"

            if node["children"]:
                color = "#dbeafe"

            if node["tree"] is latest_tree:
                color = "#bbf7d0"

            label = textwrap.fill(node["tree"].label, width=24)
            self.axis.text(
                node["x"], node["y"], label,
                ha="center", va="center", fontsize=9, fontfamily="monospace",
                bbox={"boxstyle": "round,pad=0.4", "facecolor": color, "edgecolor": "#475569"},
                zorder=2,
            )

        if nodes:
            self.axis.set_xlim(-0.7, max(node["x"] for node in nodes) + 0.7)
            self.axis.set_ylim(-0.7, max(node["y"] for node in nodes) + 0.7)
        else:
            self.axis.text(0.5, 0.5, "Empty workspace", ha="center", va="center")

        self.figure.suptitle(
            "Target: " + self.target_expression
            + f"\nMERGE {self.derivation.merge_count}"
            + f" | Attempt {self.derivation.attempt_count}"
            + f" | Workspace objects: {len(self.workspace)}",
            fontsize=14,
        )

        message = "Ready for the next MERGE."

        if self.derivation.termination_reason == "single_object":
            message = "Finished: one object remains."
        elif self.derivation.termination_reason == "max_merge":
            message = "Stopped at the MERGE attempt limit."
        elif self.derivation.termination_reason == "empty_workspace":
            message = "Finished: the workspace is empty."
        elif self.derivation.termination_reason == "no_valid_merge":
            message = "Finished: no valid MERGE remains."

        if self.derivation.best is not None:
            message += f"  Best RMSE: {self.derivation.best_fitness:.8g}"

        if self.derivation.last_merge_kind is not None:
            message += "  Last MERGE: " + self.derivation.last_merge_kind

        self.status.set_text(message)
        self.figure.canvas.draw_idle()


def main():
    seed = resolve_seed(VISUAL_CONFIG["seed"])
    dataset_seed = resolve_seed(VISUAL_CONFIG["dataset_seed"])
    generator = random.Random(seed)
    function, target_expression = TARGETS[VISUAL_CONFIG["target"]]
    variables, target = make_dataset(function, VISUAL_CONFIG, dataset_seed)

    workspace_seed = generator.getrandbits(63)
    derivation_seed = generator.getrandbits(63)
    workspace = make_workspace(variables, VISUAL_CONFIG, workspace_seed)
    derivation = Derivation(target, VISUAL_CONFIG, derivation_seed)

    print("Initial workspace: " + workspace_text(workspace))
    print("Evaluated workspace: " + workspace_text(workspace, evaluated=True))
    print("Seed:", seed, "Dataset seed:", dataset_seed)
    print("Workspace seed:", workspace_seed, "Derivation seed:", derivation_seed)

    viewer = WorkspaceViewer(derivation, workspace, target_expression)
    plt.show(block=True)
    viewer.steps.close()


if __name__ == "__main__":
    main()
