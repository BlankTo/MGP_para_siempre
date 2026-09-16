import random

import numpy as np

from .archive import Archive
from .fitness import best_index, object_fitness, readonly_target
from .merge import ExternalMerge, InternalMerge
from .objects import MATRIX_COLUMNS, SemanticObject, object_counts, workspace_text


class Derivation:
    def __init__(self, target, config, seed):
        self.target = readonly_target(target)

        for key in ("esel", "esur", "fitness_probability", "internal_merge_probability"):
            if not 0 <= config[key] <= 1:
                raise ValueError(key + " must be between 0 and 1.")

        if not isinstance(config["max_merge"], int) or config["max_merge"] < 0:
            raise ValueError("max_merge must be a nonnegative integer.")

        if not np.isfinite(config["perfect_fit_tolerance"]) or config["perfect_fit_tolerance"] < 0:
            raise ValueError("perfect_fit_tolerance must be finite and nonnegative.")

        self.config = config.copy()
        self.seed = seed

    def consider(self, current, source):
        if not isinstance(current, SemanticObject):
            return

        fitness = object_fitness(current, self.target)

        if self.best is None or fitness < self.best_fitness:
            self.best = current
            self.best_fitness = fitness
            self.best_source = source

    def run(self, workspace, archive=None, input_so=None, verbose=False):
        for current_workspace in self.steps(workspace, archive, input_so, verbose):
            pass

        return self.best

    def steps(self, workspace, archive=None, input_so=None, verbose=False):
        self.workspace = list(workspace)
        self.best = None
        self.best_fitness = float("inf")
        self.best_source = ""

        self.merge_count = 0
        self.attempt_count = 0
        self.step_counts = []
        self.initial_counts = object_counts(self.workspace)
        self.current_counts = {}
        self.last_merge = None

        for index, column in enumerate(MATRIX_COLUMNS):
            self.current_counts[column] = self.initial_counts[index]

        if archive is None:
            archive = Archive(
                self.config["archive_max_size"],
                self.config["archive_similarity"],
                self.config["archive_atomic_objects"],
            )

        self.archive = archive
        generator = random.Random(self.seed)
        merge_seed = generator.getrandbits(63)
        external_merge = ExternalMerge(self.target, self.config["fitness_probability"], merge_seed)
        internal_merge = InternalMerge()
        internal_probability = self.config["internal_merge_probability"]

        if verbose:
            print("Initial workspace: " + workspace_text(self.workspace))
            print("Evaluated workspace: " + workspace_text(self.workspace, evaluated=True))

        candidates = list(self.workspace) + list(archive.objects)

        if input_so is not None:
            if not isinstance(input_so, SemanticObject):
                raise TypeError("input_so must be semantic or None.")

            candidates.append(input_so)

        for current in candidates:
            semantic = current

            if current.kind == "incomplete":
                semantic = current.semantic

            if isinstance(semantic, SemanticObject) and semantic.values.shape != self.target.shape:
                raise ValueError("All objects must use the target dataset.")

        self.consider(input_so, "input_so")

        for current in self.workspace:
            self.consider(current, "initial_workspace")

        for current in archive.objects:
            self.consider(current, "input_archive")

        self.update_termination()
        yield self.workspace.copy()

        while not self.termination_reason:
            self.attempt_count += 1

            if verbose:
                print("\n" + "=" * 80)

            if internal_probability > 0 and generator.random() < internal_probability:
                current = generator.choice(self.workspace)
                internal_merge.merge(current)
                continue

            elite_index = None

            if self.config["esel"] > 0 or self.config["esur"] > 0:
                elite_index = best_index(self.workspace, self.target)

            if elite_index is not None and generator.random() < self.config["esel"]:
                left_index = elite_index
            else:
                left_index = generator.randrange(len(self.workspace))

            right_index = generator.randrange(len(self.workspace) - 1)

            if right_index >= left_index:
                right_index += 1

            left = self.workspace[left_index]
            right = self.workspace[right_index]
            result = external_merge.merge(left, right)
            self.last_merge = (left, right, result)

            preserve_index = None

            if elite_index in (left_index, right_index) and generator.random() < self.config["esur"]:
                preserve_index = elite_index

            for index in sorted((left_index, right_index), reverse=True):
                if index != preserve_index:
                    self.current_counts[self.workspace[index].category] -= 1
                    self.workspace.pop(index)

            self.workspace.append(result)
            self.current_counts[result.category] += 1
            self.merge_count += 1
            self.step_counts.append([self.current_counts[column] for column in MATRIX_COLUMNS])
            self.consider(result, f"merge_{self.merge_count}")

            if self.config["archive_enabled"]:
                self.archive.add(result)

            if verbose:
                print(
                    f"MERGE {self.merge_count}:"
                    f" left={left.structure.text()} right={right.structure.text()}"
                    f" -> {result.structure.text()}"
                )
                print(
                    f"Evaluated MERGE: left={left.tree()} right={right.tree()}"
                    f" -> {result.tree()} ({result.category})"
                )

                if isinstance(result, SemanticObject):
                    print("Result RMSE:", object_fitness(result, self.target))

                print("Workspace: " + workspace_text(self.workspace))
                print("Evaluated workspace: " + workspace_text(self.workspace, evaluated=True))

                if self.best is not None:
                    print(
                        f"Best: {self.best.expression}; RMSE={self.best_fitness:.8g};"
                        f" source={self.best_source}"
                    )

            self.update_termination()
            yield self.workspace.copy()

    def update_termination(self):
        self.final_object = None
        self.termination_reason = ""

        if not self.workspace:
            self.termination_reason = "empty_workspace"
        elif len(self.workspace) == 1:
            self.termination_reason = "single_object"
            self.final_object = self.workspace[0]
        elif self.attempt_count >= self.config["max_merge"]:
            self.termination_reason = "max_merge"

    def best_location(self):
        if self.best is None:
            return "none"

        locations = []

        for current in self.workspace:
            if current is self.best:
                locations.append("workspace")
                break

        for current in self.archive.objects:
            if current is self.best:
                locations.append("archive")
                break

        if not locations:
            locations.append("best_so_far")

        return " | ".join(locations)
