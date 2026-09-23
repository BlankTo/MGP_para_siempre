from itertools import combinations

from .fitness import object_fitness, rmse
from .objects import SemanticObject


POLYNOMIAL_TERMS = {
    "g1": [("x1", 1, 1), ("x1", 2, 1)],
    "g2": [("x1", 2, 1), ("x2", 2, 1)],
    "nguyen_1": [("x1", power, 1) for power in range(1, 4)],
    "nguyen_2": [("x1", power, 1) for power in range(1, 5)],
    "nguyen_3": [("x1", power, 1) for power in range(1, 6)],
    "nguyen_4": [("x1", power, 1) for power in range(1, 7)],
    "nguyen_12": [("x1", 4, 1), ("x1", 3, -1), ("x2", 2, 1), ("x2", 1, -1)],
}


def make_subtargets(target_name, variables):
    terms = POLYNOMIAL_TERMS[target_name]
    subtargets = []

    for size in range(1, len(terms)):
        for indices in combinations(range(len(terms)), size):
            values = None
            labels = []
            identifier = 0

            for index in indices:
                variable, power, sign = terms[index]
                term = variables[variable] ** power
                label = variable

                if power != 1:
                    label += "**" + str(power)

                labels.append(label)
                identifier += 1 << index

                if values is None:
                    values = term.copy()
                else:
                    values += term

            values.setflags(write=False)
            subtargets.append((identifier, " + ".join(labels), values))

    return subtargets


class RecoveryTracker:
    def __init__(self, target, subtargets, tolerance, subtarget_tolerance=None):
        self.target = target
        self.subtargets = subtargets
        self.tolerance = tolerance
        self.subtarget_tolerance = tolerance if subtarget_tolerance is None else subtarget_tolerance
        self.first_target_step = None
        self.subtarget_steps = {}
        self.seen_values = set()

    def observe(self, current, step):
        if not isinstance(current, SemanticObject):
            return

        if self.first_target_step is None and object_fitness(current, self.target) <= self.tolerance:
            self.first_target_step = step

        key = current.values.tobytes()

        if key in self.seen_values:
            return

        self.seen_values.add(key)

        for identifier, expression, values in self.subtargets:
            if identifier not in self.subtarget_steps and rmse(current.values, values) <= self.subtarget_tolerance:
                self.subtarget_steps[identifier] = step
