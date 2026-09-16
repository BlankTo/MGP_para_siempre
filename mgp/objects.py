import numpy as np


UNARY_OPERATORS = ("sin", "cos", "log", "sqrt")
BINARY_OPERATORS = ("+", "-", "*", "/")

MATRIX_COLUMNS = [
    "atomic_semantic",
    "composite_semantic",
    "atomic_operator",
    "composite_operator",
    "incomplete",
]


def operator_arity(symbol):
    if symbol in UNARY_OPERATORS:
        return 1

    if symbol in BINARY_OPERATORS:
        return 2

    raise ValueError("Unknown operator: " + symbol)


class MergeTree:
    def __init__(self, label, children=()):
        self.label = label
        self.children = tuple(children)

    def text(self):
        parts = []
        pending = [self]

        while pending:
            current = pending.pop()

            if isinstance(current, str):
                parts.append(current)
            elif not current.children:
                parts.append("{" + current.label + "}")
            else:
                left, right = current.children
                pending.extend(["}", right, " | ", left, "{"])

        return "".join(parts)


class SemanticObject:
    kind = "semantic"

    def __init__(self, expression, values, is_atomic=True):
        values = np.asarray(values, dtype=float)

        if values.ndim != 1 or values.size == 0:
            raise ValueError("A semantic object needs a nonempty one-dimensional vector.")

        self.expression = expression
        self.values = values.copy()
        self.values.setflags(write=False)
        self.is_atomic = is_atomic
        self.structure = MergeTree(expression)
        self.unary_argument = None

        self._fitness_target = None
        self._fitness_values = None
        self._fitness_value = None

    @property
    def category(self):
        if self.is_atomic:
            return "atomic_semantic"

        return "composite_semantic"

    def tree(self):
        return "{" + self.expression + "}"


class OperatorObject:
    kind = "operator"

    def __init__(self, operators, is_atomic=None):
        self.operators = tuple(operators)

        if not self.operators:
            raise ValueError("An operator object needs at least one operator.")

        self.arity = operator_arity(self.operators[0])

        for symbol in self.operators:
            if operator_arity(symbol) != self.arity:
                raise ValueError("An operator object must contain operators of the same arity.")

        if is_atomic is None:
            is_atomic = len(self.operators) == 1

        if is_atomic and len(self.operators) != 1:
            raise ValueError("An atomic operator contains exactly one operator.")

        self.is_atomic = is_atomic
        self.structure = MergeTree(self.expression)

    @property
    def category(self):
        if self.is_atomic:
            return "atomic_operator"

        return "composite_operator"

    @property
    def expression(self):
        return " | ".join(self.operators)

    def tree(self):
        return "{" + self.expression + "}"


class IncompleteObject:
    kind = "incomplete"
    category = "incomplete"
    is_atomic = False

    def __init__(self, semantic, head):
        if not isinstance(semantic, SemanticObject) or not isinstance(head, OperatorObject):
            raise TypeError("An incomplete object needs one semantic object and one operator head.")

        if head.arity != 2:
            raise ValueError("An incomplete object needs a binary operator head.")

        self.semantic = semantic
        self.head = head
        self.structure = MergeTree(self.expression, (semantic.structure, head.structure))

    @property
    def expression(self):
        return self.semantic.expression + " [" + self.head.expression + "] ?"

    def tree(self):
        return "{" + self.semantic.tree() + " | " + self.head.tree() + "}"


def object_counts(workspace):
    counts = {}

    for column in MATRIX_COLUMNS:
        counts[column] = 0

    for current in workspace:
        counts[current.category] += 1

    return [counts[column] for column in MATRIX_COLUMNS]


def workspace_text(workspace, evaluated=False):
    trees = []

    for current in workspace:
        if evaluated:
            trees.append(current.tree())
        else:
            trees.append(current.structure.text())

    return "[" + " | ".join(trees) + "]"
