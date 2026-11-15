from weakref import ref

import numpy as np

from mgp.engine.expression_graph import graph_text
from mgp.engine.operator_sequence import OperatorSequence, operator_count

UNARY_OPERATORS = ('sin', 'cos', 'log', 'sqrt')
BINARY_OPERATORS = ('+', '-', '*', '/')
MATRIX_COLUMNS = ['atomic_semantic', 'composite_semantic', 'atomic_operator', 'composite_operator', 'incomplete']


def operator_arity(symbol):
    if symbol in UNARY_OPERATORS:
        return 1
    if symbol in BINARY_OPERATORS:
        return 2
    raise ValueError('unknown operator: ' + symbol)


def apply_operator(symbol, left, right=None):
    left = np.asarray(left, dtype=float)
    undefined = np.isnan(left)
    failed = np.isinf(left)
    if right is not None:
        right = np.asarray(right, dtype=float)
        undefined = undefined | np.isnan(right)
        failed = failed | np.isinf(right)
    if symbol == '/':
        undefined = undefined | (right == 0)
    elif symbol == 'log':
        undefined = undefined | (left <= 0)
    elif symbol == 'sqrt':
        undefined = undefined | (left < 0)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore', under='ignore'):
        if symbol == '+':
            values = left + right
        elif symbol == '-':
            values = left - right
        elif symbol == '*':
            values = left * right
        elif symbol == '/':
            values = left / right
        elif symbol == 'sin':
            values = np.sin(left)
        elif symbol == 'cos':
            values = np.cos(left)
        elif symbol == 'log':
            values = np.log(left)
        elif symbol == 'sqrt':
            values = np.sqrt(left)
        else:
            raise ValueError('unknown operator: ' + symbol)
    failed = failed | ~np.isfinite(values) & ~undefined
    operands = (left,) if right is None else (left, right)
    failed = failed | zero_underflow(symbol, operands, values)
    return np.where(failed, np.inf, np.where(undefined, np.nan, values))


class MergeTree:

    def __init__(self, label, children=()):
        self.label = label
        self.children = tuple(children)
        self.lazy_label = None

    def text(self):
        parts = []
        pending = [self]
        while pending:
            current = pending.pop()
            if isinstance(current, str):
                parts.append(current)
            elif not current.children:
                label = current.label if current.lazy_label is None else current.lazy_label().expression
                parts.append('{' + label + '}')
            else:
                left, right = current.children
                pending.extend(['}', right, ' | ', left, '{'])
        return ''.join(parts)


class SemanticObject:
    kind = 'semantic'

    def __init__(self, expression, values, is_atomic=True):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError('semantic object needs a nonempty one-dimensional vector')
        self._expression = expression
        self.evaluation = expression
        self.values = values.copy()
        self.values.setflags(write=False)
        self.is_atomic = is_atomic
        self.structure = MergeTree(expression if expression is not None else '')
        if expression is None:
            self.structure.lazy_label = ref(self)
        self.children = ()
        self.internal_accessible = True
        self.accessible_count = 1
        self.unary_argument = None
        self.evaluation_children = ()
        self.used_semantic_children = ()
        self.used_operator = False
        self.semantic_identity = object()
        self.archive_origin = False
        self.ic_alternatives = ()
        self.occurrence_identity = object()
        self.symbolic_identity = None
        self._fitness_values = None
        self._fitness_cache = ()

    @property
    def expression(self):
        if self._expression is None:
            self._expression = graph_text(self.evaluation)
        return self._expression

    @expression.setter
    def expression(self, value):
        self._expression = value

    @property
    def category(self):
        if self.is_atomic:
            return 'atomic_semantic'
        return 'composite_semantic'

    def tree(self):
        return '{' + self.expression + '}'


class SemanticList(SemanticObject):

    def __init__(self, alternatives, representative, multiplicities=None):
        self.alternatives = tuple(alternatives)
        self.multiplicities = tuple([1] * len(self.alternatives) if multiplicities is None else multiplicities)
        if len(self.multiplicities) != len(self.alternatives) or any((not isinstance(count, int) or count < 1 for count in self.multiplicities)):
            raise ValueError('each semantic alternative needs a positive integer multiplicity')
        if not self.alternatives or sum(self.multiplicities) < 2:
            raise ValueError('semantic list needs at least two alternatives')
        if any((not isinstance(item, SemanticObject) or isinstance(item, SemanticList) for item in self.alternatives)):
            raise ValueError('semantic alternatives must be resolved objects')
        if not any((representative is item for item in self.alternatives)):
            raise ValueError('the representative must be one of the alternatives')
        if any((item.values.shape != representative.values.shape for item in self.alternatives)):
            raise ValueError('all semantic alternatives must use the same dataset')
        super().__init__(None, representative.values, is_atomic=False)
        self.representative = representative
        self.evaluation = None
        self.structure.lazy_label = ref(self)

    @property
    def expression(self):
        if self._expression is None:
            labels = [item.expression if count == 1 else f'{item.expression} (x{count})' for item, count in zip(self.alternatives, self.multiplicities)]
            self._expression = '[' + ' | '.join(labels) + ']'
        return self._expression


def selection_source(node):
    children = getattr(node, 'used_semantic_children', ())
    if isinstance(node, SemanticObject) and (not node.used_operator) and (len(children) == 1) and (children[0].semantic_identity is node.semantic_identity):
        return children[0]
    return None


def underlying_construction(node):
    source = selection_source(node)
    while source is not None:
        node = source
        source = selection_source(node)
    return node


def semantic_choices(current):
    if isinstance(current, SemanticList):
        return tuple(zip(current.alternatives, current.multiplicities))
    return ((current, 1),)


class OperatorObject:
    kind = 'operator'

    def __init__(self, operators, is_atomic=None):
        self.operators = operators if isinstance(operators, OperatorSequence) else tuple(operators)
        if not self.operators:
            raise ValueError('an operator object needs at least one operator')
        self.arity = self.operators.arity if isinstance(self.operators, OperatorSequence) else operator_arity(self.operators[0])
        if not isinstance(self.operators, OperatorSequence):
            for symbol in self.operators:
                if operator_arity(symbol) != self.arity:
                    raise ValueError('an operator object must contain operators of the same arity')
        if is_atomic is None:
            is_atomic = operator_count(self.operators) == 1
        if is_atomic and operator_count(self.operators) != 1:
            raise ValueError('an atomic operator contains exactly one operator')
        self.is_atomic = is_atomic
        self.structure = MergeTree(self.expression if operator_count(self.operators) <= 4096 else '')
        if operator_count(self.operators) > 4096:
            self.structure.lazy_label = ref(self)
        self.children = ()
        self.internal_accessible = True
        self.accessible_count = 1

    @property
    def category(self):
        if self.is_atomic:
            return 'atomic_operator'
        return 'composite_operator'

    @property
    def expression(self):
        return ' | '.join(self.operators)

    def tree(self):
        return '{' + self.expression + '}'


class IncompleteObject:
    kind = 'incomplete'
    category = 'incomplete'
    is_atomic = False

    def __init__(self, semantic, head):
        if not isinstance(semantic, SemanticObject) or not isinstance(head, OperatorObject):
            raise TypeError('an incomplete object needs one semantic object and one operator head')
        if head.arity != 2:
            raise ValueError('an incomplete object needs a binary operator head')
        self.semantic = semantic
        self.head = head
        self.structure = MergeTree('', (semantic.structure, head.structure))
        self.children = (semantic, head)
        self.internal_accessible = True
        self.accessible_count = 1 + semantic.accessible_count + head.accessible_count

    @property
    def expression(self):
        return self.semantic.expression + ' [' + self.head.expression + '] ?'

    def tree(self):
        return '{' + self.semantic.tree() + ' | ' + self.head.tree() + '}'


def object_counts(workspace):
    counts = {}
    for column in MATRIX_COLUMNS:
        counts[column] = 0
    for current in workspace:
        counts[current.category] += 1
    return [counts[column] for column in MATRIX_COLUMNS]


ZERO_SENSITIVE_UFUNCS = {np.multiply: '*', np.divide: '/', np.power: 'power', np.float_power: 'power', np.square: 'square', np.reciprocal: 'reciprocal', np.exp: 'exp', np.exp2: 'exp', np.ldexp: 'ldexp'}


def zero_underflow(operation, operands, result):
    if operation not in ('*', '/', 'power', 'square', 'reciprocal', 'exp', 'ldexp'):
        return False
    zeros = np.asarray(result) == 0
    if not np.any(zeros):
        return np.zeros_like(zeros, dtype=bool)
    first = np.asarray(operands[0])
    nonzero = np.isfinite(first)
    if operation != 'exp':
        nonzero = nonzero & (first != 0)
    if len(operands) == 2:
        second = np.asarray(operands[1])
        nonzero = nonzero & np.isfinite(second)
        if operation in ('*', '/'):
            nonzero = nonzero & (second != 0)
    return zeros & nonzero


def target_zero_underflow(ufunc, method, operands, result, options):
    if method == '__call__':
        failed = zero_underflow(ZERO_SENSITIVE_UFUNCS.get(ufunc), operands, result)
        return failed & options.get('where', True)
    if ufunc is np.multiply and method in ('reduce', 'accumulate'):
        values = np.asarray(operands[0])
        nonzero = np.isfinite(values) & (values != 0)
        axis = options.get('axis', 0)
        if method == 'reduce':
            nonzero = np.all(nonzero | ~np.asarray(options.get('where', True)), axis=axis, keepdims=options.get('keepdims', False))
            initial = options.get('initial', 1)
            if initial is not None:
                nonzero = nonzero & np.isfinite(initial) & (initial != 0)
        else:
            nonzero = np.logical_and.accumulate(nonzero, axis=axis)
        return (np.asarray(result) == 0) & nonzero
    return False
