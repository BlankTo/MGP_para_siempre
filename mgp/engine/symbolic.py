import hashlib
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache, cmp_to_key
from itertools import zip_longest

import sympy as sp
from sympy.printing.repr import ReprPrinter
from sympy.printing.str import StrPrinter

from mgp.engine.objects import SemanticList


def symbolic_text(expression, representation=False):
    printer = ReprPrinter() if representation else StrPrinter()
    original = printer._print
    cache = {}

    def cached_print(node, **kwargs):
        if not isinstance(node, sp.Basic):
            return original(node, **kwargs)
        key = (node, tuple(kwargs.items()))
        if key not in cache:
            cache[key] = original(node, **kwargs)
        return cache[key]
    printer._print = cached_print
    try:
        return printer.doprint(expression)
    finally:
        del printer._print


MAX_SYMBOLIC_TEXT = 4096
COMPACT_SYMBOLIC_PREFIX = 'symdag1:'
DOMAIN_FACTOR_DEGREE_LIMIT = 12
DOMAIN_FACTOR_TERMS_LIMIT = 30
_FP_SMALL_CACHE = OrderedDict()
_FP_CACHE_LIMIT = 1024
_FP_CACHE_MAX_COST = 256
_FP_NODE_NAMES = {}
_FP_CACHE_ENABLED = os.environ.get('MGP_FINGERPRINT_CACHE', '1') != '0'


def symbolic_fingerprint(expression):
    pending = [(expression, False)]
    results = {}
    small_cache = _FP_SMALL_CACHE
    node_names = _FP_NODE_NAMES
    while pending:
        node, ready = pending.pop()
        marker = id(node)
        if marker in results:
            continue
        cached = small_cache.get(marker) if _FP_CACHE_ENABLED else None
        if cached is not None and cached[0] is node:
            results[marker] = cached[1]
            continue
        children = node.args
        if children and not ready:
            pending.append((node, True))
            pending.extend((child, False) for child in reversed(children) if id(child) not in results)
            continue
        if children:
            func = node.func
            name = node_names.get(func)
            if name is None:
                name = (func.__module__ + '.' + func.__qualname__).encode('utf-8')
                node_names[func] = name
            pieces = [results[id(child)] for child in children]
            digest = hashlib.sha256(b'N' + len(name).to_bytes(4, 'big') + name + len(pieces).to_bytes(4, 'big') + b''.join(piece[0] for piece in pieces)).digest()
            cost = min(MAX_SYMBOLIC_TEXT + 1, len(name) + 8 + 2 * len(pieces) + sum(piece[1] for piece in pieces))
        else:
            atom = symbolic_text(node, True).encode('utf-8')
            digest = hashlib.sha256(b'A' + len(atom).to_bytes(8, 'big') + atom).digest()
            cost = min(MAX_SYMBOLIC_TEXT + 1, len(atom))
        value = (digest, cost)
        results[marker] = value
        if _FP_CACHE_ENABLED and cost <= _FP_CACHE_MAX_COST:
            small_cache[marker] = (node, value)
            if len(small_cache) > _FP_CACHE_LIMIT:
                small_cache.popitem(last=False)
    return results[id(expression)]


@dataclass(frozen=True, eq=False)
class CompactGuard:
    kind: str
    expression: object
    fingerprint: bytes

    def __hash__(self):
        return hash((self.kind, self.fingerprint))

    def __eq__(self, other):
        return isinstance(other, CompactGuard) and self.kind == other.kind and self.expression == other.expression


def guard_token(guard):
    if isinstance(guard, str):
        return guard
    prefix = 'nonzero:' if guard.kind == 'nonzero' else ''
    return prefix + 'symguard1:' + guard.fingerprint.hex()


def sorted_guard_tokens(guards):
    return sorted(guard_token(guard) for guard in guards)


def domain_guard(expression, kind='condition'):
    fingerprint, cost = symbolic_fingerprint(expression)
    if cost < MAX_SYMBOLIC_TEXT:
        value = symbolic_text(expression, True)
        prefix = 'nonzero:' if kind == 'nonzero' else ''
        if len(prefix) + len(value) <= MAX_SYMBOLIC_TEXT:
            return prefix + value
    return CompactGuard(kind, expression, fingerprint)


def symbolic_identity(expression, restrictions):
    fingerprint, cost = symbolic_fingerprint(expression)
    guard_strings = sorted_guard_tokens(restrictions)
    if cost < MAX_SYMBOLIC_TEXT and all(isinstance(guard, str) for guard in restrictions) and sum(map(len, guard_strings)) + cost < MAX_SYMBOLIC_TEXT:
        representation = symbolic_text(expression, True)
        key = repr((representation, tuple(guard_strings)))
        if len(key) <= MAX_SYMBOLIC_TEXT:
            return (key, symbolic_text(expression))
    guards_digest = hashlib.sha256()
    for guard in guard_strings:
        encoded = guard.encode('utf-8')
        guards_digest.update(len(encoded).to_bytes(8, 'big'))
        guards_digest.update(encoded)
    key = COMPACT_SYMBOLIC_PREFIX + fingerprint.hex() + ':' + guards_digest.hexdigest() + ':' + str(len(guard_strings)) + ':' + str(cost)
    return (key, None)


def compact_symbolic_key(key):
    return isinstance(key, str) and key.startswith(COMPACT_SYMBOLIC_PREFIX)


def _srepr_fragments(expression):
    pending = [expression]
    while pending:
        node = pending.pop()
        if isinstance(node, str):
            yield node
        elif not node.args:
            yield symbolic_text(node, True)
        else:
            if node.is_Add:
                arguments = node.as_ordered_terms()
            elif node.is_Mul:
                arguments = node.as_ordered_factors()
            else:
                arguments = node.args
            yield node.func.__name__ + '('
            pending.append(')')
            for index in range(len(arguments) - 1, -1, -1):
                pending.append(arguments[index])
                if index:
                    pending.append(', ')


def _contains_quotes(expression):
    pending = [expression]
    single = False
    double = False
    while pending:
        node = pending.pop()
        if node.args:
            pending.extend(node.args)
        else:
            text = symbolic_text(node, True)
            single |= "'" in text
            double |= '"' in text
        if single and double:
            break
    return ('"' if single and not double else "'")


def _quoted_characters(parts, quote):
    yield quote
    for fragment in parts:
        for letter in fragment:
            if letter == '\\' or letter == quote:
                yield '\\'
                yield letter
            elif letter == '\n':
                yield '\\'
                yield 'n'
            elif letter == '\r':
                yield '\\'
                yield 'r'
            elif letter == '\t':
                yield '\\'
                yield 't'
            else:
                yield letter
    yield quote


def _guard_characters(guard):
    if isinstance(guard, str):
        yield guard
        return
    if guard.kind == 'nonzero':
        yield 'nonzero:'
    yield from _srepr_fragments(guard.expression)


def _guard_compare(first, second):
    left_chars = (letter for fragment in _guard_characters(first) for letter in fragment)
    right_chars = (letter for fragment in _guard_characters(second) for letter in fragment)
    for left, right in zip_longest(left_chars, right_chars):
        if left != right:
            if left is None:
                return -1
            if right is None:
                return 1
            return (left > right) - (left < right)
    return 0


def _legacy_guard_tuple(restrictions):
    ordered = sorted(restrictions, key=cmp_to_key(_guard_compare))
    yield '('
    for index, guard in enumerate(ordered):
        if index:
            yield ', '
        if isinstance(guard, str):
            yield repr(guard)
        else:
            yield from _quoted_characters(_guard_characters(guard), _contains_quotes(guard.expression))
    if len(ordered) == 1:
        yield ','
    yield ')'


def _legacy_key_characters(expression, restrictions):
    yield '('
    yield from _quoted_characters(_srepr_fragments(expression), _contains_quotes(expression))
    yield ', '
    yield from _legacy_guard_tuple(restrictions)
    yield ')'


def symbolic_compare(first, second):
    a = symbolic_data(first)
    b = symbolic_data(second)
    if a[0] == b[0]:
        return 0
    if not compact_symbolic_key(a[0]) and not compact_symbolic_key(b[0]):
        return (a[0] > b[0]) - (a[0] < b[0])
    if a[2] == b[2] and all(isinstance(guard, str) for guard in a[3] | b[3]):
        left, right = repr(tuple(sorted(a[3]))), repr(tuple(sorted(b[3])))
        return (left > right) - (left < right)
    for left, right in zip_longest(_legacy_key_characters(a[2], a[3]), _legacy_key_characters(b[2], b[3])):
        if left != right:
            if left is None:
                return -1
            if right is None:
                return 1
            return (left > right) - (left < right)
    return 0


def archive_rank_compare(first, second):
    left, right = -first[1]['score'], -second[1]['score']
    if left < right:
        return -1
    if left > right:
        return 1
    return symbolic_compare(first[0], second[0])


def nonzero_guards(expression, degree):
    if expression.is_nonzero is True:
        return set()
    if expression.is_number:
        return set() if expression.is_zero is False else {'undefined'}
    if expression.func == sp.Abs or (expression.is_Pow and expression.exp.is_nonzero):
        return nonzero_guards(expression.args[0], degree)
    if expression.is_Mul:
        return set().union(*(nonzero_guards(arg, degree) for arg in expression.args))
    if rational_budget(expression):
        numerator = sp.fraction(sp.cancel(expression))[0]
        if numerator == 0:
            return {'undefined'}
        if numerator.is_polynomial():
            try:
                poly = sp.Poly(numerator)
                skip = poly.total_degree() > DOMAIN_FACTOR_DEGREE_LIMIT or len(poly.terms()) > DOMAIN_FACTOR_TERMS_LIMIT
            except (sp.PolynomialError, TypeError, ValueError):
                skip = False
            if skip:
                EFFORT['unfactored_domain_guards'] += 1
                return {domain_guard(numerator, 'nonzero')}
            coefficient, factors = sp.factor_list(numerator)
            return {domain_guard(factor, 'nonzero') for factor, power in factors if factor.is_nonzero is not True}
    return {domain_guard(expression, 'nonzero')}


@lru_cache(maxsize=4096)
def rational_budget(expression):
    pending = [(expression, False)]
    costs = {}
    while pending:
        node, ready = pending.pop()
        if node in costs:
            continue
        if len(costs) + len(pending) > 600:
            return False
        if node.is_Atom:
            costs[node] = 1
        elif not ready:
            pending.append((node, True))
            pending.extend(((arg, False) for arg in node.args))
        elif node.is_Add:
            costs[node] = min(4097, sum((costs[arg] for arg in node.args)))
        elif node.is_Mul:
            cost = 1
            for arg in node.args:
                cost = min(4097, cost * costs[arg])
            costs[node] = cost
        elif node.is_Pow and node.exp.is_Integer:
            power = abs(int(node.exp))
            if power > 512:
                return False
            costs[node] = min(4097, costs[node.base] ** min(power, 13))
        else:
            return False
        if node in costs and costs[node] > 4096:
            return False
    return bool(expression.is_rational_function())


EFFORT = {'identities': 0, 'cancel': 0, 'simplify': 0, 'unfactored_domain_guards': 0}


@lru_cache(maxsize=4096)
def simplify_expression(expression, degree):
    if rational_budget(expression):
        EFFORT['cancel'] += 1
        return sp.cancel(expression)
    if degree is None:
        for count, node in enumerate(sp.preorder_traversal(expression), 1):
            if count > 80:
                return expression
        EFFORT['simplify'] += 1
        return sp.simplify(expression)
    return expression


@lru_cache(maxsize=512)
def parsed_expression(text):
    names = {name: sp.Symbol(name, real=True) for name in re.findall('\\b[A-Za-z_]\\w*\\b', text) if name not in ('sin', 'cos', 'log', 'sqrt', 'Abs')}
    return sp.sympify(text, locals=names)


def expression_key(text):
    expression = parsed_expression(text)
    return symbolic_identity(simplify_expression(expression, None), frozenset())[0]


def _substitute_zero_iterative(expression):
    completed = {}
    pending = [(expression, False)]
    while pending:
        node, ready = pending.pop()
        token = id(node)
        if token in completed:
            continue
        if getattr(node, 'is_Symbol', False):
            completed[token] = sp.Integer(0)
            continue
        args = getattr(node, 'args', ())
        if not args:
            completed[token] = node
            continue
        if not ready:
            pending.append((node, True))
            pending.extend((child, False) for child in reversed(args) if id(child) not in completed)
            continue
        children = [completed[id(child)] for child in args]
        if node.is_Add:
            value = sp.Add(*children)
        elif node.is_Mul:
            value = sp.Mul(*children)
        elif node.is_Pow:
            value = sp.Pow(*children)
        elif node.func in (sp.Abs, sp.sin, sp.cos, sp.log, sp.sqrt, sp.Gt, sp.Ge, sp.Lt, sp.Le):
            value = node.func(*children)
        else:
            raise ValueError('origin substitution for this expression requires a general symbolic evaluator')
        completed[token] = value
    return completed[id(expression)]


def _origin_value(expression):
    seen = set()
    pending = [expression]
    while pending and len(seen) <= 128:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        pending.extend(getattr(node, 'args', ()))
    if pending:
        try:
            return _substitute_zero_iterative(expression)
        except (RecursionError, ValueError, TypeError):
            return None
    try:
        return expression.subs({symbol: 0 for symbol in expression.free_symbols})
    except RecursionError:
        try:
            return _substitute_zero_iterative(expression)
        except (RecursionError, ValueError, TypeError):
            return None


def recovery_status(current, target_expression):
    data = symbolic_data(current)
    expected = simplify_expression(parsed_expression(target_expression), None)
    if data[3]:
        if 'undefined' in data[3]:
            return False
        for guard in data[3]:
            if isinstance(guard, CompactGuard):
                candidate = guard.expression
                is_nonzero = guard.kind == 'nonzero'
            else:
                candidate = None
                is_nonzero = guard.startswith('nonzero:')
            if is_nonzero:
                denominator = candidate if candidate is not None else sp.sympify(guard[len('nonzero:'):])
                if _origin_value(denominator) == 0:
                    return False
            else:
                condition = candidate if candidate is not None else sp.sympify(guard)
                if _origin_value(condition) is sp.false:
                    return False
        return None
    if data[2] == expected:
        return True
    if rational_budget(data[2]) and rational_budget(expected):
        return bool(sp.cancel(data[2] - expected) == 0)
    return None


def symbolic_data(current):
    if current.symbolic_identity is not None:
        return current.symbolic_identity
    if isinstance(current, SemanticList):
        identities = sorted({symbolic_data(item)[0] for item in current.alternatives})
        key = 'alternatives:' + repr(identities)
        label = '[' + ' | '.join(sorted({symbolic_data(item)[1] or symbolic_data(item)[0] for item in current.alternatives})) + ']'
        current.symbolic_identity = (key, label, None, frozenset(), None)
        return current.symbolic_identity
    pending = [(current, False)]
    while pending:
        node, ready = pending.pop()
        if node.symbolic_identity is not None:
            continue
        children = node.evaluation_children
        if children and (not ready):
            pending.append((node, True))
            pending.extend(((child, False) for child in reversed(children)))
            continue
        guards = set()
        if not children:
            expression = sp.Symbol(node.expression, real=True)
            degree = 1
        else:
            data = [child.symbolic_identity for child in children]
            arguments = [item[2] for item in data]
            for item in data:
                guards.update(item[3])
            symbol = node.evaluation[0]
            degree = None if any((item[4] is None for item in data)) else sum((item[4] for item in data))
            first = arguments[0]
            if 'undefined' in guards:
                expression = sp.nan
            elif symbol == '+':
                expression = first + arguments[1]
            elif symbol == '-':
                expression = first - arguments[1]
            elif symbol == '*':
                expression = first * arguments[1]
            elif symbol == '/':
                guards.update(nonzero_guards(arguments[1], data[1][4]))
                expression = first / arguments[1]
            elif symbol == 'sin':
                expression = sp.sin(first)
                degree = None
            elif symbol == 'cos':
                expression = sp.cos(first)
                degree = None
            elif symbol in ('log', 'sqrt'):
                condition = sp.Gt(first, 0) if symbol == 'log' else sp.Ge(first, 0)
                if condition is sp.false:
                    guards.add('undefined')
                elif condition is not sp.true:
                    guards.add(domain_guard(condition))
                expression = sp.log(first) if symbol == 'log' else sp.sqrt(first)
                degree = None
            else:
                raise ValueError('unknown symbolic operator: ' + symbol)
            if 'undefined' in guards:
                expression, degree = (sp.nan, None)
            else:
                expression = simplify_expression(expression, degree)
        restrictions = frozenset(guards)
        key, label = symbolic_identity(expression, restrictions)
        node.symbolic_identity = (key, label, expression, restrictions, degree)
        EFFORT['identities'] += 1
    return current.symbolic_identity


def symbolic_has_variables(current):
    known = getattr(current, '_has_symbolic_variables', None)
    if known is not None:
        return known
    expression = symbolic_data(current)[2]
    if expression is None:
        return False
    allowed = {sp.Add, sp.Mul, sp.Pow, sp.sin, sp.cos, sp.log}
    pending = [expression]
    seen = set()
    result = False
    while pending:
        item = pending.pop()
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        if item.is_Symbol:
            result = True
            break
        if item.func in allowed:
            pending.extend(item.args)
        elif item.is_Atom:
            if item.free_symbols:
                result = True
                break
        else:
            result = bool(expression.free_symbols)
            break
    current._has_symbolic_variables = result
    return result


def sympy_equivalent(first, second):
    return symbolic_data(first)[0] == symbolic_data(second)[0]
