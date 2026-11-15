import hashlib
import sys

MAX_CREATED_OPERATOR_COUNT = 0


def reset_operator_statistics():
    global MAX_CREATED_OPERATOR_COUNT
    MAX_CREATED_OPERATOR_COUNT = 0


def operator_statistics():
    return {'max_operator_count': str(MAX_CREATED_OPERATOR_COUNT)}


class OperatorSequence:
    __slots__ = ('left', 'right', 'symbols', 'count', 'arity', 'depth', 'fingerprint')
    _MODULUS = 1 << 128
    _BASE = 0x100000001B3

    def __init__(self, left=None, right=None, symbols=None, arity=None):
        if symbols is not None:
            self.symbols = tuple(symbols)
            self.left = None
            self.right = None
            self.count = len(self.symbols)
            self.arity = arity
            self.depth = 0
            digest = 0
            for symbol in self.symbols:
                code = int.from_bytes(hashlib.blake2b(symbol.encode(), digest_size=16).digest(), 'big')
                digest = (digest * self._BASE + code + 1) % self._MODULUS
            self.fingerprint = digest
        else:
            if left.arity != right.arity:
                raise ValueError('operators must have the same arity')
            self.symbols = None
            self.left = left
            self.right = right
            self.count = left.count + right.count
            self.arity = left.arity
            self.depth = 1 + max(left.depth, right.depth)
            self.fingerprint = (left.fingerprint * pow(self._BASE, right.count, self._MODULUS) + right.fingerprint) % self._MODULUS

    def __len__(self):
        if self.count > sys.maxsize:
            raise OverflowError('operator sequence exceeds the platform length limit')
        return self.count

    def __bool__(self):
        return self.count != 0

    def __iter__(self):
        pending = [self]
        while pending:
            item = pending.pop()
            if item.symbols is not None:
                yield from item.symbols
            else:
                pending.extend((item.right, item.left))

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError('operator sequence indices must be integers')
        if index < 0:
            index += self.count
        if index < 0 or index >= self.count:
            raise IndexError(index)
        item = self
        while item.symbols is None:
            if index < item.left.count:
                item = item.left
            else:
                index -= item.left.count
                item = item.right
        return item.symbols[index]

    def __eq__(self, other):
        if isinstance(other, OperatorSequence):
            if self is other:
                return True
            if self.count != other.count or self.fingerprint != other.fingerprint:
                return False
            return all((a == b for a, b in zip(self, other)))
        if isinstance(other, tuple):
            if self.count > 1024 or self.count != len(other):
                return False
            return all((a == b for a, b in zip(self, other)))
        return NotImplemented

    def __hash__(self):
        if self.count <= 1024:
            return hash(tuple(self))
        return hash((self.count, self.fingerprint))

    def __add__(self, other):
        return operator_concat(self, other)

    def __radd__(self, other):
        return operator_concat(other, self)

    def __repr__(self):
        return f'OperatorSequence(count={self.count}, arity={self.arity}, depth={self.depth})'


def operator_count(operators):
    return operators.count if isinstance(operators, OperatorSequence) else len(operators)


def operator_concat(left, right):
    global MAX_CREATED_OPERATOR_COUNT
    left_count = operator_count(left)
    right_count = operator_count(right)
    count = left_count + right_count
    if count > MAX_CREATED_OPERATOR_COUNT:
        MAX_CREATED_OPERATOR_COUNT = count
    if count <= 256:
        return tuple(left) + tuple(right)
    if not isinstance(left, OperatorSequence):
        from mgp.engine.objects import operator_arity
        left = OperatorSequence(symbols=left, arity=operator_arity(left[0]))
    if not isinstance(right, OperatorSequence):
        from mgp.engine.objects import operator_arity
        right = OperatorSequence(symbols=right, arity=operator_arity(right[0]))
    return OperatorSequence(left, right)


def operator_digest(operators):
    if operator_count(operators) <= 4096:
        return ','.join(operators)
    if not isinstance(operators, OperatorSequence):
        return ','.join(operators)
    return 'rope:' + str(operators.count) + ':' + format(operators.fingerprint, '032x')
