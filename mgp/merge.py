from copy import copy
import random

from .fitness import object_fitness, readonly_target
from .objects import IncompleteObject, MergeTree, OperatorObject, SemanticObject, apply_operator


class InternalMerge:
    def __init__(self, external_merge):
        self.external_merge = external_merge

    def candidate_count(self, current):
        return current.accessible_count - int(current.internal_accessible)

    def select(self, current, generator):
        count = self.candidate_count(current)

        if count == 0:
            raise ValueError("This object has no accessible internal occurrence.")

        rank = generator.randrange(count)
        path = ()

        while current.children:
            for index in (1, 0):
                child = current.children[index]

                if rank >= child.accessible_count:
                    rank -= child.accessible_count
                    continue

                path += (index,)

                if child.internal_accessible:
                    if rank == 0:
                        return path, child

                    rank -= 1

                current = child
                break

        raise ValueError("Inconsistent internal accessibility count.")

    def candidates(self, current):
        candidates = []
        pending = []

        for index, child in enumerate(current.children):
            pending.append(((index,), child))

        while pending:
            path, child = pending.pop()

            if child.internal_accessible:
                candidates.append((path, child))

            for index, descendant in enumerate(child.children):
                pending.append((path + (index,), descendant))

        return candidates

    def merge(self, current, path=None):
        if path is None:
            path, selected = self.select(current, self.external_merge.random)
        else:
            selected = current

            if not path:
                raise ValueError("Select an internal occurrence, not the root.")

            for index in path:
                if index not in (0, 1) or index >= len(selected.children):
                    raise ValueError("The internal occurrence does not exist.")

                selected = selected.children[index]

            if not selected.internal_accessible:
                raise ValueError("The internal occurrence is not accessible.")

        parent = copy(current)
        parent.accessible_count -= 1
        original = current
        branch = parent

        for index in path:
            children = list(original.children)
            child = copy(children[index])
            child.accessible_count -= 1
            children[index] = child
            branch.children = tuple(children)
            original = original.children[index]
            branch = child

        branch.internal_accessible = False
        result = self.external_merge.merge(selected, parent)

        return result


class ExternalMerge:
    def __init__(self, target, fitness_probability=1.0, seed=0):
        if not 0 <= fitness_probability <= 1:
            raise ValueError("fitness_probability must be between 0 and 1.")

        self.target = readonly_target(target)
        self.fitness_probability = fitness_probability
        self.random = random.Random(seed)

    def choose(self, candidates):
        if len(candidates) == 1:
            return candidates[0]

        if self.random.random() >= self.fitness_probability:
            return self.random.choice(candidates)

        best = None
        best_fitness = float("inf")

        for candidate in candidates:
            fitness = object_fitness(candidate, self.target)

            if best is None or fitness < best_fitness:
                best = candidate
                best_fitness = fitness

        return best

    def merge(self, left, right):
        result = self.evaluate_merge(left, right)
        result.structure = MergeTree(result.expression, (left.structure, right.structure))
        result.children = (left, right)
        result.internal_accessible = True
        result.accessible_count = 1 + left.accessible_count + right.accessible_count

        return result

    def evaluate_merge(self, left, right):
        for current in (left, right):
            if not isinstance(current, (SemanticObject, OperatorObject, IncompleteObject)):
                raise TypeError("Unsupported syntactic object.")

            semantic = current

            if isinstance(current, IncompleteObject):
                semantic = current.semantic

            if isinstance(semantic, SemanticObject) and semantic.values.shape != self.target.shape:
                raise ValueError("All objects must use the target dataset.")

        if isinstance(left, SemanticObject) and isinstance(right, SemanticObject):
            selected = self.choose([left, right])
            result = copy(selected)
            result.is_atomic = False
            return result

        if isinstance(left, OperatorObject) and isinstance(right, OperatorObject):
            if left.is_atomic != right.is_atomic:
                if left.is_atomic:
                    return OperatorObject(left.operators)

                return OperatorObject(right.operators)

            if left.arity != right.arity:
                if left.arity < right.arity:
                    return OperatorObject(left.operators, is_atomic=left.is_atomic)

                return OperatorObject(right.operators, is_atomic=right.is_atomic)

            return OperatorObject(left.operators + right.operators, is_atomic=False)

        if isinstance(left, OperatorObject) or isinstance(right, OperatorObject):
            if isinstance(left, OperatorObject):
                head, other = left, right
            else:
                head, other = right, left

            if isinstance(other, IncompleteObject):
                other = other.semantic

            if isinstance(other, SemanticObject):
                if head.arity == 1:
                    return self.apply_unary(head, other)

                return IncompleteObject(other, head)

            raise TypeError("Unsupported syntactic object.")

        operators = []

        if isinstance(left, IncompleteObject):
            operators.extend(left.head.operators)
            left = left.semantic

        if isinstance(right, IncompleteObject):
            operators.extend(right.head.operators)
            right = right.semantic

        if not operators or not isinstance(left, SemanticObject) or not isinstance(right, SemanticObject):
            raise TypeError("Unsupported syntactic objects.")

        candidates = []
        evaluated = {}

        for symbol in operators:
            if symbol not in evaluated:
                evaluated[symbol] = self.evaluate_operator(symbol, left, right)

            candidates.append(evaluated[symbol])

        return self.choose(candidates)

    def apply_unary(self, head, semantic):
        candidates = []
        evaluated = {}

        for symbol in head.operators:
            if symbol not in evaluated:
                evaluated[symbol] = self.evaluate_unary(symbol, semantic)

            candidates.append(evaluated[symbol])

        return self.choose(candidates)

    def evaluate_unary(self, symbol, semantic):
        argument = semantic

        if semantic.unary_argument is not None:
            argument = semantic.unary_argument

        values = apply_operator(symbol, argument.values)
        result = SemanticObject(f"{symbol}({argument.expression})", values, is_atomic=False)
        result.evaluation = (symbol, argument.evaluation)
        result.unary_argument = argument

        return result

    def evaluate_operator(self, symbol, left, right):
        values = apply_operator(symbol, left.values, right.values)
        expression = f"({left.expression} {symbol} {right.expression})"
        result = SemanticObject(expression, values, is_atomic=False)
        result.evaluation = (symbol, left.evaluation, right.evaluation)

        return result
