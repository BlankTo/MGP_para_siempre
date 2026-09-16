from copy import copy
import random

import numpy as np

from .fitness import object_fitness, readonly_target
from .objects import IncompleteObject, MergeTree, OperatorObject, SemanticObject


class InternalMerge:
    def merge(self, current):
        raise NotImplementedError("Internal MERGE is not implemented.")


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

        with np.errstate(divide="ignore", invalid="ignore"):
            if symbol == "sin":
                values = np.sin(argument.values)
            elif symbol == "cos":
                values = np.cos(argument.values)
            elif symbol == "log":
                values = np.log(argument.values)
            elif symbol == "sqrt":
                values = np.sqrt(argument.values)
            else:
                raise ValueError("Unknown unary operator: " + symbol)

        result = SemanticObject(f"{symbol}({argument.expression})", values, is_atomic=False)
        result.unary_argument = argument

        return result

    def evaluate_operator(self, symbol, left, right):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if symbol == "+":
                values = left.values + right.values
            elif symbol == "-":
                values = left.values - right.values
            elif symbol == "*":
                values = left.values * right.values
            elif symbol == "/":
                values = left.values / right.values
            else:
                raise ValueError("Unknown operator: " + symbol)

        expression = f"({left.expression} {symbol} {right.expression})"

        return SemanticObject(expression, values, is_atomic=False)
