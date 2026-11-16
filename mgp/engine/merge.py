import random
from collections import Counter
from copy import copy
from weakref import WeakKeyDictionary

from mgp.engine.fitness import object_fitness, readonly_target
from mgp.engine.merge_strategies import AlternativeLimit, AlternativeReplay, constructive_pair, merge_kind, merge_pair
from mgp.engine.objects import IncompleteObject, MergeTree, OperatorObject, SemanticList, SemanticObject, apply_operator, semantic_choices
from mgp.engine.operator_sequence import operator_concat


def _copy_mgp_node(node):
    cls = type(node)
    if cls in (SemanticObject, SemanticList, OperatorObject, IncompleteObject):
        duplicate = object.__new__(cls)
        duplicate.__dict__ = node.__dict__.copy()
        return duplicate
    return copy(node)


def _constructive_child_predicate(kind, allowed):
    operator, incomplete, semantic = (merge_pair(kind, other) in allowed for other in ('operator', 'incomplete', 'semantic'))

    def legal(child):
        if isinstance(child, OperatorObject):
            return operator
        if isinstance(child, IncompleteObject):
            return incomplete
        return semantic
    return legal


class InternalMerge:

    def __init__(self, external_merge):
        self.external_merge = external_merge
        self.constructive_cache = {}

    def candidate_count(self, current):
        if self.external_merge.strategy == 'constructive':
            counts = self.constructive_counts(current)
            return sum((counts[child] for child in current.children))
        return current.accessible_count - int(current.internal_accessible)

    def constructive_counts(self, current):
        kind = merge_kind(current)
        entry = self.constructive_cache.get(kind)
        if entry is None:
            entry = self.constructive_cache[kind] = (WeakKeyDictionary(), _constructive_child_predicate(kind, self.external_merge.constructive_merges))
        counts, allowed = entry
        pending = [(current, False)]
        while pending:
            node, ready = pending.pop()
            if node in counts:
                continue
            if ready:
                counts[node] = int(node.internal_accessible and allowed(node)) + sum((counts[child] for child in node.children))
            else:
                pending.append((node, True))
                pending.extend(((child, False) for child in node.children if child not in counts))
        return counts

    def select(self, current, generator):
        if self.external_merge.strategy == 'constructive':
            parent = current
            counts = self.constructive_counts(parent)
            count = sum((counts[child] for child in parent.children))
            if not count:
                raise ValueError('this object has no legal constructive internal occurrence')
            rank, path = generator.randrange(count), []
            while current.children:
                for index in (1, 0):
                    child = current.children[index]
                    if rank >= counts[child]:
                        rank -= counts[child]
                        continue
                    path.append(index)
                    if child.internal_accessible and constructive_pair(child, parent, self.external_merge.constructive_merges):
                        if rank == 0:
                            return (tuple(path), child)
                        rank -= 1
                    current = child
                    break
            raise ValueError('inconsistent constructive occurrence count')
        count = self.candidate_count(current)
        if count == 0:
            raise ValueError('this object has no accessible internal occurrence')
        rank = generator.randrange(count)
        path = []
        while current.children:
            for index in (1, 0):
                child = current.children[index]
                if rank >= child.accessible_count:
                    rank -= child.accessible_count
                    continue
                path.append(index)
                if child.internal_accessible:
                    if rank == 0:
                        return (tuple(path), child)
                    rank -= 1
                current = child
                break
        raise ValueError('inconsistent internal accessibility count')

    def merge(self, current, path=None):
        if path is None:
            path, selected = self.select(current, self.external_merge.random)
        else:
            selected = current
            if not path:
                raise ValueError('internal occurrence required, root selected')
            for index in path:
                if index not in (0, 1) or index >= len(selected.children):
                    raise ValueError('the internal occurrence does not exist')
                selected = selected.children[index]
            if not selected.internal_accessible:
                raise ValueError('the internal occurrence is not accessible')
        parent = _copy_mgp_node(current)
        parent.accessible_count -= 1
        original = current
        branch = parent
        for index in path:
            children = list(original.children)
            child = _copy_mgp_node(children[index])
            child.accessible_count -= 1
            children[index] = child
            branch.children = tuple(children)
            original = original.children[index]
            branch = child
        branch.internal_accessible = False
        result = self.external_merge.merge(selected, parent)
        return result


class ExternalMerge:

    def __init__(self, target, fitness_probability=1.0, seed=0, semantic_list_enabled=False):
        if not 0 <= fitness_probability <= 1:
            raise ValueError('fitness_probability must be between 0 and 1')
        self.target = readonly_target(target)
        self.fitness_probability = fitness_probability
        self.random = random.Random(seed)
        self.semantic_list_enabled = semantic_list_enabled
        self.operator_evaluations = 0
        self.ic_candidates = 0
        self.ic_decisions = 0
        self.ms_decisions = 0
        self.fitness_evaluations = 0
        self.candidate_observer = None
        self.choice_observer = None
        self.strategy = 'standard'
        self.constructive_merges = frozenset()
        self.complete_into_hole = False
        self.alternative_evaluation_limit = 512
        self.alternative_variant_limit = 64
        self.alternative_pair_limit = 512
        self.replay_pairs = 0
        self.replay_duplicates = 0
        self.replay_expanded_merges = 0
        self.replay_variants_cache = {}
        self.replay_cache_limits = {}
        self.replay_limit_scopes = []
        self.replaying = False
        self.replay_operator_evaluations = 0
        self.replay_truncated_merges = 0
        self.replay_limit_hits = Counter()
        self.current_replay_limits = set()

    def choose(self, candidates, weights=None, force_best=False):
        selected = self.choose_original(candidates, weights, force_best)
        if self.choice_observer is not None:
            self.choice_observer(candidates, weights, selected)
        if self.strategy == 'revisit_ic' and len(candidates) > 1:
            result = _copy_mgp_node(selected)
            result.ic_alternatives = tuple(candidates)
            return result
        return selected

    def choose_original(self, candidates, weights=None, force_best=False):
        self.ic_candidates += len(candidates)
        if len(candidates) == 1 and (weights is None or weights[0] == 1):
            return candidates[0]
        self.ic_decisions += 1
        if not force_best and self.random.random() >= self.fitness_probability:
            if weights is not None:
                rank = self.random.randrange(sum(weights))
                for candidate, weight in zip(candidates, weights):
                    if rank < weight:
                        return candidate
                    rank -= weight
            return self.random.choice(candidates)
        best = None
        best_fitness = float('inf')
        self.fitness_evaluations += len(candidates)
        for candidate in candidates:
            fitness = object_fitness(candidate, self.target)
            if best is None or fitness < best_fitness:
                best = candidate
                best_fitness = fitness
        return best

    def merge(self, left, right):
        self.current_replay_limits = set()
        if self.strategy == 'revisit_ic':
            result = AlternativeReplay(self, self.alternative_variant_limit, self.alternative_pair_limit).run(left, right)
            if self.current_replay_limits:
                self.replay_truncated_merges += 1
        else:
            result = self.evaluate_merge(left, right)
        result.structure = MergeTree('', (left.structure, right.structure))
        result.children = (left, right)
        result._used_counts = None
        if isinstance(result, SemanticObject):
            result.occurrence_identity = object()
        result.internal_accessible = True
        result.accessible_count = 1 + left.accessible_count + right.accessible_count
        return result

    def resolve_list(self, current):
        selected = self.choose(current.alternatives, current.multiplicities)
        result = _copy_mgp_node(selected)
        result.is_atomic = False
        result.children = current.children
        result.structure = MergeTree('', current.structure.children)
        result.internal_accessible = current.internal_accessible
        result.accessible_count = current.accessible_count
        result.occurrence_identity = current.occurrence_identity
        result.used_semantic_children = (selected,)
        result.used_operator = False
        result._used_counts = None
        return result

    def evaluate_merge(self, left, right):
        for current in (left, right):
            if not isinstance(current, (SemanticObject, OperatorObject, IncompleteObject)):
                raise TypeError('unsupported syntactic object')
            semantic = current
            if isinstance(current, IncompleteObject):
                semantic = current.semantic
            if isinstance(semantic, SemanticObject) and semantic.values.shape != self.target.shape:
                raise ValueError('all objects must use the target dataset')
        if isinstance(left, SemanticObject) and isinstance(right, SemanticObject):
            if self.semantic_list_enabled:
                if isinstance(left, SemanticList) != isinstance(right, SemanticList):
                    self.ms_decisions += 1
                    selected = right if isinstance(left, SemanticList) else left
                    result = _copy_mgp_node(selected)
                    result.is_atomic = False
                    result.used_semantic_children = (selected,)
                    result.used_operator = False
                    return result
                choices = {}
                for item, count in semantic_choices(left) + semantic_choices(right):
                    token = item.occurrence_identity
                    if token not in choices:
                        choices[token] = [item, 0]
                    choices[token][1] += count
                alternatives = tuple((item for item, count in choices.values()))
                multiplicities = tuple((count for item, count in choices.values()))
                representative = min(alternatives, key=lambda item: object_fitness(item, self.target))
                return SemanticList(alternatives, representative, multiplicities)
            selected = self.choose([left, right])
            result = _copy_mgp_node(selected)
            result.is_atomic = False
            result.used_semantic_children = (selected,)
            result.used_operator = False
            return result
        if isinstance(left, OperatorObject) and isinstance(right, OperatorObject):
            if left.is_atomic != right.is_atomic:
                self.ms_decisions += 1
                if left.is_atomic:
                    return OperatorObject(left.operators)
                return OperatorObject(right.operators)
            if left.arity != right.arity:
                self.ms_decisions += 1
                if left.arity < right.arity:
                    return OperatorObject(left.operators, is_atomic=left.is_atomic)
                return OperatorObject(right.operators, is_atomic=right.is_atomic)
            return OperatorObject(operator_concat(left.operators, right.operators), is_atomic=False)
        if isinstance(left, OperatorObject) or isinstance(right, OperatorObject):
            if isinstance(left, OperatorObject):
                head, other = (left, right)
            else:
                head, other = (right, left)
            if isinstance(other, IncompleteObject):
                self.ms_decisions += 1
                other = other.semantic
            if isinstance(other, SemanticObject):
                if head.arity == 1:
                    return self.apply_unary(head, other)
                return IncompleteObject(other, head)
            raise TypeError('unsupported syntactic object')
        first_semantic = left.semantic if isinstance(left, IncompleteObject) else left
        second_semantic = right.semantic if isinstance(right, IncompleteObject) else right
        heads = []
        if isinstance(left, IncompleteObject):
            heads.append((left.head.operators, first_semantic, second_semantic))
        if isinstance(right, IncompleteObject):
            heads.append((right.head.operators, second_semantic, first_semantic) if self.complete_into_hole else (right.head.operators, first_semantic, second_semantic))
        if not heads or not isinstance(first_semantic, SemanticObject) or (not isinstance(second_semantic, SemanticObject)):
            raise TypeError('unsupported syntactic objects')
        candidates = []
        evaluated = {}
        weights = []
        try:
            for operators, inner, hole in heads:
                for symbol in operators:
                    for first, first_count in semantic_choices(inner):
                        for second, second_count in semantic_choices(hole):
                            key = (symbol, id(first), id(second))
                            if key not in evaluated:
                                evaluated[key] = self.evaluate_operator(symbol, first, second)
                            candidates.append(evaluated[key])
                            weights.append(first_count * second_count)
        except AlternativeLimit:
            if not candidates:
                raise
            return self.choose(candidates, force_best=True)
        return self.choose(candidates, weights if any((weight != 1 for weight in weights)) else None)

    def apply_unary(self, head, semantic):
        candidates = []
        evaluated = {}
        weights = []
        try:
            for symbol in head.operators:
                for candidate, count in semantic_choices(semantic):
                    key = (symbol, id(candidate))
                    if key not in evaluated:
                        evaluated[key] = self.evaluate_unary(symbol, candidate)
                    candidates.append(evaluated[key])
                    weights.append(count)
        except AlternativeLimit:
            if not candidates:
                raise
            return self.choose(candidates, force_best=True)
        return self.choose(candidates, weights if any((weight != 1 for weight in weights)) else None)

    def evaluate_unary(self, symbol, semantic):
        self.check_alternative_budget()
        self.operator_evaluations += 1
        argument = semantic
        if semantic.unary_argument is not None:
            argument = semantic.unary_argument
        values = apply_operator(symbol, argument.values)
        result = SemanticObject(None, values, is_atomic=False)
        result.evaluation = (symbol, argument.evaluation)
        result.unary_argument = argument
        result.evaluation_children = (argument,)
        result.used_semantic_children = (argument,)
        result.used_operator = True
        if self.candidate_observer is not None:
            self.candidate_observer(result)
        return result

    def evaluate_operator(self, symbol, left, right):
        self.check_alternative_budget()
        self.operator_evaluations += 1
        values = apply_operator(symbol, left.values, right.values)
        result = SemanticObject(None, values, is_atomic=False)
        result.evaluation = (symbol, left.evaluation, right.evaluation)
        result.evaluation_children = (left, right)
        result.used_semantic_children = (left, right)
        result.used_operator = True
        if self.candidate_observer is not None:
            self.candidate_observer(result)
        return result

    def check_alternative_budget(self):
        if self.replaying:
            if self.alternative_evaluation_limit is not None and self.replay_operator_evaluations >= self.alternative_evaluation_limit:
                self.mark_replay_limit('operator_evaluations')
                raise AlternativeLimit('operator_evaluations')
            self.replay_operator_evaluations += 1

    def mark_replay_limit(self, reason):
        for scope in self.replay_limit_scopes:
            scope.add(reason)
        if reason not in self.current_replay_limits:
            self.current_replay_limits.add(reason)
            self.replay_limit_hits[reason] += 1
