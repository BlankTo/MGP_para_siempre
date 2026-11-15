from itertools import chain, product

from mgp.engine.objects import IncompleteObject, OperatorObject, SemanticList, SemanticObject
from mgp.engine.operator_sequence import OperatorSequence, operator_count


class AlternativeLimit(RuntimeError):
    pass


def constructive_pair(left, right):
    if isinstance(left, SemanticObject) and isinstance(right, SemanticObject):
        return False
    if isinstance(left, OperatorObject) and isinstance(right, OperatorObject):
        return left.is_atomic == right.is_atomic and left.arity == right.arity
    if isinstance(left, OperatorObject) or isinstance(right, OperatorObject):
        head, other = (left, right) if isinstance(left, OperatorObject) else (right, left)
        if isinstance(other, IncompleteObject):
            return False
        if not isinstance(other, SemanticObject):
            return False
        return head.arity == 2 or (operator_count(head.operators) == 1 and (not isinstance(other, SemanticList)))
    if isinstance(left, IncompleteObject) and isinstance(right, IncompleteObject):
        return False
    partial, semantic = (left, right) if isinstance(left, IncompleteObject) else (right, left)
    return isinstance(partial, IncompleteObject) and isinstance(semantic, SemanticObject) and (len(partial.head.operators) == 1) and (not isinstance(partial.semantic, SemanticList)) and (not isinstance(semantic, SemanticList))


def constructive_class(node):
    if isinstance(node, OperatorObject):
        return ('operator', node.is_atomic, node.arity, operator_count(node.operators))
    if isinstance(node, IncompleteObject):
        return ('incomplete', len(node.head.operators) == 1, isinstance(node.semantic, SemanticList))
    return ('semantic', isinstance(node, SemanticList))


def legal_partners(workspace):
    groups, representatives, classes = ({}, {}, [])
    for index, node in enumerate(workspace):
        category = constructive_class(node)
        classes.append(category)
        groups.setdefault(category, []).append(index)
        representatives.setdefault(category, node)
    allowed = {}
    for category, node in representatives.items():
        allowed[category] = sorted((index for other, indices in groups.items() if constructive_pair(node, representatives[other]) for index in indices))
    partners = {}
    for index, category in enumerate(classes):
        others = allowed[category]
        if index in others:
            others = [other for other in others if other != index]
        if others:
            partners[index] = others
    return partners


def variant_key(node):
    if isinstance(node, IncompleteObject):
        return ('incomplete', OperatorSequence(symbols=node.head.operators, arity=node.head.arity) if not isinstance(node.head.operators, OperatorSequence) else node.head.operators, variant_key(node.semantic))
    if isinstance(node, OperatorObject):
        return ('operator', OperatorSequence(symbols=node.operators, arity=node.arity) if not isinstance(node.operators, OperatorSequence) else node.operators, node.is_atomic)
    return ('semantic', node.evaluation)


class AlternativeReplay:

    def __init__(self, merge, max_variants, max_pairs):
        self.merge, self.max_variants, self.max_pairs = (merge, max_variants, max_pairs)
        self.cache = merge.replay_variants_cache

    def unique(self, nodes):
        found = {}
        for node in nodes:
            key = variant_key(node)
            if key in found:
                self.merge.replay_duplicates += 1
            else:
                if len(found) == self.max_variants:
                    self.merge.mark_replay_limit('variants_per_tree')
                    break
                found[key] = node
        return list(found.values())

    def variants(self, node):
        if node in self.cache:
            for reason in self.merge.replay_cache_limits.get(node, ()):
                self.merge.mark_replay_limit(reason)
            return self.cache[node]
        limits = set()
        self.merge.replay_limit_scopes.append(limits)
        try:
            result = self.build_variants(node)
        finally:
            self.merge.replay_limit_scopes.pop()
        self.cache[node] = result
        self.merge.replay_cache_limits[node] = tuple(sorted(limits))
        return result

    def build_variants(self, node):
        choices = getattr(node, 'ic_alternatives', ())
        if choices:
            result = self.unique(chain([node], (item for choice in choices for item in self.variants(choice))))
        elif isinstance(node, IncompleteObject):
            result = self.unique(chain([node], (IncompleteObject(item, node.head) for item in self.variants(node.semantic))))
        elif isinstance(node, SemanticList):
            raise ValueError('revisit_ic requires semantic_list_enabled=False')
        elif isinstance(node, SemanticObject) and node.evaluation_children:
            child_sets = [self.variants(child) for child in node.evaluation_children]
            result = self.unique(self.reconstruct(node, child_sets))
        else:
            result = [node]
        return result

    def reconstruct(self, node, child_sets):
        yield node
        for index, children in enumerate(product(*child_sets)):
            if index >= self.max_pairs:
                self.merge.mark_replay_limit('reconstruction_pairs')
                break
            evaluation = (node.evaluation[0], *(child.evaluation for child in children))
            if evaluation == node.evaluation:
                continue
            try:
                if len(children) == 1:
                    yield self.merge.evaluate_unary(node.evaluation[0], children[0])
                else:
                    yield self.merge.evaluate_operator(node.evaluation[0], *children)
            except AlternativeLimit:
                break

    def run(self, left, right):
        baseline = self.merge.evaluate_merge(left, right)
        self.merge.replaying = True
        try:
            return self.expand(left, right, baseline)
        finally:
            self.merge.replaying = False

    def expand(self, left, right, baseline):
        lefts, rights = (self.variants(left), self.variants(right))
        results = self.unique(self.merge_results(left, right, lefts, rights, baseline))
        if len(results) == 1:
            return results[0]
        self.merge.replay_expanded_merges += 1
        if isinstance(results[0], IncompleteObject):
            return IncompleteObject(self.merge.choose([item.semantic for item in results], force_best=bool(self.merge.current_replay_limits)), results[0].head)
        if not all((isinstance(item, SemanticObject) for item in results)):
            raise ValueError('counterfactual alternatives must retain the same syntactic type')
        return self.merge.choose(results, force_best=bool(self.merge.current_replay_limits))

    def merge_results(self, left, right, lefts, rights, baseline):
        yield baseline
        original = (variant_key(left), variant_key(right))
        for index, (a, b) in enumerate(product(lefts, rights)):
            if index >= self.max_pairs:
                self.merge.mark_replay_limit('merge_pairs')
                break
            self.merge.replay_pairs += 1
            if (variant_key(a), variant_key(b)) == original:
                continue
            try:
                yield self.merge.evaluate_merge(a, b)
            except AlternativeLimit:
                break
