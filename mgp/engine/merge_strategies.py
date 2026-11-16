from itertools import chain, product

from mgp.engine.objects import IncompleteObject, OperatorObject, SemanticList, SemanticObject
from mgp.engine.operator_sequence import OperatorSequence


class AlternativeLimit(RuntimeError):
    pass


KINDS = ('incomplete', 'operator', 'semantic')


def merge_kind(node):
    if isinstance(node, OperatorObject):
        return 'operator'
    if isinstance(node, IncompleteObject):
        return 'incomplete'
    return 'semantic'


def merge_pair(left_kind, right_kind):
    return '+'.join(sorted((left_kind, right_kind)))


def constructive_merges(spec):
    pairs = set()
    for item in spec:
        kinds = item.split('+') if isinstance(item, str) else ()
        if len(kinds) != 2 or any(kind not in KINDS for kind in kinds):
            raise ValueError(f'unknown constructive merge {item!r}, use two of {", ".join(KINDS)} joined by +')
        pairs.add(merge_pair(*kinds))
    return tuple(sorted(pairs))


def constructive_pair(left, right, allowed):
    return merge_pair(merge_kind(left), merge_kind(right)) in allowed


def legal_partners(workspace, allowed):
    kinds = [merge_kind(node) for node in workspace]
    groups = {}
    for index, kind in enumerate(kinds):
        groups.setdefault(kind, []).append(index)
    lists = {kind: sorted(index for other, indices in groups.items() if merge_pair(kind, other) in allowed for index in indices) for kind in groups}
    with_self = {kind for kind in groups if merge_pair(kind, kind) in allowed}
    partners = {}
    for index, kind in enumerate(kinds):
        others = [other for other in lists[kind] if other != index] if kind in with_self else lists[kind]
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
