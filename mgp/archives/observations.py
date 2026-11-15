from collections import Counter
from weakref import WeakKeyDictionary, WeakSet

import numpy as np

from mgp.engine.objects import SemanticList, SemanticObject
from mgp.engine.semantics import symbolic_enabled
from mgp.engine.symbolic import symbolic_data


def graph_occurrences(root, used=False, decay=1.0):
    ordered, visited, children = ([], set(), {})
    pending = [(root, False)]
    while pending:
        node, ready = pending.pop()
        if node in visited:
            continue
        if ready:
            visited.add(node)
            ordered.append(node)
            continue
        if used:
            descendants = tuple(((child, 1) for child in node.used_semantic_children))
        elif node.children:
            descendants = tuple(((child, 1) for child in node.children))
        elif isinstance(node, SemanticList):
            descendants = tuple(zip(node.alternatives, node.multiplicities))
        elif isinstance(node, SemanticObject):
            descendants = tuple(((child, 1) for child in node.used_semantic_children))
        else:
            descendants = ()
        children[node] = descendants
        pending.append((node, True))
        pending.extend(((child, False) for child, count in descendants))
    counts, mass, depths = ({root: 1}, {root: 1.0}, {root: 0})
    for node in reversed(ordered):
        for child, multiplicity in children[node]:
            counts[child] = counts.get(child, 0) + counts[node] * multiplicity
            mass[child] = mass.get(child, 0.0) + mass[node] * multiplicity * decay
            depths[child] = min(depths.get(child, depths[node] + 1), depths[node] + 1)
    return [(node, counts[node], mass[node], depths[node]) for node in reversed(ordered)]


def used_counts(root):
    pending = [(root, False)]
    while pending:
        node, ready = pending.pop()
        if getattr(node, '_used_counts', None) is not None:
            continue
        children = node.used_semantic_children
        if not ready and children:
            pending.append((node, True))
            pending.extend(((child, False) for child in children))
            continue
        bottom = sum((child._used_counts[0] for child in children))
        if not node.is_atomic and (not bottom):
            bottom = 1
        operators = int(node.used_operator) + sum((child._used_counts[1] for child in children))
        node._used_counts = (bottom, operators)
    return root._used_counts


class WorkspaceMetrics:

    def __init__(self, workspace, symbolic_counts):
        self.symbolic_counts = symbolic_counts
        self.resolved = 0
        self.bottom = 0
        self.operators = 0
        self.semantic_lists = 0
        self.alternatives = 0
        self.key_counts = Counter()
        self.nonatomic_unique = 0
        self.cache = WeakKeyDictionary()
        for node in workspace:
            self.add(node)

    def contribution(self, node):
        result = self.cache.get(node)
        if result is not None:
            return result
        resolved = isinstance(node, SemanticObject) and (not isinstance(node, SemanticList))
        bottom, operators = used_counts(node) if resolved else (0, 0)
        is_list = isinstance(node, SemanticList)
        keys = ()
        if self.symbolic_counts.enabled:
            options = node.alternatives if is_list else (node,) if resolved else ()
            keys = frozenset((symbolic_data(item)[0] for item in options if not np.any(np.isinf(item.values)) and np.any(np.isfinite(item.values))))
        result = (int(resolved), bottom, operators, int(is_list), len(node.alternatives) if is_list else 0, keys)
        self.cache[node] = result
        return result

    def change(self, node, delta):
        resolved, bottom, operators, lists, alternatives, keys = self.contribution(node)
        self.resolved += delta * resolved
        self.bottom += delta * bottom
        self.operators += delta * operators
        self.semantic_lists += delta * lists
        self.alternatives += delta * alternatives
        for key in keys:
            previous = self.key_counts[key]
            current = previous + delta
            if current < 0:
                raise ValueError('workspace metrics are out of sync')
            if previous == 0 and current > 0 and (key not in self.symbolic_counts.atoms):
                self.nonatomic_unique += 1
            elif previous > 0 and current == 0 and (key not in self.symbolic_counts.atoms):
                self.nonatomic_unique -= 1
            if current:
                self.key_counts[key] = current
            else:
                self.key_counts.pop(key, None)

    def add(self, node):
        self.change(node, 1)

    def remove(self, node):
        self.change(node, -1)

    def measure(self):
        result = {'resolved_tree_count': self.resolved, 'mean_bottom_used_semantic_count': self.bottom / self.resolved if self.resolved else None, 'mean_used_operator_count': self.operators / self.resolved if self.resolved else None, 'semantic_list_count': self.semantic_lists, 'semantic_alternative_count': self.alternatives}
        if self.symbolic_counts.enabled:
            result.update({'symbolic_unique_semantic_count': len(self.key_counts), 'symbolic_unique_nonatomic_count': self.nonatomic_unique, 'cumulative_symbolic_unique_semantic_count': len(self.symbolic_counts.seen), 'cumulative_symbolic_unique_nonatomic_count': self.symbolic_counts.seen_nonatomic_count})
        return result


def available_semantics(workspace):
    for node in workspace:
        if isinstance(node, SemanticList):
            yield from node.alternatives
        elif isinstance(node, SemanticObject):
            yield node


class SymbolicCounts:

    def __init__(self, config, atoms):
        self.enabled = symbolic_enabled(config)
        self.seen = set()
        self.seen_nonatomic_count = 0
        self.nodes = WeakSet()
        self.semantic_occurrences = set()
        self.atoms = {symbolic_data(atom)[0] for atom in atoms} if self.enabled else set()

    def observe(self, current):
        if not self.enabled:
            return
        pending = [current]
        while pending:
            node = pending.pop()
            if isinstance(node, SemanticObject):
                token = node.occurrence_identity
                if token in self.semantic_occurrences:
                    continue
                self.semantic_occurrences.add(token)
            else:
                if node in self.nodes:
                    continue
                self.nodes.add(node)
            pending.extend(node.children or getattr(node, 'used_semantic_children', ()))
            if isinstance(node, SemanticList):
                pending.extend(node.alternatives)
            if isinstance(node, SemanticObject) and (not isinstance(node, SemanticList)):
                if not np.any(np.isinf(node.values)) and np.any(np.isfinite(node.values)):
                    key = symbolic_data(node)[0]
                    if key not in self.seen:
                        self.seen.add(key)
                        if key not in self.atoms:
                            self.seen_nonatomic_count += 1

    def measure(self, workspace):
        if not self.enabled:
            return {}
        keys = {symbolic_data(node)[0] for node in available_semantics(workspace) if not np.any(np.isinf(node.values)) and np.any(np.isfinite(node.values))}
        return {'symbolic_unique_semantic_count': len(keys), 'symbolic_unique_nonatomic_count': len(keys - self.atoms), 'cumulative_symbolic_unique_semantic_count': len(self.seen), 'cumulative_symbolic_unique_nonatomic_count': self.seen_nonatomic_count}


def used_path_depths(root):
    ordered = [node for node, _, _, _ in graph_occurrences(root, used=True)]
    masses = {root: {0: 1}}
    for node in ordered:
        for child in node.used_semantic_children:
            bucket = masses.setdefault(child, {})
            for depth, count in masses[node].items():
                bucket[depth + 1] = bucket.get(depth + 1, 0) + count
    return masses
