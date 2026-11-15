import hashlib
import json
from functools import cmp_to_key

import numpy as np

from mgp.archives.observations import graph_occurrences, used_path_depths
from mgp.engine.expression_graph import archive_expression
from mgp.engine.fitness import object_fitness, rmse
from mgp.engine.objects import IncompleteObject, SemanticList, SemanticObject
from mgp.engine.protocol import current_protocol
from mgp.engine.semantics import equivalence_name, select_equivalence, symbolic_enabled
from mgp.engine.symbolic import archive_rank_compare, symbolic_data, symbolic_has_variables
from mgp.engine.symbolic import expression_key


class BuildingBlockArchive:

    def __init__(self, target, config):
        self.target = target
        self.posthoc = current_protocol(config)
        self.weight = config.get('building_block_root_weight', 0.25)
        self.decay = config.get('building_block_depth_decay', 0.9)
        if not np.isfinite(self.weight) or not 0 <= self.weight <= 1:
            raise ValueError('building_block_root_weight must be between zero and one')
        if not np.isfinite(self.decay) or not 0 <= self.decay <= 1:
            raise ValueError('building_block_depth_decay must be between zero and one')
        self.symbolic = symbolic_enabled(config)
        self.equivalence = select_equivalence(equivalence_name(config), config.get('equivalence_tolerance', 1e-10), config.get('equivalence_atol', 1e-10), config.get('equivalence_rtol', 1e-10))
        finite_target = target[np.isfinite(target)]
        self.scale = rmse(np.zeros_like(finite_target), finite_target) or 1.0
        self.entries = []
        self.total_score = 0.0
        self.representatives = []
        self.keys = {}
        self.identities = {}
        self.seen_roots = set()
        self.credit_version = config.get('bb_credit_version', 1)
        if self.credit_version not in (1, 2):
            raise ValueError('bb_credit_version must be 1 or 2')
        self.history_seen = set()
        self.constructions = {}
        self.descendant_depths = {}
        self.eligibility = {}
        self._depth_weight = {0: 1.0}
        self.defer_expression_payload = config.get('defer_building_block_expressions', False)
        self.defer_credit = bool(config.get('defer_building_block_credit', False))
        self._credit_order = []
        self._credit_finalized = False
        self.imported_credit = config.get('bb_imported_credit', False)
        if type(self.imported_credit) is not bool:
            raise ValueError('bb_imported_credit must be boolean')

    def quality(self, current):
        error = object_fitness(current, self.target)
        if not np.isfinite(error):
            return 0.0
        return 1.0 / (1.0 + error / self.scale)

    def entry_index(self, current):
        token = current.semantic_identity
        if token in self.identities:
            return self.identities[token]
        canonical = None
        if self.symbolic:
            identity, canonical = symbolic_data(current)[:2]
        else:
            values = current.values.copy()
            values[values == 0] = 0.0
            identity = hashlib.sha256(values.tobytes()).hexdigest()
        index = self.keys.get(identity)
        if index is None and (not self.symbolic):
            for number, representative in enumerate(self.representatives):
                if self.equivalence(current.values, representative.values):
                    index = number
                    break
        if index is None:
            expression, expression_graph = (None, None) if self.defer_expression_payload else archive_expression(current)
            if not self.symbolic:
                canonical = expression
            index = len(self.entries)
            self.representatives.append(current)
            self.entries.append({'equivalence_key': identity, 'expression': expression, 'expression_graph': expression_graph, 'canonical_expression': canonical, 'occurrences': 0, 'used_occurrences': 0, 'tree_count': 0, 'individual_credit': 0.0, 'inherited_credit': 0.0, 'score': 0.0, 'root_quality_sum': 0.0, 'best_root_quality': 0.0, 'min_depth': None, **({'credit_by_depth': {}} if self.posthoc else {}), 'values_json': json.dumps([float(value) if np.isfinite(value) else None for value in current.values], allow_nan=False)})
        self.keys[identity] = index
        self.identities[token] = index
        return index

    def observe(self, root):
        if self.credit_version == 2:
            self.observe_snapshot(root)
        else:
            self.observe_legacy(root)

    def eligible(self, node):
        if node.is_atomic:
            return False
        token = node.semantic_identity
        if token not in self.eligibility:
            finite = node.values[np.isfinite(node.values)]
            self.eligibility[token] = not np.any(np.isinf(node.values)) and len(finite) > 0 and (not np.all(finite == finite[0])) and (not self.symbolic or symbolic_has_variables(node))
        return self.eligibility[token]

    def observe_snapshot(self, root):
        if self.defer_credit:
            return self.observe_deferred(root)
        pending = [(root, False)]
        while pending:
            node, ready = pending.pop()
            if node in self.history_seen:
                continue
            if not ready:
                pending.append((node, True))
                children = list(getattr(node, 'children', ()))
                children.extend(getattr(node, 'evaluation_children', ()))
                if isinstance(node, SemanticList):
                    children.extend(node.alternatives)
                if isinstance(node, IncompleteObject):
                    children.append(node.semantic)
                pending.extend(((child, False) for child in children))
                continue
            self.history_seen.add(node)
            if not isinstance(node, SemanticObject) or isinstance(node, SemanticList):
                continue
            token = node.semantic_identity
            if token in self.constructions:
                continue
            self.constructions[token] = node
            qualifies = self.eligible(node)
            distances = {}
            for child in node.evaluation_children:
                child_token = child.semantic_identity
                if self.eligibility.get(child_token, False):
                    bucket = distances.setdefault(child_token, {})
                    bucket[1] = bucket.get(1, 0) + 1
                for descendant, depths in self.descendant_depths.get(child_token, {}).items():
                    bucket = distances.setdefault(descendant, {})
                    for depth, count in depths.items():
                        next_depth = depth + 1
                        bucket[next_depth] = bucket.get(next_depth, 0) + count
            self.descendant_depths[token] = distances
            quality = self.quality(node)
            touched = set()
            imported = node.archive_origin and (not self.imported_credit)
            if qualifies:
                index = self.entry_index(node)
                row = self.entries[index]
                if not imported:
                    row['occurrences'] += 1
                    row['used_occurrences'] += 1
                    row['individual_credit'] += quality
                    row['root_quality_sum'] += quality
                    row['best_root_quality'] = max(row['best_root_quality'], quality)
                    row['min_depth'] = 0
                    touched.add(index)
            if imported:
                continue
            for descendant, depths in distances.items():
                index = self.entry_index(self.constructions[descendant])
                row = self.entries[index]
                uses = sum(depths.values())
                depth = min(depths)
                row['min_depth'] = depth if row['min_depth'] is None else min(row['min_depth'], depth)
                row['used_occurrences'] += uses
                row['root_quality_sum'] += quality * uses
                row['best_root_quality'] = max(row['best_root_quality'], quality)
                weight = self._depth_weight
                for depth in depths:
                    if depth not in weight:
                        weight[depth] = self.decay ** depth
                row['inherited_credit'] += self.weight * quality * sum((count * weight[depth] for depth, count in depths.items()))
                if self.posthoc:
                    for depth, count in depths.items():
                        bucket = row['credit_by_depth']
                        bucket[depth] = bucket.get(depth, 0.0) + quality * count
                touched.add(index)
            for index in touched:
                row = self.entries[index]
                row['tree_count'] += 1
                old_score = row['score']
                row['score'] = row['individual_credit'] + row['inherited_credit']
                self.total_score += row['score'] - old_score

    def observe_deferred(self, root):
        pending = [(root, False)]
        while pending:
            node, ready = pending.pop()
            if node in self.history_seen:
                continue
            if not ready:
                pending.append((node, True))
                children = list(getattr(node, 'children', ()))
                children.extend(getattr(node, 'evaluation_children', ()))
                if isinstance(node, SemanticList):
                    children.extend(node.alternatives)
                if isinstance(node, IncompleteObject):
                    children.append(node.semantic)
                pending.extend((child, False) for child in children if child not in self.history_seen)
                continue
            self.history_seen.add(node)
            if not isinstance(node, SemanticObject) or isinstance(node, SemanticList):
                continue
            token = node.semantic_identity
            if token in self.constructions:
                continue
            self.constructions[token] = node
            self._credit_order.append(token)
            if self.eligible(node):
                self.entry_index(node)

    def finalize_deferred(self):
        if not self.defer_credit or self._credit_finalized:
            return
        self._credit_finalized = True
        order = self._credit_order
        if not order:
            return
        positions = {token: i for i, token in enumerate(order)}
        incoming = {}
        incoming_paths = {}
        incoming_origins = {}
        incoming_best = {}
        incoming_depth = {}
        scored_roots = {}
        quality = {}
        active = {}
        for token in order:
            node = self.constructions[token]
            quality[token] = self.quality(node)
            active[token] = not node.archive_origin or self.imported_credit
        for token in reversed(order):
            node = self.constructions[token]
            offset = positions[token]
            root_active = active[token]
            q = quality[token]
            weighted = incoming.pop(token, {})
            paths = incoming_paths.pop(token, 0)
            origins = incoming_origins.pop(token, 0)
            best = incoming_best.pop(token, 0.0)
            closest = incoming_depth.pop(token, None)
            if root_active:
                origins |= 1 << offset
                best = max(best, q)
                closest = 0
            index = self.identities.get(token)
            if index is not None:
                row = self.entries[index]
                ancestor_uses = paths
                row['occurrences'] += int(root_active)
                row['used_occurrences'] += ancestor_uses + int(root_active)
                scored_roots[index] = scored_roots.get(index, 0) | origins
                row['individual_credit'] += q if root_active else 0.0
                row['root_quality_sum'] += sum(weighted.values()) + (q if root_active else 0.0)
                row['best_root_quality'] = max(row['best_root_quality'], best)
                if closest is not None:
                    row['min_depth'] = closest if row['min_depth'] is None else min(row['min_depth'], closest)
                if self.posthoc:
                    for depth, amount in weighted.items():
                        bucket = row['credit_by_depth']
                        bucket[depth] = bucket.get(depth, 0.0) + amount
                row['inherited_credit'] += self.weight * sum((value * self.decay ** depth for depth, value in weighted.items()))
            if not node.evaluation_children:
                continue
            for child in node.evaluation_children:
                child_token = child.semantic_identity
                target = incoming.setdefault(child_token, {})
                if root_active:
                    target[1] = target.get(1, 0.0) + q
                for depth, amount in weighted.items():
                    nxt = depth + 1
                    target[nxt] = target.get(nxt, 0.0) + amount
                incoming_paths[child_token] = incoming_paths.get(child_token, 0) + paths + int(root_active)
                incoming_origins[child_token] = incoming_origins.get(child_token, 0) | origins
                incoming_best[child_token] = max(incoming_best.get(child_token, 0.0), best)
                if closest is not None:
                    distance = closest + 1
                    incoming_depth[child_token] = min(incoming_depth.get(child_token, distance), distance)
        for index, row in enumerate(self.entries):
            row['tree_count'] = scored_roots.get(index, 0).bit_count()
            row['score'] = row['individual_credit'] + row['inherited_credit']
        self.total_score = sum(row['score'] for row in self.entries)

    def observe_legacy(self, root):
        if isinstance(root, IncompleteObject):
            self.observe(root.semantic)
            return
        if isinstance(root, SemanticList):
            for alternative in root.alternatives:
                self.observe(alternative)
            return
        if not isinstance(root, SemanticObject) or root.occurrence_identity in self.seen_roots:
            return
        self.seen_roots.add(root.occurrence_identity)
        root_quality = self.quality(root)
        used = {}
        for node, count, mass, depth in graph_occurrences(root, used=True, decay=self.decay):
            previous = used.get(node.occurrence_identity, (0, 0.0))
            used[node.occurrence_identity] = (previous[0] + count, previous[1] + mass)
        depths = {}
        if self.posthoc:
            for node, distances in used_path_depths(root).items():
                bucket = depths.setdefault(node.occurrence_identity, {})
                for distance, count in distances.items():
                    bucket[distance] = bucket.get(distance, 0) + count
        touched = set()
        occurrences = {}
        for current, count, mass, depth in graph_occurrences(root):
            if not isinstance(current, SemanticObject) or isinstance(current, SemanticList):
                continue
            if current.is_atomic or np.any(np.isinf(current.values)) or (not np.any(np.isfinite(current.values))):
                continue
            finite_values = current.values[np.isfinite(current.values)]
            if np.all(finite_values == finite_values[0]):
                continue
            if self.symbolic and (not symbolic_data(current)[2].free_symbols):
                continue
            token = current.occurrence_identity
            previous = occurrences.get(token, (current, 0, depth))
            occurrences[token] = (current, previous[1] + count, min(previous[2], depth))
        for token, (current, count, depth) in occurrences.items():
            index = self.entry_index(current)
            row = self.entries[index]
            used_count, discounted = used.get(token, (0, 0.0))
            individual = self.quality(current) * count
            inherited = self.weight * discounted * root_quality if current is not root else 0.0
            row['occurrences'] += count
            row['used_occurrences'] += used_count
            row['individual_credit'] += individual
            row['inherited_credit'] += inherited
            if self.posthoc and current is not root:
                for distance, multiplicity in depths.get(token, {}).items():
                    bucket = row['credit_by_depth']
                    bucket[distance] = bucket.get(distance, 0.0) + root_quality * multiplicity
            old_score = row['score']
            row['score'] = row['individual_credit'] + row['inherited_credit']
            self.total_score += row['score'] - old_score
            row['root_quality_sum'] += root_quality * used_count
            if used_count:
                row['best_root_quality'] = max(row['best_root_quality'], root_quality)
            row['min_depth'] = depth if row['min_depth'] is None else min(depth, row['min_depth'])
            touched.add(index)
        for index in touched:
            self.entries[index]['tree_count'] += 1

    def limit(self, size):
        self.unbounded_entries, self.unbounded_representatives = list(self.entries), list(self.representatives)
        ranked = sorted(range(len(self.entries)), key=lambda j: (-self.entries[j]['score'], self.entries[j]['equivalence_key']))
        keep = sorted(ranked[:size])
        self.entries = [self.entries[j] for j in keep]
        self.representatives = [self.representatives[j] for j in keep]
        self.keys = {e['equivalence_key']: i for i, e in enumerate(self.entries)}
        self.total_score = sum(e['score'] for e in self.entries)

    def rows(self, subtargets=(), tolerance=1e-08):
        rows = []
        for entry, representative in zip(self.entries, self.representatives):
            matches = [identifier for identifier, expression, values in subtargets if (symbolic_data(representative)[0] == expression_key(expression) if self.posthoc else rmse(representative.values, values) <= tolerance)]
            rows.append({**{key: value for key, value in entry.items() if key != 'credit_by_depth'}, 'matched_subtarget_ids': json.dumps(matches)})
        if self.symbolic:
            ordered = sorted(zip(self.representatives, rows), key=cmp_to_key(archive_rank_compare))
            return [row for node, row in ordered]
        return sorted(rows, key=lambda row: (-row['score'], row['equivalence_key']))
