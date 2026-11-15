import math

from mgp.archives.building_blocks import BuildingBlockArchive
from mgp.engine.objects import IncompleteObject, SemanticList, SemanticObject, selection_source


def validate_best_bb(config):
    enabled = config.get('best_bb_archive_enabled', False)
    if type(enabled) is not bool:
        raise ValueError('best_bb_archive_enabled must be a boolean')
    for name, default in (('best_bb_alpha', 0.25), ('best_bb_decay', 0.9)):
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or (not math.isfinite(value)) or (not 0 <= value <= 1):
            raise ValueError(name + ' must be finite and between 0 and 1')
    if enabled and config.get('sympy_enabled') is not True:
        raise ValueError('the best BB archive requires sympy_enabled=True')


def active_children(node):
    if isinstance(node, SemanticList):
        raise ValueError('the selected best must have resolved active alternatives')
    if isinstance(node, IncompleteObject):
        return (node.semantic,)
    return getattr(node, 'used_semantic_children', ())


class BestBuildingBlockArchive(BuildingBlockArchive):

    def __init__(self, target, config):
        validate_best_bb(config)
        super().__init__(target, {**config, 'protocol_version': 2, 'sympy_enabled': True, 'building_block_root_weight': config.get('best_bb_alpha', 0.25), 'building_block_depth_decay': config.get('best_bb_decay', 0.9)})
        self.built = False

    def build(self, best):
        if self.built:
            return
        self.built = True
        if best is None:
            return
        if not isinstance(best, SemanticObject) or isinstance(best, SemanticList):
            raise ValueError('the best must be a resolved semantic object')
        ordered, children, visited, active = ([], {}, set(), set())
        pending = [(best, False)]
        while pending:
            node, ready = pending.pop()
            if node in visited:
                continue
            if ready:
                active.remove(node)
                visited.add(node)
                ordered.append(node)
                continue
            if node in active:
                raise ValueError('cycle in active evaluation dependencies')
            active.add(node)
            children[node] = tuple(active_children(node))
            pending.append((node, True))
            pending.extend(((child, False) for child in children[node]))
        counts, incoming, depths = ({best: 1}, {best: {}}, {best: 0})
        root_quality = self.quality(best)
        for node in reversed(ordered):
            count = counts[node]
            coefficients = incoming[node]
            semantic = isinstance(node, SemanticObject)
            quality = self.quality(node) if semantic else 0.0
            imported = semantic and node.archive_origin and (not self.imported_credit)
            transparent_import = imported and bool(children[node]) and (not node.used_operator)
            transparent = transparent_import or (semantic and selection_source(node) is not None)
            if semantic:
                if self.eligible(node) and (not transparent):
                    row = self.entries[self.entry_index(node)]
                    row['occurrences'] += 0 if imported else count
                    row['used_occurrences'] += count if not imported or coefficients else 0
                    row['tree_count'] = int(not imported or bool(coefficients))
                    row['individual_credit'] += 0.0 if imported else quality * count
                    for depth, value in coefficients.items():
                        bucket = row['credit_by_depth']
                        bucket[depth] = bucket.get(depth, 0.0) + value
                    row['root_quality_sum'] += root_quality * count if not imported or coefficients else 0.0
                    row['best_root_quality'] = root_quality if not imported or coefficients else 0.0
                    row['min_depth'] = min(row['min_depth'], depths[node]) if row['min_depth'] is not None else depths[node]
            outgoing = {depth + 1: value for depth, value in coefficients.items()} if semantic and (not transparent) else coefficients
            if semantic and (not imported) and (not transparent):
                outgoing[1] = outgoing.get(1, 0.0) + quality * count
            for child in children[node]:
                counts[child] = counts.get(child, 0) + count
                bucket = incoming.setdefault(child, {})
                for depth, value in outgoing.items():
                    bucket[depth] = bucket.get(depth, 0.0) + value
                distance = depths[node] + int(semantic and (not transparent))
                depths[child] = min(depths.get(child, distance), distance)
        for row in self.entries:
            row['inherited_credit'] = self.weight * sum((value * self.decay ** depth for depth, value in row['credit_by_depth'].items()))
            row['score'] = row['individual_credit'] + row['inherited_credit']
