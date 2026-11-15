from mgp.engine.fitness import object_fitness
from mgp.engine.objects import SemanticObject
from mgp.engine.symbolic import symbolic_data


def macro_copy(original):
    node = SemanticObject(original.expression, original.values, is_atomic=False)
    node.evaluation = original.evaluation
    node.symbolic_identity = original.symbolic_identity
    node.archive_origin = True
    return node


def archive_candidates(derivation, policy):
    if policy == 'fresh':
        return []
    if policy == 'base_fitness':
        return [(node, -object_fitness(node, derivation.target)) for node in derivation.archive.objects]
    archive = derivation.best_bb_archive if policy == 'best_bb' else derivation.building_block_archive
    return [(node, row['score']) for node, row in zip(archive.representatives, archive.entries)]


class ReuseMemory:

    def __init__(self, policy, capacity, shortlist, injected):
        self.policy, self.capacity = (policy, capacity)
        self.shortlist, self.injected = (shortlist, injected)
        self.entries = {}
        self.evictions = 0

    def ranked(self):
        return sorted(self.entries, key=lambda key: (-self.entries[key]['score'], key))

    def update(self, derivation, index):
        self.last_evicted = {}
        admitted, updated = ([], [])
        for node, score in archive_candidates(derivation, self.policy):
            if self.policy != 'base_fitness' and score <= 0:
                continue
            key = symbolic_data(node)[0]
            if key not in self.entries:
                self.entries[key] = {'node': macro_copy(node), 'score': score, 'born': index, 'last_evidence': index, 'evidence_runs': 1}
                admitted.append(key)
            else:
                entry = self.entries[key]
                entry['score'] = max(entry['score'], score) if self.policy == 'base_fitness' else entry['score'] + score
                entry['last_evidence'] = index
                entry['evidence_runs'] += 1
                updated.append(key)
        evicted = []
        if self.capacity is not None:
            for key in self.ranked()[self.capacity:]:
                self.last_evicted[key] = self.entries[key]
                del self.entries[key]
                evicted.append(key)
        self.evictions += len(evicted)
        return {'admitted': admitted, 'updated': updated, 'evicted': evicted}

    def inject(self, workspace, generator):
        keys = sorted(self.entries)
        if self.policy == 'random_bb':
            generator.shuffle(keys)
        else:
            keys = self.ranked()
        shortlist = keys[:self.shortlist]
        positions = [i for i, node in enumerate(workspace) if isinstance(node, SemanticObject)]
        generator.shuffle(positions)
        selected = generator.sample(shortlist, min(self.injected, len(positions), len(shortlist)))
        events = []
        for position, key in zip(positions, selected):
            entry = self.entries[key]
            workspace[position] = macro_copy(entry['node'])
            events.append({'key': key, 'expression': entry['node'].expression, 'score': entry['score'], 'rank': keys.index(key) + 1, 'born': entry['born'], 'last_evidence': entry['last_evidence'], 'position': position})
        return events

    def snapshot(self, limit=20):
        return [{'key': key, 'expression': self.entries[key]['node'].expression, **{k: v for k, v in self.entries[key].items() if k != 'node'}} for key in self.ranked()[:limit]]


def policy_config(settings, policy):
    return {**settings, 'archive_enabled': policy in ('observe', 'base_fitness'), 'base_archive_policies': ['symbolic'], 'bb_credit_version': 2, 'building_block_archive_enabled': policy in ('observe', 'random_bb', 'bb_snapshot', 'bb_import_credit_control'), 'best_bb_archive_enabled': policy in ('observe', 'best_bb'), 'bb_imported_credit': policy == 'bb_import_credit_control'}
