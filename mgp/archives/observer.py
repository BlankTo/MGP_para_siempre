import hashlib
import json
import math
from collections import Counter
from functools import cmp_to_key

import numpy as np

from mgp.archives.diagnostics import ProbeEvaluator, make_probe_dataset
from mgp.archives.diagnostics import archive_quality
from mgp.archives.records import ObjectRecords
from mgp.engine.expression_graph import archive_expression, graph_digest
from mgp.engine.fitness import object_fitness
from mgp.engine.fitness import object_fitness, rmse
from mgp.engine.objects import IncompleteObject, OperatorObject, SemanticList
from mgp.engine.objects import IncompleteObject, OperatorObject, SemanticList, SemanticObject
from mgp.engine.operator_sequence import operator_digest
from mgp.engine.symbolic import archive_rank_compare, expression_key, recovery_status, sorted_guard_tokens, symbolic_data
from mgp.engine.symbolic import symbolic_data
from mgp.engine.targets import TARGETS
from mgp.engine.targets import coverage_measures
from mgp.engine.targets import lexical_component_ids, make_subtargets


def semantic_nodes(node):
    if isinstance(node, SemanticList):
        return list(node.alternatives)
    if isinstance(node, IncompleteObject):
        return semantic_nodes(node.semantic)
    return [node] if isinstance(node, SemanticObject) else []


def imported_keys(node):
    found, seen, pending = (set(), set(), [node])
    while pending:
        current = pending.pop()
        if current.semantic_identity in seen:
            continue
        seen.add(current.semantic_identity)
        if current.archive_origin:
            found.add(symbolic_data(current)[0])
        pending.extend(current.evaluation_children)
    return found


def object_descriptor(node, records=None):
    head = node if isinstance(node, OperatorObject) else node.head if isinstance(node, IncompleteObject) else None
    if head is not None and records is not None:
        operators, expression = records.operator_payload(head)
        if node is not head:
            expression = node.expression
    else:
        operators = list(head.operators) if head is not None else []
        expression = node.expression
    return {'category': node.category, 'expression': expression, 'semantic_keys': [symbolic_data(n)[0] for n in semantic_nodes(node)], 'operators': operators, 'imported': bool(getattr(node, 'archive_origin', False))}


def construction_summary(best):
    out = {'best_construction_merges': None, 'best_inherited_units': None, 'best_inherited_operators': None, 'best_atoms': None}
    if best is None:
        return out
    seen, pending = (set(), [best])
    merges = inherited = inherited_operators = atoms = 0
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if getattr(node, 'archive_origin', False):
            inherited += 1
            inherited_operators += expression_size(node.evaluation)['expression_operators']
            continue
        children = getattr(node, 'evaluation_children', ()) or ()
        if not children:
            atoms += 1
            continue
        merges += 1
        pending.extend(children)
    return {'best_construction_merges': merges, 'best_inherited_units': inherited, 'best_inherited_operators': inherited_operators, 'best_atoms': atoms}


def expression_size(evaluation):
    sizes, pending = ({}, [(evaluation, False)])
    while pending:
        node, ready = pending.pop()
        token = id(node)
        if token in sizes:
            continue
        if isinstance(node, str):
            sizes[token] = (0, 1, 0)
        elif ready:
            children = [sizes[id(child)] for child in node[1:]]
            sizes[token] = (1 + sum((c[0] for c in children)), sum((c[1] for c in children)), 1 + max((c[2] for c in children)))
        else:
            pending.append((node, True))
            pending.extend(((child, False) for child in node[1:]))
    operators, leaves, depth = sizes[id(evaluation)]
    return {'expression_operators': operators, 'expression_leaves': leaves, 'expression_depth': depth}


def selected_measurement(best, labels, clean, test, settings):
    result = {'selected_expression': None, 'selected_key': None, 'train_rmse': None, 'clean_rmse': None, 'test_rmse': None, 'test_nrmse': None, 'test_success': 0, 'symbolic_recovery': None, 'domain_guards': [], 'origin_invalid': None, 'probe_invalid': None, 'test_invalid': None}
    if best is None:
        return result
    values, _ = ProbeEvaluator(test[0]).evaluate(best.evaluation)
    error = rmse(values, test[1])
    scale = rmse(np.zeros_like(test[1]), test[1]) or 1.0
    data = symbolic_data(best)
    try:
        status = recovery_status(best, TARGETS[settings['target']][1])
    except RecursionError:
        status = None
    probes = make_probe_dataset(TARGETS[settings['target']][0], list(test[0]), {**settings, 'probe_axis_points': 5, 'probe_near_zero': 1e-08})
    probe_values, _ = ProbeEvaluator(probes[0]).evaluate(best.evaluation)
    best_text = archive_expression(best, defer_graph=True)[0]
    result.update(selected_expression=best_text, selected_structure_digest=graph_digest(best.evaluation)[:32] if best_text is None else None, selected_key=data[0], train_rmse=object_fitness(best, labels), clean_rmse=object_fitness(best, clean), test_rmse=error, test_nrmse=error / scale, test_success=int(error <= settings['perfect_fit_tolerance']), symbolic_recovery=None if status is None else int(status), domain_guards=sorted_guard_tokens(data[3]), test_invalid=int(np.sum(~np.isfinite(values))), probe_invalid=int(np.sum(~np.isfinite(probe_values))), origin_invalid=int(np.any(~np.isfinite(probe_values[probes[3]['origin']]))), **expression_size(best.evaluation))
    return result


def sort_campaign_archives(entries, compact_ties=False):
    if compact_ties:
        return sorted(entries, key=lambda entry: (-entry[1]['score'], symbolic_data(entry[0])[0]))
    return sorted(entries, key=cmp_to_key(archive_rank_compare))


class DerivationObserver:

    def __init__(self, settings, variables, labels, clean):
        self.settings, self.labels, self.clean = (settings, labels, clean)
        self.expected = {expression_key(expression): identifier for identifier, expression, _ in make_subtargets(settings['target'], variables, 2)}
        self.identifiers = set(self.expected.values())
        self.lexical = lexical_component_ids(settings['target'])
        self.target_key = expression_key(TARGETS[settings['target']][1])
        self.trace, self.events, self.candidates = ([], [], {})
        self.evaluated, self.retained, self.retained_tokens = (set(), set(), set())
        self.used_imports, self.best_imports = (set(), set())
        self.merged_imports, self.retained_with_imports = (set(), 0)
        self.pending = []
        self.step = 0
        self.first = {'evaluated': {}, 'retained': {}, 'workspace': {}, 'archive': {}}
        self.last_workspace = set()
        self.workspace_roots = []
        self.workspace_key_counts = Counter()
        self.root_keys = {}
        self.best_fitness = math.inf
        self.last_improvement = 0
        self.improvements = 0
        self.transitions = Counter()
        self.initial_workspace = []
        self.archive_metrics = []
        self.records = ObjectRecords(labels, clean)
        self.initial_refs = []
        self.workspace_checkpoints = []
        self.candidate_events, self.choices = ([], [])
        self.pending_events = []
        self.has_imports = False

    def on_choice(self, candidates, weights, selected):
        self.choices.append({'merge': self.step + 1, 'candidates': [self.records.ref(node) for node in candidates], 'multiplicities': list(weights) if weights is not None else [1] * len(candidates), 'selected_ref': self.records.ref(selected)})

    def on_candidate(self, node):
        event = {'merge': self.step + 1, 'ordinal': len(self.candidate_events) + 1, 'node_ref': self.records.ref(node), 'committed_merge': False, 'admitted_after_merge': False}
        self.candidate_events.append(event)
        self.pending_events.append((node.semantic_identity, event))
        key = symbolic_data(node)[0]
        fitness = object_fitness(node, self.labels)
        entry = self.candidates.get(key)
        if entry is None:
            entry = {'key': key, 'expression': node.expression,
                     'first_merge': self.step + 1, 'evaluations': 0,
                     'ever_retained': False, 'train_rmse': fitness,
                     'invalid_samples': int(np.sum(~np.isfinite(node.values)))}
            self.candidates[key] = entry
        entry['evaluations'] += 1
        for child in getattr(node, 'evaluation_children', ()) or ():
            if getattr(child, 'archive_origin', False):
                self.merged_imports.add(symbolic_data(child)[0])
        self.pending.append((node.semantic_identity, key, fitness))
        self.evaluated.add(key)
        if key in self.expected:
            self.first['evaluated'].setdefault(self.expected[key], self.step + 1)

    def workspace_root_keys(self, root):
        keys = self.root_keys.get(root)
        if keys is None:
            keys = frozenset((symbolic_data(node)[0] for node in semantic_nodes(root)))
            self.root_keys[root] = keys
        return keys

    def add_workspace_root(self, root, delta):
        for key in self.workspace_root_keys(root):
            count = self.workspace_key_counts[key] + delta
            if count < 0:
                raise ValueError('observer workspace keys are out of sync')
            if count:
                self.workspace_key_counts[key] = count
            else:
                self.workspace_key_counts.pop(key, None)

    def coverage(self, keys):
        return coverage_measures({self.expected[k] for k in keys if k in self.expected}, self.identifiers, self.lexical)

    def observe(self, derivation):
        step = derivation.merge_count
        if step == 0:
            self.initial_workspace = [object_descriptor(node, self.records) for node in derivation.workspace]
            self.initial_refs = [self.records.ref(node) for node in derivation.workspace]
            self.has_imports = any((imported_keys(s) for root in derivation.workspace for s in semantic_nodes(root)))
        if step == 0:
            self.workspace_roots = list(derivation.workspace)
            for root in self.workspace_roots:
                self.add_workspace_root(root, 1)
        else:
            for position in sorted(derivation.last_selection['removed_indices'], reverse=True):
                root = self.workspace_roots.pop(position)
                self.add_workspace_root(root, -1)
            result = derivation.last_merge[2]
            self.workspace_roots.append(result)
            self.add_workspace_root(result, 1)
        workspace_keys = set(self.workspace_key_counts)
        roots = derivation.workspace if step == 0 else [derivation.last_merge[2]]
        outputs = [s for root in roots for s in semantic_nodes(root)]
        new = set()
        admitted_tokens = set()
        for node in outputs:
            key = symbolic_data(node)[0]
            admitted_tokens.add(node.semantic_identity)
            if key not in self.retained:
                new.add(key)
            self.retained.add(key)
            if key in self.candidates:
                self.candidates[key]['ever_retained'] = True
            if not node.archive_origin and node.semantic_identity not in self.retained_tokens:
                uses = imported_keys(node)
                self.used_imports.update(uses)
                self.retained_with_imports += bool(uses)
                for used in sorted(uses):
                    self.events.append({'merge': step, 'event': 'import_used', 'key': used, 'result_key': key})
            self.retained_tokens.add(node.semantic_identity)
        archive_keys = (set(derivation.archive.symbolic_keys) if derivation.archive.symbolic
                        else {symbolic_data(node)[0] for node in derivation.archive.objects})
        bb = derivation.building_block_archive
        if bb is not None:
            archive_keys.update(bb.keys)
        for label, keys in (('retained', self.retained), ('workspace', workspace_keys), ('archive', archive_keys)):
            for key in keys & self.expected.keys():
                self.first[label].setdefault(self.expected[key], step)
        for key in (workspace_keys ^ self.last_workspace) & self.expected.keys():
            self.events.append({'merge': step, 'event': 'component_present' if key in workspace_keys else 'component_lost', 'key': key, 'subtarget_id': self.expected[key]})
        improved = derivation.best_fitness < self.best_fitness
        if improved:
            if step:
                self.improvements += 1
            self.last_improvement = step
            self.best_fitness = derivation.best_fitness
            if derivation.best is not None:
                self.best_imports = imported_keys(derivation.best)
                self.events.append({'merge': step, 'event': 'best_improved', 'expression': derivation.best.expression, 'train_rmse': self.best_fitness, 'import_keys': sorted(self.best_imports)})
        record = {**derivation.metric_steps[-1], 'merge': step, 'best_ref': self.records.ref(derivation.best), **derivation.last_selection, 'ic_decisions': derivation.external_merge.ic_decisions, 'ms_decisions': derivation.external_merge.ms_decisions, 'replay_pairs': derivation.external_merge.replay_pairs, 'replay_operator_evaluations': derivation.external_merge.replay_operator_evaluations, 'replay_truncated_merges': derivation.external_merge.replay_truncated_merges, 'replay_limits': sorted(derivation.external_merge.current_replay_limits), 'replay_expanded_merges': derivation.external_merge.replay_expanded_merges, 'workspace_size': len(derivation.workspace), **derivation.current_counts, 'evaluated_unique': len(self.evaluated), 'retained_unique': len(self.retained), 'new_retained_unique': len(new), 'evaluations_this_merge': len(self.pending), 'evaluations_admitted': sum((token in admitted_tokens for token, _, _ in self.pending)), 'candidate_best_train_rmse': min((fit for _, _, fit in self.pending), default=None), 'admitted_best_train_rmse': min((object_fitness(n, self.labels) for n in outputs), default=None), 'best_improved': int(improved and step > 0), 'stagnation_merges': step - self.last_improvement, 'workspace_target': int(self.target_key in workspace_keys), 'retained_target_ever': int(self.target_key in self.retained), 'evaluated_target_ever': int(self.target_key in self.evaluated), 'used_imports_unique': len(self.used_imports), 'best_imports_unique': len(self.best_imports), 'selected_clean_rmse': object_fitness(derivation.best, self.clean) if derivation.best else None, 'selected_symbolic_recovery': int(derivation.best is not None and symbolic_data(derivation.best)[0] == self.target_key)}
        record['oracle_best_clean_rmse'] = record.pop('best_clean_rmse')
        for label, keys in (('workspace', workspace_keys), ('retained', self.retained), ('archive', archive_keys)):
            record.update({label + '_' + k: v for k, v in self.coverage(keys).items()})
        if derivation.last_merge is not None:
            left, right, result = derivation.last_merge
            record['transition'] = f'{left.category}+{right.category}->{result.category}'
            record.update(left=object_descriptor(left, self.records), right=object_descriptor(right, self.records), result=object_descriptor(result, self.records), left_ref=self.records.ref(left), right_ref=self.records.ref(right), result_ref=self.records.ref(result))
            self.transitions[record['transition']] += 1
        else:
            record['transition'] = 'initial'
        self.trace.append(record)
        interval = self.settings.get('workspace_checkpoint_interval', 64)
        if type(interval) is not int or interval < 1:
            raise ValueError('workspace_checkpoint_interval must be a positive integer')
        if step % interval == 0:
            self.workspace_checkpoints.append([step, [self.records.ref(n) for n in derivation.workspace]])
        for token, event in self.pending_events:
            event.update(committed_merge=True, admitted_after_merge=token in admitted_tokens)
        self.pending_events.clear()
        self.pending.clear()
        self.last_workspace = workspace_keys
        self.step = step

    def archive_rows(self, derivation, detail_limit=None, aggregate_only=False, identity_sink=None):
        if aggregate_only and not self.settings.get('count_archive_diagnostics', False):
            self.archive_metrics = []
            if identity_sink is not None:
                archives = {'base_fitness': derivation.archive.objects}
                similarity = (getattr(derivation, 'base_archives', {}).get('similarity')
                              if getattr(self, 'include_similarity_archive', False) else None)
                if similarity is not None and similarity is not derivation.archive:
                    archives['base_similarity'] = similarity.objects
                for name, archive in (('bb_snapshot', derivation.building_block_archive),
                                      ('best_bb', derivation.best_bb_archive)):
                    if archive is not None:
                        archives[name] = archive.representatives
                for name, nodes in archives.items():
                    identities = {hashlib.blake2b(symbolic_data(node)[0].encode('utf-8'),
                                                 digest_size=16).digest() for node in nodes}
                    identity_sink(name, identities)
                if derivation.best is not None:
                    identity_sink('selected_best', {
                        hashlib.blake2b(symbolic_data(derivation.best)[0].encode('utf-8'),
                                        digest_size=16).digest()})
            return []
        train_errors = {}
        def train_error(node):
            token = id(node)
            if token not in train_errors:
                train_errors[token] = (node, object_fitness(node, self.labels))
            return train_errors[token][1]

        clean_errors = {}
        evaluation_sizes = {}
        identity_hashes = {}
        def clean_error(node):
            token = id(node)
            if token not in clean_errors:
                clean_errors[token] = (node, object_fitness(node, self.clean))
            return clean_errors[token][1]

        def cached_size(evaluation):
            token = id(evaluation)
            if token not in evaluation_sizes:
                evaluation_sizes[token] = (evaluation, expression_size(evaluation))
            return evaluation_sizes[token][1]

        def identity_hash(key):
            if key not in identity_hashes:
                identity_hashes[key] = hashlib.blake2b(key.encode('utf-8'), digest_size=16).digest()
            return identity_hashes[key]

        archives = {'base_fitness': [(node, {'score': -train_error(node)}) for node in derivation.archive.objects]}
        similarity = getattr(derivation, 'base_archives', {}).get('similarity') if getattr(self, 'include_similarity_archive', False) else None
        if similarity is not None and similarity is not derivation.archive:
            archives['base_similarity'] = [(node, {'score': -train_error(node)}) for node in similarity.objects]
        for name, archive in (('bb_snapshot', derivation.building_block_archive), ('best_bb', derivation.best_bb_archive)):
            if archive is not None:
                archives[name] = list(zip(archive.representatives, archive.entries))
        result = []
        self.archive_metrics = []
        finite = np.isfinite(self.clean)
        signal_rms = rmse(np.zeros(int(finite.sum())), self.clean[finite])
        for name, entries in archives.items():
            ordered = sort_campaign_archives(entries, self.settings.get('campaign_compact_tie_break', False))
            member_rows, values = ([], [])
            identities = set() if identity_sink is not None else None
            sampled_ranks = set()
            if not aggregate_only and detail_limit is not None and len(ordered) > detail_limit:
                import random
                seed = int.from_bytes(hashlib.sha256(
                    f'{self.settings.get("target")}:{name}:{len(ordered)}'.encode()).digest()[:8], 'big')
                local_rng = random.Random(seed)
                sampled_ranks = set(local_rng.sample(range(detail_limit + 1, len(ordered) + 1),
                                                     min(16, len(ordered) - detail_limit)))
            for rank, (node, row) in enumerate(ordered, 1):
                data = symbolic_data(node)
                if identities is not None:
                    identities.add(identity_hash(data[0]))
                detailed = not aggregate_only and (detail_limit is None or rank <= detail_limit or rank in sampled_ranks)
                if aggregate_only:
                    member = {'key': data[0], 'clean_rmse': clean_error(node),
                              **cached_size(node.evaluation)}
                else:
                    member = {'archive': name, 'key': data[0], 'rank': rank,
                              'matched_id': self.expected.get(data[0]),
                              'train_rmse': train_error(node),
                              'clean_rmse': clean_error(node),
                              'score': row['score'], **cached_size(node.evaluation)}
                if detailed:
                    expression, expression_graph = archive_expression(node, defer_graph=getattr(self, 'defer_archive_graphs', False))
                    member.update({**{k: v for k, v in row.items() if k != 'values_json'},
                                   'expression': expression, 'expression_graph': expression_graph,
                                   'node_ref': self.records.ref(node), 'canonical': data[1],
                                   'individual_credit': row.get('individual_credit'),
                                   'inherited_credit': row.get('inherited_credit'),
                                   'credit_by_depth': row.get('credit_by_depth', {}),
                                   'occurrences': row.get('occurrences'),
                                   **cached_size(node.evaluation),
                                   'domain_guards': sorted_guard_tokens(data[3])})
                    if self.has_imports:
                        member.update(inherited_self=int(bool(getattr(node, 'archive_origin', False))),
                                      inherited_used=len(imported_keys(node)))
                if not aggregate_only:
                    result.append(member)
                member_rows.append(member)
                values.append(node.values[finite])
            if identities is not None:
                identity_sink(name, identities)
            if self.settings.get('count_archive_diagnostics', False):
                if len(member_rows) <= 20:
                    keys = {row['key'] for row in member_rows}
                    shared = {'archive': name, 'entries': len(member_rows),
                              **self.coverage(keys),
                              **archive_quality(member_rows, values, signal_rms)}
                    self.archive_metrics.append({'top': 20, **shared})
                    self.archive_metrics.append({'top': 'all', **shared})
                else:
                    for top in (20, 'all'):
                        selected_rows = member_rows if top == 'all' else member_rows[:top]
                        selected_values = values if top == 'all' else values[:top]
                        self.archive_metrics.append({'archive': name, 'top': top,
                            'entries': len(selected_rows),
                            **self.coverage({row['key'] for row in selected_rows}),
                            **archive_quality(selected_rows, selected_values, signal_rms)})
        best = derivation.best
        if best is not None and not aggregate_only:
            data = symbolic_data(best)
            best_expression, best_graph = archive_expression(best, defer_graph=getattr(self, 'defer_archive_graphs', False))
            if not any((row['key'] == data[0] and row.get('expression') == best_expression and (row.get('expression_graph') is best_graph or (isinstance(best_graph, dict) and row.get('expression_graph') == best_graph)) for row in result)):
                result.append({'archive': 'selected_best', 'key': data[0], 'expression': best_expression, 'expression_graph': best_graph, 'node_ref': self.records.ref(best), 'rank': 1, 'matched_id': self.expected.get(data[0]), 'score': -object_fitness(best, self.labels), 'train_rmse': object_fitness(best, self.labels), 'clean_rmse': object_fitness(best, self.clean), **expression_size(best.evaluation), 'domain_guards': sorted_guard_tokens(data[3])})
        if identity_sink is not None and derivation.best is not None:
            identity_sink('selected_best', {identity_hash(symbolic_data(derivation.best)[0])})
        return result


class _NoRecordGraph:

    @staticmethod
    def ref(node):
        return None


class ArchiveOnlyObserver(DerivationObserver):

    def __init__(self, settings, variables, labels, clean):
        super().__init__(settings, variables, labels, clean)
        self.records = _NoRecordGraph()
        self.defer_archive_graphs = True
        self.trajectory_digest = hashlib.sha256()
        self._last_coverage_sources = None
        self._eval_digest_cache = {}
        self._selected_cache_node = None
        self._selected_cache_clean = None
        self._selected_cache_symbolic = 0

    def node_digest(self, node):
        if isinstance(node, IncompleteObject):
            return hashlib.sha256(('incomplete:' + self.node_digest(node.semantic) + ':' + operator_digest(node.head.operators)).encode()).hexdigest()
        if isinstance(node, OperatorObject):
            return hashlib.sha256(('operator:' + operator_digest(node.operators)).encode()).hexdigest()
        if isinstance(node, SemanticList):
            parts = [(self.node_digest(item), count) for item, count in zip(node.alternatives, node.multiplicities)]
            return hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).hexdigest()
        return self.graph_digest_cached(node.evaluation)

    def graph_digest_cached(self, root):
        cache = self._eval_digest_cache
        pending = [(root, False)]
        while pending:
            current, ready = pending.pop()
            marker = id(current)
            if marker in cache:
                continue
            if isinstance(current, str):
                encoded = current.encode('utf-8')
                cache[marker] = (current, hashlib.sha256(b'a' + len(encoded).to_bytes(8, 'big') + encoded).digest())
            elif ready:
                symbol = current[0].encode('utf-8')
                payload = b'o' + len(symbol).to_bytes(2, 'big') + symbol + b''.join(cache[id(child)][1] for child in current[1:])
                cache[marker] = (current, hashlib.sha256(payload).digest())
            else:
                pending.append((current, True))
                pending.extend((child, False) for child in reversed(current[1:]) if id(child) not in cache)
        return cache[id(root)][1].hex()

    def on_choice(self, candidates, weights, selected):
        if selected is not None:
            self.trajectory_digest.update(b'C' + bytes.fromhex(self.node_digest(selected)))

    def on_candidate(self, node):
        key = symbolic_data(node)[0]
        self.pending.append((node.semantic_identity, key, object_fitness(node, self.labels)))
        self.evaluated.add(key)
        for child in getattr(node, 'evaluation_children', ()) or ():
            if getattr(child, 'archive_origin', False):
                self.merged_imports.add(symbolic_data(child)[0])
        if key in self.expected:
            self.first['evaluated'].setdefault(self.expected[key], self.step + 1)

    def observe(self, derivation):
        step = derivation.merge_count
        if step == 0:
            self.workspace_roots = list(derivation.workspace)
            for root in self.workspace_roots:
                self.add_workspace_root(root, 1)
            self.has_imports = any((imported_keys(s) for root in derivation.workspace for s in semantic_nodes(root)))
        else:
            for position in sorted(derivation.last_selection['removed_indices'], reverse=True):
                root = self.workspace_roots.pop(position)
                self.add_workspace_root(root, -1)
            result = derivation.last_merge[2]
            self.workspace_roots.append(result)
            self.add_workspace_root(result, 1)
        workspace_keys = self.workspace_key_counts.keys()
        roots = derivation.workspace if step == 0 else [derivation.last_merge[2]]
        outputs = [s for root in roots for s in semantic_nodes(root)]
        admitted = {node.semantic_identity for node in outputs}
        new = set()
        for node in outputs:
            key = symbolic_data(node)[0]
            if key not in self.retained:
                new.add(key)
            self.retained.add(key)
            if not node.archive_origin and node.semantic_identity not in self.retained_tokens:
                uses = imported_keys(node) if self.has_imports else set()
                self.used_imports.update(uses)
                self.retained_with_imports += bool(uses)
            self.retained_tokens.add(node.semantic_identity)
        coverage_due = self.needs_coverage(step)
        archive = derivation.archive
        bb = derivation.building_block_archive
        base_keys = archive.symbolic_keys if archive.symbolic else None
        if coverage_due:
            archive_keys = (set(base_keys) if base_keys is not None
                            else {symbolic_data(node)[0] for node in archive.objects})
            if bb is not None:
                archive_keys.update(bb.keys)
        else:
            archive_keys = None
        expected = self.expected
        for label, keys in (('retained', self.retained), ('workspace', workspace_keys)):
            for key in expected.keys() & keys:
                self.first[label].setdefault(expected[key], step)
        if archive_keys is not None:
            archive_expected = expected.keys() & archive_keys
        elif base_keys is not None:
            archive_expected = (key for key in expected if key in base_keys or (bb is not None and key in bb.keys))
        else:
            archive_expected = (key for key in expected if
                                any(symbolic_data(node)[0] == key for node in archive.objects)
                                or (bb is not None and key in bb.keys))
        for key in archive_expected:
            self.first['archive'].setdefault(expected[key], step)
        improved = derivation.best_fitness < self.best_fitness
        if improved:
            self.improvements += int(step > 0)
            self.last_improvement = step
            self.best_fitness = derivation.best_fitness
            if derivation.best is not None:
                self.best_imports = imported_keys(derivation.best) if self.has_imports else set()
        selected = derivation.best
        if selected is not self._selected_cache_node:
            self._selected_cache_clean = object_fitness(selected, self.clean) if selected is not None else None
            self._selected_cache_symbolic = int(symbolic_data(selected)[0] == self.target_key) if selected is not None else 0
            self._selected_cache_node = selected
        record = {**derivation.metric_steps[-1], 'merge': step, 'ic_decisions': derivation.external_merge.ic_decisions, 'ms_decisions': derivation.external_merge.ms_decisions, 'replay_pairs': derivation.external_merge.replay_pairs, 'replay_operator_evaluations': derivation.external_merge.replay_operator_evaluations, 'replay_truncated_merges': derivation.external_merge.replay_truncated_merges, 'replay_limits': sorted(derivation.external_merge.current_replay_limits), 'replay_expanded_merges': derivation.external_merge.replay_expanded_merges, 'workspace_size': len(derivation.workspace), **derivation.current_counts, 'evaluated_unique': len(self.evaluated), 'retained_unique': len(self.retained), 'new_retained_unique': len(new), 'evaluations_this_merge': len(self.pending), 'evaluations_admitted': sum((token in admitted for token, _, _ in self.pending)), 'candidate_best_train_rmse': min((fit for _, _, fit in self.pending), default=None), 'admitted_best_train_rmse': min((object_fitness(n, self.labels) for n in outputs), default=None), 'best_improved': int(improved and step > 0), 'stagnation_merges': step - self.last_improvement, 'workspace_target': int(self.target_key in workspace_keys), 'retained_target_ever': int(self.target_key in self.retained), 'evaluated_target_ever': int(self.target_key in self.evaluated), 'used_imports_unique': len(self.used_imports), 'best_imports_unique': len(self.best_imports), 'selected_clean_rmse': self._selected_cache_clean, 'selected_symbolic_recovery': self._selected_cache_symbolic}
        record['oracle_best_clean_rmse'] = record.pop('best_clean_rmse')
        if coverage_due:
            self._last_coverage_sources = (set(workspace_keys), archive_keys)
            self.add_coverage(record)
        if derivation.last_merge is None:
            record['transition'] = 'initial'
        else:
            left, right, result = derivation.last_merge
            transition = f'{left.category}+{right.category}->{result.category}'
            record['transition'] = transition
            self.transitions[transition] += 1
            digest_part = [step, transition, derivation.last_merge_kind, derivation.last_selection.get('removed_indices')]
            self.trajectory_digest.update(json.dumps(digest_part, separators=(',', ':')).encode())
            for node in (left, right, result):
                self.trajectory_digest.update(bytes.fromhex(self.node_digest(node)))
        self.trace[:] = [record]
        self.pending.clear()
        self.step = step

    @staticmethod
    def needs_coverage(step):
        return step == 0 or (step & (step - 1)) == 0 or step % 250 == 0

    def add_coverage(self, record):
        workspace_keys, archive_keys = self._last_coverage_sources
        for label, keys in (('workspace', workspace_keys), ('retained', self.retained), ('archive', archive_keys)):
            record.update({label + '_' + k: value for k, value in self.coverage(keys).items()})

    def finalize_coverage(self, derivation=None):
        if not self.trace or 'workspace_component_coverage' in self.trace[-1]:
            return
        if derivation is None:
            raise ValueError('final archive coverage requires the final derivation')
        archive = derivation.archive
        archive_keys = (set(archive.symbolic_keys) if archive.symbolic
                        else {symbolic_data(node)[0] for node in archive.objects})
        if derivation.building_block_archive is not None:
            archive_keys.update(derivation.building_block_archive.keys)
        self._last_coverage_sources = (set(self.workspace_key_counts), archive_keys)
        self.add_coverage(self.trace[-1])
