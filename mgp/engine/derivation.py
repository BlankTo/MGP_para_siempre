import random
import time

import numpy as np

from mgp.archives.archive import Archive
from mgp.archives.best_building_blocks import BestBuildingBlockArchive, validate_best_bb
from mgp.archives.building_blocks import BuildingBlockArchive
from mgp.archives.observations import SymbolicCounts, WorkspaceMetrics, graph_occurrences
from mgp.engine.fitness import best_index, object_fitness, readonly_target
from mgp.engine.merge import ExternalMerge, InternalMerge
from mgp.engine.merge_strategies import constructive_merges, legal_partners
from mgp.engine.objects import MATRIX_COLUMNS, SemanticList, SemanticObject, object_counts
from mgp.engine.protocol import current_protocol
from mgp.engine.semantics import symbolic_enabled, select_comparison


class Derivation:

    def __init__(self, target, config, seed, clean_target=None, candidate_observer=None, choice_observer=None):
        self.target = readonly_target(target)
        self.clean_target = self.target if clean_target is None else readonly_target(clean_target)
        if self.clean_target.shape != self.target.shape:
            raise ValueError('clean and training targets must have the same shape')
        if self.clean_target is not self.target and np.array_equal(self.clean_target, self.target, equal_nan=True):
            self.clean_target = self.target
        for key in ('esel', 'esur', 'fitness_probability', 'internal_merge_probability'):
            if not 0 <= config[key] <= 1:
                raise ValueError(key + ' must be between 0 and 1')
        if not isinstance(config['max_merge'], int) or config['max_merge'] < 0:
            raise ValueError('max_merge must be a nonnegative integer')
        if not np.isfinite(config['perfect_fit_tolerance']) or config['perfect_fit_tolerance'] < 0:
            raise ValueError('perfect_fit_tolerance must be finite and nonnegative')
        validate_best_bb(config)
        self.config = config.copy()
        symbolic_enabled(config)
        self.seed = seed
        self.candidate_observer = candidate_observer
        self.choice_observer = choice_observer
        self.merge_strategy = config.get('merge_strategy', 'standard')
        if self.merge_strategy not in ('standard', 'revisit_ic', 'constructive'):
            raise ValueError('unknown merge_strategy')
        if self.merge_strategy == 'revisit_ic' and config.get('semantic_list_enabled'):
            raise ValueError('revisit_ic requires semantic_list_enabled=False')
        for name in ('revisit_ic_variants', 'revisit_ic_pairs', 'revisit_ic_budget'):
            v = config.get(name)
            if v is not None and (type(v) is not int or v < 1):
                raise ValueError(name + ' must be a positive integer or None')
        for name in ('esur_in_IM', 'IM_terminate_on_one', 'semantic_list_enabled', 'building_block_archive_enabled', 'archive_constants', 'complete_into_hole'):
            if not isinstance(config.get(name, name == 'IM_terminate_on_one'), bool):
                raise ValueError(name + ' must be a boolean')
        spec = config.get('constructive_merges')
        if spec is None and self.merge_strategy == 'constructive':
            raise ValueError('constructive_merges must list the merges allowed to the constructive strategy')
        self.constructive_merges = frozenset(constructive_merges(spec or ()))

    def consider(self, current, source):
        if not isinstance(current, SemanticObject):
            return
        if isinstance(current, SemanticList):
            for alternative in current.alternatives:
                self.consider(alternative, source)
            return
        fitness = object_fitness(current, self.target)
        clean_fitness = fitness if self.clean_target is self.target else object_fitness(current, self.clean_target)
        if self.clean_best is None or clean_fitness < self.clean_best_fitness:
            self.clean_best = current
            self.clean_best_fitness = clean_fitness
        if self.best is None or fitness < self.best_fitness:
            self.best = current
            self.best_fitness = fitness
            self.best_source = source

    def run(self, workspace, archive=None, input_so=None, verbose=False):
        for current_workspace in self.steps(workspace, archive, input_so, verbose):
            pass
        return self.best

    def steps(self, workspace, archive=None, input_so=None, verbose=False):
        self.workspace = list(workspace)
        self._constructive_workspace = None
        self._constructive_partners = None
        self.metric_steps = []
        self.started = time.perf_counter()
        self.started_cpu = time.process_time()
        atoms = [node for root in self.workspace for node, count, mass, depth in graph_occurrences(root) if isinstance(node, SemanticObject) and node.is_atomic]
        self.symbolic_counts = SymbolicCounts(self.config, atoms)
        self.best = None
        self.best_fitness = float('inf')
        self.best_source = ''
        self.clean_best_fitness = float('inf')
        self.clean_best = None
        self.building_block_archive = None
        self.best_bb_archive = None
        self.best_bb_seconds = 0.0
        self.best_bb_cpu_seconds = 0.0
        self.best_bb_finalized = False
        self.archive_observation_seconds = 0.0
        self.metrics_observation_seconds = 0.0
        self.symbolic_observation_seconds = 0.0
        self.archive_observation_cpu_seconds = 0.0
        self.metrics_observation_cpu_seconds = 0.0
        self.symbolic_observation_cpu_seconds = 0.0
        if self.config.get('building_block_archive_enabled', False):
            self.building_block_archive = BuildingBlockArchive(self.target, self.config)
        self.archive_mask = {'base_fitness': True, 'bb_snapshot': True, 'best_bb': True, **(self.config.get('observational_archive_mask') or {})}
        self.merge_count = 0
        self.attempt_count = 0
        self.internal_candidates_inspected = 0
        self.base_archive_insertions = 0
        self.base_archive_rejections = 0
        self.step_counts = []
        self.initial_counts = object_counts(self.workspace)
        self.current_counts = {}
        self.last_merge = None
        self.last_merge_kind = None
        self.last_selection = {}
        self.alternative_limit_reason = ''
        self.internal_merge_count = 0
        self.external_merge_count = 0
        self.internal_fallback_count = 0
        for index, column in enumerate(MATRIX_COLUMNS):
            self.current_counts[column] = self.initial_counts[index]
        self.workspace_metrics = WorkspaceMetrics(self.workspace, self.symbolic_counts)
        if archive is None:
            archive = Archive(self.config.get('single_archive_max_size'), self.config['archive_similarity'], self.config['archive_atomic_objects'], symbolic=symbolic_enabled(self.config),
                              constants=self.config.get('archive_constants', False))
        self.archive = archive
        self.base_archives = {'symbolic' if archive.symbolic else 'similarity': archive}
        if current_protocol(self.config):
            if archive.objects:
                raise ValueError('the paired observational archives require an empty input archive')
            self.base_archives = {}
            method = self.config.get('archive_comparison_method', 'cosine')
            for policy in self.config.get('base_archive_policies', ['similarity', 'symbolic']):
                self.base_archives[policy] = Archive(self.config.get('single_archive_max_size'), self.config['archive_similarity'], self.config['archive_atomic_objects'], comparison=None if method == 'cosine' else select_comparison(method, self.config.get('archive_comparison_parameters', {})), comparison_is_distance=method == 'rmse', symbolic=policy == 'symbolic', target=self.target, constants=self.config.get('archive_constants', False))
            self.archive = self.base_archives['symbolic']
        self.base_archive_seconds = dict.fromkeys(self.base_archives, 0.0)
        self.base_archive_cpu_seconds = dict.fromkeys(self.base_archives, 0.0)
        self.last_archive_candidates = []
        generator = random.Random(self.seed)
        merge_seed = generator.getrandbits(63)
        external_merge = ExternalMerge(self.target, self.config['fitness_probability'], merge_seed, self.config.get('semantic_list_enabled', False))
        self.external_merge = external_merge
        external_merge.candidate_observer = self.candidate_observer
        external_merge.choice_observer = self.choice_observer
        external_merge.strategy = self.merge_strategy
        external_merge.constructive_merges = self.constructive_merges
        external_merge.complete_into_hole = self.config.get('complete_into_hole', False)
        if self.merge_strategy == 'revisit_ic':
            for name, key, default in (('alternative_evaluation_limit', 'revisit_ic_budget', 512), ('alternative_variant_limit', 'revisit_ic_variants', 64),
                                       ('alternative_pair_limit', 'revisit_ic_pairs', 512)):
                setattr(external_merge, name, self.config.get(key, default))
        internal_merge = InternalMerge(external_merge)
        self.internal_merge = internal_merge
        internal_probability = self.config['internal_merge_probability']
        candidates = list(self.workspace) + list(archive.objects)
        if input_so is not None:
            if not isinstance(input_so, SemanticObject):
                raise TypeError('input_so must be semantic or None')
            candidates.append(input_so)
        for current in candidates:
            semantic = current
            if current.kind == 'incomplete':
                semantic = current.semantic
            if isinstance(semantic, SemanticObject) and semantic.values.shape != self.target.shape:
                raise ValueError('all objects must use the target dataset')
        self.consider(input_so, 'input_so')
        for current in self.workspace:
            self.observe_symbolic(current)
            self.consider(current, 'initial_workspace')
            if current_protocol(self.config) and self.config['archive_enabled'] and self.archive_mask['base_fitness']:
                self.observe_base_archives(current)
            self.observe_building_blocks(current)
        for current in archive.objects:
            self.consider(current, 'input_archive')
        self.update_termination()
        self.record_metrics()
        observer_paused_seconds = 0.0
        paused_at = time.perf_counter()
        yield self.workspace.copy()
        observer_paused_seconds += time.perf_counter() - paused_at
        while not self.termination_reason:
            wall_limit = self.config.get('research_wall_limit_seconds')
            if wall_limit is not None and time.perf_counter() - self.started - observer_paused_seconds >= wall_limit:
                self.termination_reason = 'time_limit'
                break
            self.last_archive_candidates = []
            self.attempt_count += 1
            elite_index = None
            if self.config['esel'] > 0 or self.config['esur'] > 0:
                elite_index = best_index(self.workspace, self.target)
            attempt_internal = False
            if internal_probability > 0:
                attempt_internal = len(self.workspace) == 1 or generator.random() < internal_probability
            partners = self.constructive_partners() if self.merge_strategy == 'constructive' else None
            candidate_counts = {}
            def count_candidates(parent):
                key = id(parent)
                if key not in candidate_counts:
                    candidate_counts[key] = (parent, internal_merge.candidate_count(parent))
                return candidate_counts[key][1]
            constructive_fallback = False
            external_fallback = False
            if partners is not None:
                internal_parents = [i for i, parent in enumerate(self.workspace) if count_candidates(parent)] if internal_probability > 0 and (attempt_internal or not partners) else []
                if attempt_internal and (not internal_parents):
                    self.internal_fallback_count += 1
                    constructive_fallback = True
                if not partners and internal_parents:
                    external_fallback = not attempt_internal
                    attempt_internal = True
                eligible = internal_parents if attempt_internal and internal_parents else list(partners)
                if elite_index in eligible and generator.random() < self.config['esel']:
                    left_index = elite_index
                else:
                    left_index = generator.choice(eligible)
            elif elite_index is not None and generator.random() < self.config['esel']:
                left_index = elite_index
            else:
                left_index = generator.randrange(len(self.workspace))
            internal_pair = None
            internal_candidate_count = None
            if attempt_internal and (not constructive_fallback):
                parent = self.workspace[left_index]
                internal_candidate_count = count_candidates(parent)
                self.internal_candidates_inspected += internal_candidate_count or 0
                if internal_candidate_count:
                    internal_pair = internal_merge.select(parent, generator)
                else:
                    self.internal_fallback_count += 1
            if internal_pair is not None:
                parent_index = left_index
                path, left = internal_pair
                right = self.workspace[parent_index]
                result = internal_merge.merge(right, path)
                removed_indices = [parent_index]
                if self.config.get('esur_in_IM', False) and elite_index == parent_index:
                    if generator.random() < self.config['esur']:
                        removed_indices = []
                self.last_merge_kind = 'internal'
                self.internal_merge_count += 1
            else:
                if partners is not None:
                    right_index = generator.choice(partners[left_index])
                else:
                    right_index = generator.randrange(len(self.workspace) - 1)
                    if right_index >= left_index:
                        right_index += 1
                left = self.workspace[left_index]
                right = self.workspace[right_index]
                result = external_merge.merge(left, right)
                removed_indices = [left_index, right_index]
                if elite_index in removed_indices and generator.random() < self.config['esur']:
                    removed_indices.remove(elite_index)
                self.last_merge_kind = 'external'
                self.external_merge_count += 1
            self.last_merge = (left, right, result)
            self.last_selection = {'merge_kind': self.last_merge_kind, 'attempted_internal': int(attempt_internal), 'internal_fallback': int(attempt_internal and internal_pair is None), 'external_fallback': int(external_fallback), 'internal_candidate_count': internal_candidate_count, 'internal_path': list(internal_pair[0]) if internal_pair is not None else None, 'left_index': left_index, 'right_index': left_index if internal_pair is not None else right_index, 'removed_indices': list(removed_indices), 'elite_available': int(elite_index is not None), 'left_was_elite': int(elite_index is not None and left_index == elite_index), 'elite_in_pair': int(elite_index is not None and (elite_index == left_index if internal_pair is not None else elite_index in (left_index, right_index))), 'elite_removed': int(elite_index is not None and elite_index in removed_indices), 'removed_objects': len(removed_indices)}
            for index in sorted(removed_indices, reverse=True):
                self.current_counts[self.workspace[index].category] -= 1
                self.workspace_metrics.remove(self.workspace[index])
                self.workspace.pop(index)
            self.workspace.append(result)
            self.workspace_metrics.add(result)
            self.current_counts[result.category] += 1
            self.merge_count += 1
            self.step_counts.append([self.current_counts[column] for column in MATRIX_COLUMNS])
            self.consider(result, f'merge_{self.merge_count}')
            self.observe_symbolic(result)
            if self.config['archive_enabled'] and self.archive_mask['base_fitness']:
                self.observe_base_archives(result)
            self.observe_building_blocks(result)
            self.update_termination()
            self.record_metrics()
            paused_at = time.perf_counter()
            yield self.workspace.copy()
            observer_paused_seconds += time.perf_counter() - paused_at
        self.finalize_best_archive()

    def finalize_best_archive(self):
        if self.best_bb_finalized or not self.termination_reason:
            return
        self.best_bb_finalized = True
        if self.config.get('best_bb_archive_enabled', False) and self.archive_mask['best_bb']:
            started = time.perf_counter()
            cpu_started = time.process_time()
            self.best_bb_archive = BestBuildingBlockArchive(self.target, self.config)
            self.best_bb_archive.build(self.best)
            if self.config.get('single_archive_max_size') is not None:
                self.best_bb_archive.limit(self.config['single_archive_max_size'])
            self.best_bb_seconds = time.perf_counter() - started
            self.best_bb_cpu_seconds = time.process_time() - cpu_started

    def observe_symbolic(self, node):
        started = time.perf_counter()
        cpu_started = time.process_time()
        self.symbolic_counts.observe(node)
        self.symbolic_observation_seconds += time.perf_counter() - started
        self.symbolic_observation_cpu_seconds += time.process_time() - cpu_started

    def record_metrics(self):
        started = time.perf_counter()
        cpu_started = time.process_time()
        metrics = self.workspace_metrics.measure()
        metrics.update({'archive_size': len(self.archive.objects), 'best_training_rmse': self.best_fitness if np.isfinite(self.best_fitness) else None, 'best_clean_rmse': self.clean_best_fitness if np.isfinite(self.clean_best_fitness) else None, 'target_recovered': int((self.best_fitness if current_protocol(self.config) else self.clean_best_fitness) <= self.config['perfect_fit_tolerance']), 'operator_evaluations': self.external_merge.operator_evaluations, 'ic_candidates': self.external_merge.ic_candidates, 'external_merge_count': self.external_merge_count, 'internal_merge_count': self.internal_merge_count, 'internal_fallback_count': self.internal_fallback_count, 'elapsed_seconds': time.perf_counter() - self.started, 'elapsed_cpu_seconds': time.process_time() - self.started_cpu, 'archive_observation_seconds': self.archive_observation_seconds})
        if self.building_block_archive is not None:
            entries = self.building_block_archive.entries
            metrics['building_block_archive_size'] = len(entries)
            metrics['building_block_score'] = self.building_block_archive.total_score
        self.metric_steps.append(metrics)
        self.metrics_observation_seconds += time.perf_counter() - started
        self.metrics_observation_cpu_seconds += time.process_time() - cpu_started

    def constructive_partners(self):
        workspace = tuple(self.workspace)
        if workspace != self._constructive_workspace:
            self._constructive_partners = legal_partners(workspace, self.constructive_merges)
            self._constructive_workspace = workspace
        return self._constructive_partners

    def update_termination(self):
        self.final_object = None
        self.termination_reason = ''
        if not self.workspace:
            self.termination_reason = 'empty_workspace'
        elif len(self.workspace) == 1 and (self.config.get('IM_terminate_on_one', True) or self.config['internal_merge_probability'] == 0):
            self.termination_reason = 'single_object'
            self.final_object = self.workspace[0]
        elif self.merge_count >= self.config['max_merge']:
            self.termination_reason = 'max_merge'
        elif self.merge_strategy == 'constructive' and (not self.constructive_partners()) and (not (self.config['internal_merge_probability'] > 0 and any((self.internal_merge.candidate_count(parent) for parent in self.workspace)))):
            self.termination_reason = 'no_legal_merge'
        elif len(self.workspace) == 1 and (not self.internal_merge.candidate_count(self.workspace[0])):
            self.termination_reason = 'no_valid_merge'
        if self.termination_reason and len(self.workspace) == 1:
            if isinstance(self.workspace[0], SemanticList):
                current = self.workspace[0]
                selected = self.external_merge.resolve_list(current)
                self.current_counts[current.category] -= 1
                self.current_counts[selected.category] += 1
                self.workspace_metrics.remove(current)
                self.workspace_metrics.add(selected)
                self.workspace[0] = selected
                if self.step_counts:
                    self.step_counts[-1] = [self.current_counts[name] for name in MATRIX_COLUMNS]
                else:
                    self.initial_counts = object_counts(self.workspace)
                self.consider(selected, 'terminal_resolution')
                if self.last_merge is not None:
                    self.last_merge = (*self.last_merge[:2], selected)
                if self.config['archive_enabled'] and self.archive_mask['base_fitness']:
                    self.observe_base_archives(selected)
                self.observe_building_blocks(selected)
                self.observe_symbolic(selected)
            self.final_object = self.workspace[0]

    def observe_base_archives(self, current):
        for policy, archive in self.base_archives.items():
            started = time.perf_counter()
            cpu_started = time.process_time()
            previous = set(archive.objects) if archive is self.archive else set()
            added = archive.add(current)
            if archive is self.archive:
                self.base_archive_insertions += int(bool(added))
                self.base_archive_rejections += int(not added)
            if added and archive is self.archive:
                self.last_archive_candidates.extend((node for node in archive.objects if node not in previous))
            self.base_archive_seconds[policy] += time.perf_counter() - started
            self.base_archive_cpu_seconds[policy] += time.process_time() - cpu_started

    def observe_building_blocks(self, current):
        if self.building_block_archive is not None and self.archive_mask['bb_snapshot']:
            started = time.perf_counter()
            cpu_started = time.process_time()
            self.building_block_archive.observe(current)
            self.archive_observation_seconds += time.perf_counter() - started
            self.archive_observation_cpu_seconds += time.process_time() - cpu_started
