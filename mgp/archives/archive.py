import numpy as np

from mgp.engine.fitness import object_fitness
from mgp.engine.objects import SemanticList, SemanticObject, underlying_construction
from mgp.engine.semantics import select_comparison
from mgp.engine.symbolic import symbolic_data, symbolic_has_variables


class Archive:

    def __init__(self, max_size=None, similarity=0.99, atomic_objects=False, comparison=None, comparison_is_distance=False, symbolic=False, target=None):
        if max_size is not None and (not isinstance(max_size, int) or max_size < 1):
            raise ValueError('single_archive_max_size must be a positive integer or None')
        if not 0 <= similarity <= 1:
            raise ValueError('archive_similarity must be between 0 and 1')
        self.max_size = max_size
        self.similarity = similarity
        self.atomic_objects = atomic_objects
        self.objects = []
        self.directions = []
        self.default_comparison = comparison is None
        self.comparison = comparison or select_comparison('cosine')
        self.comparison_is_distance = comparison_is_distance
        self.symbolic = symbolic
        self.symbolic_keys = set()
        self._symbolic_positions = {}
        self.target = target
        self.accepted_count = 0
        self.replacements = 0
        self.errors = []
        self.versions = []
        self.ordinals = {}
        self.version = 0

    def error(self, current):
        f = object_fitness(current, self.target)
        return f if np.isfinite(f) else np.inf

    def stamp(self, index, current):
        if self.max_size is None:
            return
        self.version += 1
        record = (self.error(current), self.version)
        if index == len(self.errors):
            self.errors.append(record[0])
            self.versions.append(record[1])
        else:
            self.errors[index], self.versions[index] = record

    def prefer(self, current, index):
        return self.target is not None and object_fitness(current, self.target) < object_fitness(self.objects[index], self.target)

    def add(self, current):
        if isinstance(current, SemanticList):
            accepted = [self.add(item) for item in current.alternatives]
            return any(accepted)
        if not isinstance(current, SemanticObject):
            return False
        if np.any(np.isinf(current.values)) or not np.any(np.isfinite(current.values)):
            return False
        if underlying_construction(current).is_atomic and (not self.atomic_objects):
            return False
        if self.objects and self.objects[0].values.shape != current.values.shape:
            raise ValueError('archive objects must use the same dataset')
        finite = np.isfinite(current.values)
        values = current.values[finite]
        if (len(values) > 1 or np.all(finite)) and np.all(values == values[0]):
            return False
        if self.symbolic:
            data = symbolic_data(current)
            if not symbolic_has_variables(current):
                return False
            key = data[0]
            if key in self.symbolic_keys:
                index = self._symbolic_positions[key]
                if self.prefer(current, index):
                    self.objects[index] = current
                    self.stamp(index, current)
                    self.replacements += 1
                    self.accepted_count += 1
                    return True
                return False
            if self.max_size is not None and len(self.objects) >= self.max_size:
                worst = max(range(len(self.objects)), key=lambda i: (self.errors[i], self.versions[i]))
                if not self.error(current) < self.errors[worst]:
                    return False
                old = symbolic_data(self.objects[worst])[0]
                self.symbolic_keys.discard(old)
                del self._symbolic_positions[old]
                self.symbolic_keys.add(key)
                self._symbolic_positions[key] = worst
                self.objects[worst] = current
                self.stamp(worst, current)
                self.replacements += 1
                self.accepted_count += 1
                return True
            self.symbolic_keys.add(key)
            self._symbolic_positions[key] = len(self.objects)
            self.objects.append(current)
            self.stamp(len(self.objects) - 1, current)
            self.accepted_count += 1
            return True
        scale = np.max(np.abs(values))
        direction = np.zeros_like(current.values)
        if scale:
            scaled = values / scale
            direction[finite] = scaled / np.linalg.norm(scaled)
        direction[~finite] = np.nan
        candidate_key = symbolic_data(current)[0] if self.target is not None else None
        if self.max_size is not None:
            self.ordinals.setdefault(candidate_key, len(self.ordinals))
        close_indices = []
        for index, previous in enumerate(self.directions):
            if candidate_key is not None and symbolic_data(self.objects[index])[0] == candidate_key:
                close_indices.append(index)
                continue
            if previous.shape != direction.shape:
                raise ValueError('archive objects must use the same dataset')
            if self.default_comparison:
                if not np.array_equal(np.isnan(previous), np.isnan(direction)):
                    continue
                score = np.dot(previous[finite], direction[finite])
            else:
                score = self.comparison(self.objects[index].values, current.values)
            if score is None:
                continue
            close = score >= self.similarity
            if self.comparison_is_distance:
                close = score <= self.similarity
            if self.default_comparison and np.array_equal(self.objects[index].values, current.values, equal_nan=True):
                close = True
            if self.default_comparison and self.similarity == 1.0 and np.array_equal(previous, direction, equal_nan=True):
                close = True
            if close:
                close_indices.append(index)
        if close_indices:
            if not all((self.prefer(current, index) for index in close_indices)):
                return False
            first = close_indices[0]
            self.objects[first] = current
            self.directions[first] = direction
            self.stamp(first, current)
            for index in reversed(close_indices[1:]):
                self.objects.pop(index)
                self.directions.pop(index)
                if self.max_size is not None:
                    self.errors.pop(index)
                    self.versions.pop(index)
            self.replacements += len(close_indices)
            self.accepted_count += 1
            return True
        if self.max_size is not None and len(self.objects) >= self.max_size:
            rank = lambda i: (self.errors[i], self.ordinals[symbolic_data(self.objects[i])[0]])
            worst = max(range(len(self.objects)), key=rank)
            if not self.error(current) < self.errors[worst]:
                return False
            self.objects[worst] = current
            self.directions[worst] = direction
            self.stamp(worst, current)
            self.replacements += 1
            self.accepted_count += 1
            return True
        self.objects.append(current)
        self.directions.append(direction)
        self.stamp(len(self.objects) - 1, current)
        self.accepted_count += 1
        return True
