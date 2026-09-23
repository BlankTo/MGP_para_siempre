import numpy as np

from .objects import SemanticObject
from .semantics import select_comparison


class Archive:
    def __init__(self, max_size=None, similarity=0.99, atomic_objects=False,
                 comparison=None, comparison_is_distance=False):
        if max_size is not None and (not isinstance(max_size, int) or max_size < 0):
            raise ValueError("archive_max_size must be a nonnegative integer or None.")

        if not 0 <= similarity <= 1:
            raise ValueError("archive_similarity must be between 0 and 1.")

        self.max_size = max_size
        self.similarity = similarity
        self.atomic_objects = atomic_objects
        self.objects = []
        self.directions = []
        self.default_comparison = comparison is None
        self.comparison = comparison or select_comparison("cosine")
        self.comparison_is_distance = comparison_is_distance

    def add(self, current):
        if not isinstance(current, SemanticObject):
            return False

        if not np.all(np.isfinite(current.values)):
            return False

        if current.is_atomic and not self.atomic_objects:
            return False

        if self.max_size is not None and len(self.objects) >= self.max_size:
            return False

        if np.all(current.values == current.values[0]):
            return False

        scaled = current.values / np.max(np.abs(current.values))
        direction = scaled / np.linalg.norm(scaled)

        for index, previous in enumerate(self.directions):
            if previous.shape != direction.shape:
                raise ValueError("Archive objects must use the same dataset.")

            if self.default_comparison:
                score = np.dot(previous, direction)
            else:
                score = self.comparison(self.objects[index].values, current.values)

            if score is None:
                continue

            close = score >= self.similarity

            if self.comparison_is_distance:
                close = score <= self.similarity

            if close:
                return False

            if self.default_comparison and self.similarity == 1.0 and np.array_equal(previous, direction):
                return False

        self.objects.append(current)
        self.directions.append(direction)

        return True
