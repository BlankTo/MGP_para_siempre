import numpy as np

from .objects import SemanticObject


class Archive:
    def __init__(self, max_size=None, similarity=0.99, atomic_objects=False):
        if max_size is not None and (not isinstance(max_size, int) or max_size < 0):
            raise ValueError("archive_max_size must be a nonnegative integer or None.")

        if not 0 <= similarity <= 1:
            raise ValueError("archive_similarity must be between 0 and 1.")

        self.max_size = max_size
        self.similarity = similarity
        self.atomic_objects = atomic_objects
        self.objects = []
        self.directions = []

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

        for previous in self.directions:
            if previous.shape != direction.shape:
                raise ValueError("Archive objects must use the same dataset.")

            if np.dot(previous, direction) >= self.similarity:
                return False

            if self.similarity == 1.0 and np.array_equal(previous, direction):
                return False

        self.objects.append(current)
        self.directions.append(direction)

        return True
