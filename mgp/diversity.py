from collections import Counter
from itertools import combinations_with_replacement
from math import fsum

import numpy as np

from .objects import SemanticObject
from .semantics import cosine_similarity, rmse_equivalent


WORKSPACE_CLASSES = ("AS", "AO", "CS", "CO", "I", "ICO")


def workspace_composition(workspace):
    counts = dict.fromkeys(WORKSPACE_CLASSES, 0)

    for current in workspace:
        if current.kind == "semantic":
            category = "AS" if current.is_atomic else "CS"
        elif current.kind == "operator":
            category = "AO" if current.is_atomic else "CO"
        else:
            category = "I" if current.head.is_atomic else "ICO"

        counts[category] += 1

    return counts


class WorkspaceDiversity:
    def __init__(self, comparison=cosine_similarity, equivalence=rmse_equivalent, bin_edges=None):
        if bin_edges is None:
            bin_edges = np.linspace(-1, 1, 21)

        self.edges = np.asarray(bin_edges, dtype=float)

        if self.edges.ndim != 1 or len(self.edges) < 2:
            raise ValueError("histogram_edges needs at least two boundaries.")

        if not np.all(np.isfinite(self.edges)) or not np.all(np.diff(self.edges) > 0):
            raise ValueError("Histogram boundaries must be finite and strictly increasing.")

        self.comparison = comparison
        self.equivalence = equivalence
        self.vector_keys = {}
        self.previous_signature = None
        self.previous_metrics = None
        self.previous_histogram = None
        self.counts = Counter()
        self.vectors = {}
        self.histogram = np.zeros(len(self.edges) - 1, dtype=np.int64)
        self.pair_sum = 0.0
        self.pair_correction = 0.0
        self.valid_pairs = 0

    def vector_key(self, values):
        identity = id(values)

        if identity not in self.vector_keys:
            key = None

            if np.all(np.isfinite(values)):
                canonical = values.copy()
                canonical[canonical == 0] = 0.0
                key = canonical.tobytes()

            self.vector_keys[identity] = (values, key)

        return self.vector_keys[identity][1]

    def change_pairs(self, first, second, weight):
        if weight == 0:
            return

        value = self.comparison(self.vectors[first], self.vectors[second])

        if value is None or not np.isfinite(value):
            return

        if value < self.edges[0] or value > self.edges[-1]:
            raise ValueError("A comparison score is outside histogram_edges.")

        bin_index = int(np.searchsorted(self.edges, value, side="right")) - 1
        bin_index = min(bin_index, len(self.histogram) - 1)
        self.histogram[bin_index] += weight
        # Compensated summation limits drift when many pairs are added and removed.
        change = weight * float(value)
        updated = self.pair_sum + change
        if abs(self.pair_sum) >= abs(change):
            self.pair_correction += (self.pair_sum - updated) + change
        else:
            self.pair_correction += (change - updated) + self.pair_sum
        self.pair_sum = updated
        self.valid_pairs += weight

    def change_count(self, key, new_count):
        old_count = self.counts[key]
        difference = new_count - old_count

        for other, count in self.counts.items():
            if other != key:
                self.change_pairs(key, other, difference * count)

        old_pairs = old_count * (old_count - 1) // 2
        new_pairs = new_count * (new_count - 1) // 2
        self.change_pairs(key, key, new_pairs - old_pairs)
        self.counts[key] = new_count

    def measure(self, workspace):
        counts = Counter()
        semantic_count = 0
        nonfinite_count = 0

        for current in workspace:
            if not isinstance(current, SemanticObject):
                continue

            semantic_count += 1
            key = self.vector_key(current.values)

            if key is None:
                nonfinite_count += 1
            else:
                counts[key] += 1
                self.vectors[key] = current.values

        signature = (tuple(counts.items()), nonfinite_count)

        if signature == self.previous_signature:
            return self.previous_metrics.copy(), self.previous_histogram.copy()

        previous_pair_count = self.valid_pairs
        keys = list(self.counts)

        for key in counts:
            if key not in self.counts:
                keys.append(key)

        for key in keys:
            if self.counts[key] != counts[key]:
                self.change_count(key, counts[key])

        for key in keys:
            if not counts[key]:
                del self.counts[key]
                del self.vectors[key]

        # After a large contraction, rounding in old weighted products can
        # dominate the few remaining pairs. Rebuild only in that case.
        if 0 < self.valid_pairs < previous_pair_count / 2:
            contributions = []
            for first, second in combinations_with_replacement(self.counts, 2):
                weight = self.counts[first] * self.counts[second]
                if first == second:
                    weight = self.counts[first] * (self.counts[first] - 1) // 2
                if not weight:
                    continue
                value = self.comparison(self.vectors[first], self.vectors[second])
                if value is not None and np.isfinite(value):
                    contributions.append(weight * float(value))
            self.pair_sum = fsum(contributions)
            self.pair_correction = 0.0

        representatives = []
        entries = list(counts)

        for first in entries:
            equivalent = False

            for representative in representatives:
                if self.equivalence(self.vectors[first], representative):
                    equivalent = True
                    break

            if not equivalent:
                representatives.append(self.vectors[first])

        mean = None

        if self.valid_pairs:
            mean = (self.pair_sum + self.pair_correction) / self.valid_pairs
            mean = float(np.clip(mean, self.edges[0], self.edges[-1]))
        else:
            self.pair_sum = 0.0
            self.pair_correction = 0.0

        total_pairs = semantic_count * (semantic_count - 1) // 2
        metrics = {
            "semantic_count": semantic_count,
            "nonfinite_semantic_count": nonfinite_count,
            "unique_semantic_count": len(representatives),
            "pair_count": total_pairs,
            "valid_pair_count": self.valid_pairs,
            "undefined_pair_count": total_pairs - self.valid_pairs,
            "mean_comparison": mean,
        }
        self.previous_signature = signature
        self.previous_metrics = metrics
        self.previous_histogram = self.histogram.copy()

        return metrics.copy(), self.histogram.copy()
