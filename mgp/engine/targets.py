import hashlib
from functools import lru_cache
from itertools import combinations

import numpy as np


def addition(x1, x2):
    return x1 + x2


def square_plus(x1, x2):
    return x1 * x1 + x2


def product_sum(x1, x2):
    return x1 * (x1 + x2)


def difference_squares(x1, x2):
    return (x1 - x2) * (x1 + x2)


def cubic(x1):
    return x1 * x1 * x1 + x1 * x1 + x1


def sine(x1):
    return np.sin(x1)


def polynomial_sine(x1):
    return x1 * x1 + x1 + np.sin(x1)


def g1(x1):
    return x1 + x1 ** 2


def g2(x1, x2):
    return x1 ** 2 + x2 ** 2


def nguyen_1(x1):
    return x1 + x1 ** 2 + x1 ** 3


def nguyen_2(x1):
    return x1 + x1 ** 2 + x1 ** 3 + x1 ** 4


def nguyen_3(x1):
    return x1 + x1 ** 2 + x1 ** 3 + x1 ** 4 + x1 ** 5


def nguyen_4(x1):
    return x1 + x1 ** 2 + x1 ** 3 + x1 ** 4 + x1 ** 5 + x1 ** 6


def nguyen_5(x1):
    return np.sin(x1 ** 2) * np.cos(x1)


def nguyen_6(x1):
    return np.sin(x1) + np.sin(x1 + x1 ** 2)


def nguyen_7(x1):
    return np.log(x1 + 1) + np.log(x1 ** 2)


def nguyen_8(x1):
    return np.sqrt(x1)


def nguyen_9(x1, x2):
    return np.sin(x1) + np.sin(x2 ** 2)


def nguyen_10(x1, x2):
    return np.sin(x1) * np.cos(x2)


def nguyen_12(x1, x2):
    return x1 ** 4 - x1 ** 3 + x2 ** 2 - x2


def heldout_u1(x1):
    return x1 ** 5 + x1 ** 3 + x1


def heldout_u2(x1):
    return x1 ** 6 + x1 ** 3 - x1


def heldout_m1(x1, x2):
    return x1 ** 3 + x2 ** 3 + x2


def heldout_m2(x1, x2):
    return x1 ** 4 + x1 ** 2 - x2 ** 3 + x2


def heldout_s1(x1, x2):
    return x1 * x2 * (x1 + x2)


def heldout_s2(x1, x2):
    return x2 * (x1 + x1 ** 2)


TARGETS = {'g1': (g1, 'x1 + x1**2'), 'g2': (g2, 'x1**2 + x2**2'), 'addition': (addition, 'x1 + x2'), 'square_plus': (square_plus, 'x1 * x1 + x2'), 'product_sum': (product_sum, 'x1 * (x1 + x2)'), 'difference_squares': (difference_squares, '(x1 - x2) * (x1 + x2)'), 'cubic': (cubic, 'x1 * x1 * x1 + x1 * x1 + x1'), 'sine': (sine, 'sin(x1)'), 'polynomial_sine': (polynomial_sine, 'x1 * x1 + x1 + sin(x1)'), 'nguyen_1': (nguyen_1, 'x1 + x1**2 + x1**3'), 'nguyen_2': (nguyen_2, 'x1 + x1**2 + x1**3 + x1**4'), 'nguyen_3': (nguyen_3, 'x1 + x1**2 + x1**3 + x1**4 + x1**5'), 'nguyen_4': (nguyen_4, 'x1 + x1**2 + x1**3 + x1**4 + x1**5 + x1**6'), 'nguyen_5': (nguyen_5, 'sin(x1**2) * cos(x1)'), 'nguyen_6': (nguyen_6, 'sin(x1) + sin(x1 + x1**2)'), 'nguyen_7': (nguyen_7, 'log(x1 + 1) + log(x1**2)'), 'nguyen_8': (nguyen_8, 'sqrt(x1)'), 'nguyen_9': (nguyen_9, 'sin(x1) + sin(x2**2)'), 'nguyen_10': (nguyen_10, 'sin(x1) * cos(x2)'), 'nguyen_12': (nguyen_12, 'x1**4 - x1**3 + x2**2 - x2'), 'heldout_u1': (heldout_u1, 'x1**5 + x1**3 + x1'), 'heldout_u2': (heldout_u2, 'x1**6 + x1**3 - x1'), 'heldout_m1': (heldout_m1, 'x1**3 + x2**3 + x2'), 'heldout_m2': (heldout_m2, 'x1**4 + x1**2 - x2**3 + x2'), 'heldout_s1': (heldout_s1, 'x1 * x2 * (x1 + x2)'), 'heldout_s2': (heldout_s2, 'x2 * (x1 + x1**2)')}
TARGET_DOMAINS = {g1: (-1.0, 1.0), g2: (0.0, 1.0), nguyen_1: (-1.0, 1.0), nguyen_2: (-1.0, 1.0), nguyen_3: (-1.0, 1.0), nguyen_4: (-1.0, 1.0), nguyen_5: (-1.0, 1.0), nguyen_6: (-1.0, 1.0), nguyen_7: (0.0, 2.0), nguyen_8: (0.0, 4.0), nguyen_9: (0.0, 1.0), nguyen_10: (0.0, 1.0), nguyen_12: (0.0, 1.0), heldout_u1: (-1.0, 1.0), heldout_u2: (-1.0, 1.0), heldout_m1: (0.0, 1.0), heldout_m2: (0.0, 1.0), heldout_s1: (0.0, 1.0), heldout_s2: (0.0, 1.0)}
POLYNOMIAL_TERMS = {'g1': [('x1', 1, 1), ('x1', 2, 1)], 'g2': [('x1', 2, 1), ('x2', 2, 1)], 'nguyen_1': [('x1', power, 1) for power in range(1, 4)], 'nguyen_2': [('x1', power, 1) for power in range(1, 5)], 'nguyen_3': [('x1', power, 1) for power in range(1, 6)], 'nguyen_4': [('x1', power, 1) for power in range(1, 7)], 'nguyen_12': [('x1', 4, 1), ('x1', 3, -1), ('x2', 2, 1), ('x2', 1, -1)], 'heldout_u1': [('x1', 5, 1), ('x1', 3, 1), ('x1', 1, 1)], 'heldout_u2': [('x1', 6, 1), ('x1', 3, 1), ('x1', 1, -1)], 'heldout_m1': [('x1', 3, 1), ('x2', 3, 1), ('x2', 1, 1)], 'heldout_m2': [('x1', 4, 1), ('x1', 2, 1), ('x2', 3, -1), ('x2', 1, 1)], 'heldout_s1': [], 'heldout_s2': []}


def lexical_component_ids(target_name):
    return frozenset((1 << index for index, (variable, power, sign) in enumerate(POLYNOMIAL_TERMS[target_name]) if power == 1))


def subtarget_specs(target_name, version=1):
    terms = POLYNOMIAL_TERMS[target_name]
    result = []
    for size in range(1, len(terms) + int(version >= 2)):
        for indices in combinations(range(len(terms)), size):
            labels = []
            identifier = 0
            for index in indices:
                variable, power, sign = terms[index]
                label = variable if power == 1 else variable + '**' + str(power)
                negative = version >= 2 and size > 1 and (sign < 0)
                labels.append((' - ' if negative else ' + ') + label)
                identifier += 1 << index
            expression = ''.join(labels).lstrip()
            if expression.startswith('+ '):
                expression = expression[2:]
            result.append((identifier, expression, indices))
    return result


def make_subtargets(target_name, variables, version=1):
    terms = POLYNOMIAL_TERMS[target_name]
    subtargets = []
    for identifier, expression, indices in subtarget_specs(target_name, version):
        values = np.zeros_like(next(iter(variables.values())))
        for index in indices:
            variable, power, sign = terms[index]
            coefficient = sign if version >= 2 and len(indices) > 1 else 1
            values += coefficient * variables[variable] ** power
        values.setflags(write=False)
        subtargets.append((identifier, expression, values))
    return subtargets


@lru_cache(maxsize=32)
def component_sets(components):
    return tuple((sum(items) for size in range(2, len(components) + 1) for items in combinations(components, size)))


def coverage_measures(hits, identifiers, lexical=frozenset()):
    hits, identifiers = (set(hits), set(identifiers))
    singletons = {item for item in identifiers if item.bit_count() == 1}
    components = singletons - set(lexical)
    composites = identifiers - singletons
    groups = component_sets(tuple(sorted(components)))
    available = sum(hits & components)
    together = sum((mask & available == mask for mask in groups))
    return {'component_coverage': len(hits & components) / len(components) if components else None, 'composite_coverage': len(hits & composites) / len(composites) if composites else None, 'cooccurrence_coverage': together / len(groups) if groups else None, 'components_present': len(hits & components), 'components_total': len(components), 'composites_present': len(hits & composites), 'composites_total': len(composites), 'component_sets_present': together, 'component_sets_total': len(groups), 'all_components_present': int(components <= hits) if components else None}


def catalog(target):
    from mgp.engine.symbolic import expression_key
    keys = {}
    for identifier, text, _ in subtarget_specs(target, 2):
        digest = hashlib.blake2b(expression_key(text).encode('utf-8'), digest_size=16).digest()
        if digest in keys and keys[digest] != identifier:
            raise ValueError('subtarget catalog digest collision')
        keys[digest] = identifier
    return keys, frozenset(keys.values()), lexical_component_ids(target), coverage_measures
