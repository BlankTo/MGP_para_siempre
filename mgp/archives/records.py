import io

import numpy as np

from mgp.engine.fitness import object_fitness
from mgp.engine.objects import IncompleteObject, OperatorObject, SemanticList, SemanticObject
from mgp.engine.symbolic import sorted_guard_tokens, symbolic_data


class ObjectRecords:

    def __init__(self, labels, clean):
        self.labels, self.clean = (labels, clean)
        self.objects, self.indices = ([], {})
        self.constructions, self.occurrences = ({}, {})
        self.evaluations, self.evaluation_indices = ([], {})
        self.operator_rows = {}

    def operator_payload(self, node):
        token = id(node.operators)
        if token not in self.operator_rows:
            self.operator_rows[token] = (node.operators, list(node.operators), node.expression)
        return self.operator_rows[token][1:]

    def ref(self, node):
        if node is None:
            return None
        if node not in self.indices:
            self.indices[node] = len(self.objects)
            self.objects.append(node)
        return self.indices[node]

    def evaluation_ref(self, value):
        if value is None:
            return None
        key = ('variable', value) if isinstance(value, str) else ('object', id(value))
        if key not in self.evaluation_indices:
            self.evaluation_indices[key] = len(self.evaluations)
            self.evaluations.append(value)
        return self.evaluation_indices[key]

    def export(self):
        rows = []
        semantic_rows = {}
        index = 0
        while index < len(self.objects):
            node = self.objects[index]
            if isinstance(node, OperatorObject):
                operators, expression = self.operator_payload(node)
            else:
                expression = node.expression
            row = {'node_id': index, 'kind': node.kind, 'category': node.category, 'expression': expression, 'is_atomic': node.is_atomic, 'children': [self.ref(n) for n in node.children], 'internal_accessible': node.internal_accessible, 'accessible_count': node.accessible_count}
            if isinstance(node, OperatorObject):
                row.update(operators=operators, arity=node.arity)
            if isinstance(node, IncompleteObject):
                row.update(semantic=self.ref(node.semantic), head=self.ref(node.head))
            if isinstance(node, SemanticObject):
                construction = self.constructions.setdefault(node.semantic_identity, len(self.constructions))
                occurrence = self.occurrences.setdefault(node.occurrence_identity, len(self.occurrences))
                row.update(construction_id=construction, occurrence_id=occurrence, archive_origin=node.archive_origin, evaluation_ref=self.evaluation_ref(node.evaluation), evaluation_children=[self.ref(n) for n in node.evaluation_children], used_semantic_children=[self.ref(n) for n in node.used_semantic_children], used_operator=node.used_operator, unary_argument=self.ref(node.unary_argument), ic_alternatives=[self.ref(n) for n in node.ic_alternatives])
                if construction not in semantic_rows:
                    shared = dict(train_rmse=object_fitness(node, self.labels), clean_rmse=object_fitness(node, self.clean), nan_samples=int(np.isnan(node.values).sum()), inf_samples=int(np.isinf(node.values).sum()))
                    if not isinstance(node, SemanticList):
                        data = symbolic_data(node)
                        shared.update(symbolic_key=data[0], canonical=data[1], domain_guards=sorted_guard_tokens(data[3]))
                    semantic_rows[construction] = shared
                row.update(semantic_rows[construction])
                if isinstance(node, SemanticList):
                    row.update(alternatives=[self.ref(n) for n in node.alternatives], multiplicities=list(node.multiplicities), representative=self.ref(node.representative))
            rows.append(row)
            index += 1
        evaluations = []
        index = 0
        while index < len(self.evaluations):
            value = self.evaluations[index]
            if isinstance(value, str):
                row = {'evaluation_id': index, 'variable': value}
            else:
                row = {'evaluation_id': index, 'operator': value[0], 'arguments': [self.evaluation_ref(child) for child in value[1:]]}
            evaluations.append(row)
            index += 1
        return {'nodes': rows, 'evaluations': evaluations}


def pack_dataset(variables, clean, labels, test):
    arrays = {'variable_names': np.array(list(variables)), 'train_clean': clean, 'train_labels': labels, 'test_clean': test[1]}
    for name, values in variables.items():
        arrays['train_' + name] = values
        arrays['test_' + name] = test[0][name]
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()
