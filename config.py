CONFIG = {
    "seed": None, #42,
    "dataset_seed": None, #1,
    "target": "nguyen_1", #"square_plus",

    "sample_count": 1000,
    "sample_min": None,
    "sample_max": None,

    "variable_atom_count": 50,
    "operator_count": 20,
    "unary_operators_on": False,
    "binary_operators_on": True,
    "ternary_operators_on": False,

    "fitness_probability": 1.0,
    "internal_merge_probability": 0.0,
    "perfect_fit_tolerance": 1e-14, #0.0,
    "max_merge": 1000,
    "esel": 0.5,
    "esur": 0.0,

    "archive_enabled": True,
    "archive_max_size": None,
    "archive_similarity": 0.99,
    "archive_atomic_objects": False,
}
