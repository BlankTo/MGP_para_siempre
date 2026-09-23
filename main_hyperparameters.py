import argparse

from mgp.study import run_study


STUDY_CONFIG = {
    "seed": 42,
    "dataset_seed": 1,
    "targets": ["g1", "nguyen_1", "nguyen_2", "nguyen_3", "nguyen_4", "g2", "nguyen_12"],
    "target": "g1",
    "repetitions": 10,
    "sample_count": 64,
    "sample_min": -20.0,
    "sample_max": 20.0,

    "variable_atom_count": 8,
    "operator_count": 4,
    "esel": 0.0,
    "esur": 0.0,
    "unary_operators_on": False,
    "binary_operators_on": True,
    "ternary_operators_on": False,
    "fitness_probability": 1.0,
    "internal_merge_probability": 0.0,
    "esur_in_IM": False,
    "IM_terminate_on_one": True,
    "max_merge": 1000,
    "perfect_fit_tolerance": 1e-8,
    "subtarget_tolerance": 1e-8,

    "archive_enabled": True,
    "archive_max_size": None,
    "archive_similarity": 0.99,
    "archive_atomic_objects": False,

    "comparison_method": "cosine",
    "comparison_parameters": {},
    "equivalence_method": "rmse",
    "equivalence_tolerance": 1e-10,
    "equivalence_atol": 1e-10,
    "equivalence_rtol": 1e-10,
    "histogram_edges": [-1.0 + index / 10 for index in range(21)],

    "probe_diagnostics": True,
    "probe_axis_points": 9,
    "probe_near_zero": 1e-10,
    "save_report": True,
    "trace_run_ids": [],
    "output_dir": "hyperparameter_output",
    "progress_every": 100,
}

HYPERPARAMETER_GRID = {
    "esel": [0.0, 0.25, 0.5, 0.75, 1.0],
    "esur": [0.0, 0.25, 0.5, 0.75, 1.0],
    "variable_atom_count": [8, 32, 64, 256],
    "operator_count": [4, 16, 32, 128],
}


def main():
    parser = argparse.ArgumentParser(description="Run or resume the paired EM-only study.")
    parser.add_argument("--resume", help="Directory of an existing study; uses its saved settings.")
    parser.add_argument("--max-runs", type=int, help="Maximum additional runs in this session.")
    arguments = parser.parse_args()
    run_study(STUDY_CONFIG, HYPERPARAMETER_GRID, resume=arguments.resume, max_runs=arguments.max_runs)


if __name__ == "__main__":
    main()
