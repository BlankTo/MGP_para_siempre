from config import CONFIG
from mgp.repetition import print_results, repeat_derivations


REPEAT_CONFIG = {
    **CONFIG,

    #"target": "square_plus", "unary_operators_on": False, "variable_atom_count": 20, "operator_count": 10,
    #"target": "nguyen_1", "unary_operators_on": False, "variable_atom_count": 20, "operator_count": 10,
    #"target": "nguyen_8", "unary_operators_on": True, "variable_atom_count": 20, "operator_count": 10,

    "max_derivations": 10000,
    "progress_every": 100,
}


def main():
    result = repeat_derivations(
        REPEAT_CONFIG,
        count=REPEAT_CONFIG["max_derivations"],
        stop_on_solution=True,
        progress_every=REPEAT_CONFIG["progress_every"],
    )

    print_results(result)

    if result["solutions_found"]:
        print("Stopped after finding a solution.")
    else:
        print("Stopped at max_derivations without finding a solution.")


if __name__ == "__main__":
    main()
