import random

from config import CONFIG
from mgp import Derivation, SemanticObject, make_dataset, make_workspace, resolve_seed
from mgp.fitness import object_fitness
from targets import TARGETS


def main():
    seed = resolve_seed(CONFIG["seed"])
    dataset_seed = resolve_seed(CONFIG["dataset_seed"])
    generator = random.Random(seed)
    function, target_expression = TARGETS[CONFIG["target"]]
    variables, target = make_dataset(function, CONFIG, dataset_seed)

    workspace_seed = generator.getrandbits(63)
    derivation_seed = generator.getrandbits(63)
    workspace = make_workspace(variables, CONFIG, workspace_seed)
    derivation = Derivation(target, CONFIG, derivation_seed)
    best = derivation.run(workspace, verbose=True)

    print("\n" + "=" * 80)
    print("Target:", target_expression)
    print("Seed:", seed, "Dataset seed:", dataset_seed)
    print("Workspace seed:", workspace_seed, "Derivation seed:", derivation_seed)
    print("Termination:", derivation.termination_reason)
    print("MERGE operations:", derivation.merge_count)

    if derivation.final_object is not None:
        print("Final structure:", derivation.final_object.structure.text())
        print("Final object:", derivation.final_object.tree())
        print("Final type:", derivation.final_object.category)

        if isinstance(derivation.final_object, SemanticObject):
            print("Final RMSE:", object_fitness(derivation.final_object, derivation.target))

    if best is None:
        print("Best object: none")
    else:
        print("Best structure:", best.structure.text())
        print("Best object:", best.expression)
        print("Best RMSE:", derivation.best_fitness)
        print("Best source:", derivation.best_source)
        print("Best location:", derivation.best_location())

    print("Archive size:", len(derivation.archive.objects))


if __name__ == "__main__":
    main()
