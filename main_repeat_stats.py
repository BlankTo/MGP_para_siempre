from config import CONFIG
from mgp.repetition import print_results, repeat_derivations


REPEAT_STATS_CONFIG = {
    **CONFIG,

    "derivations": 100,
    "progress_every": 10,
}


def main():
    result = repeat_derivations(
        REPEAT_STATS_CONFIG,
        count=REPEAT_STATS_CONFIG["derivations"],
        progress_every=REPEAT_STATS_CONFIG["progress_every"],
    )

    print_results(result)


if __name__ == "__main__":
    main()
