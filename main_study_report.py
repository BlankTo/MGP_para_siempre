import argparse

from mgp.study_report import create_report


def main():
    parser = argparse.ArgumentParser(description="Generate English figures, tables and findings from a saved study.")
    parser.add_argument("directory", help="Study directory containing study.json and study.sqlite3.")
    parser.add_argument("--target", help="Report one target only.")
    parser.add_argument("--baseline", type=int, help="Reference configuration for paired comparisons.")
    parser.add_argument("--configurations", nargs="+", type=int, help="Configurations for detailed step CSVs; workspace figures always pool all runs.")
    parser.add_argument("--output", help="Separate report directory; useful for preserving an earlier report.")
    arguments = parser.parse_args()
    create_report(arguments.directory, arguments.target, arguments.baseline, arguments.configurations, arguments.output)


if __name__ == "__main__":
    main()
