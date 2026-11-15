import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from config import DEFAULT
from mgp.analysis import cli, data, groups, style

INPUT = DEFAULT
WHERE = DEFAULT
OUTPUT = DEFAULT
GROUP_TARGET = DEFAULT
DIVERSITY = DEFAULT
FINAL_ARCHIVE_MAX_SIZE = DEFAULT


def main():
    args = cli.arguments('res_group_n5000_cols', globals(), ('study7',), ('GROUP_TARGET', 'DIVERSITY', 'FINAL_ARCHIVE_MAX_SIZE'))
    path = args.paths['study7']
    o = args.option
    growth = groups.coverage_growth7(data.where(data.study7(path), args.filters), o['GROUP_TARGET'], final=o['FINAL_ARCHIVE_MAX_SIZE'], path7=path,
                                     diversity=o['DIVERSITY'])
    style.growth_figure(growth, cli.output(args, 'res_group_n5000_cols' + cli.suffix(args), path), o['DIVERSITY'],
                        o['FINAL_ARCHIVE_MAX_SIZE'] is not None)


if __name__ == '__main__':
    main()
