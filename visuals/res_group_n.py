import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from config import DEFAULT
from mgp.analysis import cli, groups, style

INPUT = DEFAULT
WHERE = DEFAULT
OUTPUT = DEFAULT
GROUP_TARGET = DEFAULT
DIVERSITY = DEFAULT
FINAL_ARCHIVE_MAX_SIZE = DEFAULT


def main():
    args = cli.arguments('res_group_n_cols', globals(), ('study2',), ('GROUP_TARGET', 'DIVERSITY', 'FINAL_ARCHIVE_MAX_SIZE'))
    path = args.paths['study2']
    o = args.option
    growth = groups.group_growth(path, o['GROUP_TARGET'], args.filters, diversity=o['DIVERSITY'], final=o['FINAL_ARCHIVE_MAX_SIZE'])
    style.growth_figure(growth, cli.output(args, 'res_group_n_cols' + cli.suffix(args), path), o['DIVERSITY'], o['FINAL_ARCHIVE_MAX_SIZE'] is not None)


if __name__ == '__main__':
    main()
