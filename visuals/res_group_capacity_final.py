import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from config import DEFAULT
from mgp.analysis import capacity, cli

INPUT = DEFAULT
WHERE = DEFAULT
OUTPUT = DEFAULT
GROUP_TARGET = DEFAULT
CAPACITY_POOLS = DEFAULT
CAPACITY_COLUMNS = DEFAULT
DIVERSITY = DEFAULT
PART = 'dinf'
VISUAL = 'res_group_final_limited'


def main():
    args = cli.arguments(VISUAL, globals(), ('study2',), ('GROUP_TARGET', 'CAPACITY_POOLS', 'CAPACITY_COLUMNS', 'DIVERSITY'))
    path = args.paths['study2']
    o = args.option
    pools = tuple(o['CAPACITY_POOLS']) if o['CAPACITY_POOLS'] is not None else capacity.complete_pools(path, o['GROUP_TARGET'])
    columns = tuple(o['CAPACITY_COLUMNS']) + ((capacity.DIVERSITY_LABEL,) if o['DIVERSITY'] else ())
    vals = capacity.values(path, o['GROUP_TARGET'], pools, args.filters, diversity=o['DIVERSITY'])
    tag = f'pools{min(pools)}-{max(pools)}' if pools == tuple(range(min(pools), max(pools) + 1)) else 'pools' + '-'.join(map(str, pools))
    name = f'{VISUAL}_{tag}' + cli.suffix(args, ('GROUP_TARGET', 'CAPACITY_COLUMNS', 'DIVERSITY'))
    capacity.panels(vals, PART, cli.output(args, name, path), columns)


if __name__ == '__main__':
    main()
