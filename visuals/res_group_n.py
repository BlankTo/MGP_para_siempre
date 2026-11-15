import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from config import DEFAULT
from mgp.analysis import cli, groups, style
from mgp.analysis.stats import SN
from mgp.analysis.style import GOALS, LW, MS, W, plt

INPUT = DEFAULT
WHERE = DEFAULT
OUTPUT = DEFAULT
GROUP_TARGET = DEFAULT
DIVERSITY = DEFAULT
FINAL_ARCHIVE_MAX_SIZE = DEFAULT
ROWS = (('BBA', 'BA and BBA'), ('FBBA', 'FBBA'))
BOUNDED_ROWS = (('BA', 'BA'), ('BBA', 'BBA'), ('FBBA', 'FBBA'))
DIVERSITY_ROW = ('BAdiv', 'BA diversity')


def ticks(size):
    return [10 ** k for k in range(len(str(size))) if 10 ** k <= size]


def figure(growth, base, diversity=False, bounded=False):
    rows = (BOUNDED_ROWS if bounded else ROWS) + ((DIVERSITY_ROW,) if diversity else ())
    style.use('grid')
    res, n_grid, strategies = growth['values'], growth['n'], growth['strategies']
    xticks = ticks(growth['group_size'])
    fig, axes = plt.subplots(len(rows), len(strategies), figsize=(W, round(1.1 * len(rows), 2)), sharex=True, sharey=True, squeeze=False)
    for i, (an, _) in enumerate(rows):
        for j, s in enumerate(strategies):
            ax = axes[i, j]
            for g, _, c, ls, mk in GOALS:
                ax.plot(n_grid, res[f'{s}|{an}|{g}'], color=c, ls=ls, marker=mk, ms=MS, lw=LW)
            ax.set_xscale('log')
            ax.set_xticks(xticks, [f'{t:,}' for t in xticks])
            ax.minorticks_off()
            ax.set_ylim(-3, 103)
            ax.set_yticks([0, 50, 100])
    for j, s in enumerate(strategies):
        axes[0, j].set_title(SN[s], fontsize=8.5, fontweight='bold')
    axes[-1, len(strategies) // 2].set_xlabel('derivations in the group')
    for i, (_, t) in enumerate(rows):
        axes[i, 0].set_ylabel(f'{t}\ncoverage (%)')
    fig.tight_layout(h_pad=0.5, w_pad=0.6)
    style.goal_legend(fig, 0.995)
    style.save(fig, base)


def main():
    args = cli.arguments('res_group_n_cols', globals(), ('study2',), ('GROUP_TARGET', 'DIVERSITY', 'FINAL_ARCHIVE_MAX_SIZE'))
    path = args.paths['study2']
    o = args.option
    growth = groups.group_growth(path, o['GROUP_TARGET'], args.filters, diversity=o['DIVERSITY'], final=o['FINAL_ARCHIVE_MAX_SIZE'])
    figure(growth, cli.output(args, 'res_group_n_cols' + cli.suffix(args), path), o['DIVERSITY'], o['FINAL_ARCHIVE_MAX_SIZE'] is not None)


if __name__ == '__main__':
    main()
