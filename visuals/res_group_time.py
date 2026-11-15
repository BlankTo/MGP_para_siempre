import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np
from matplotlib.lines import Line2D

from config import DEFAULT
from mgp.analysis import cli, groups, style
from mgp.analysis.stats import SN
from mgp.analysis.style import COL, LW, W, plt

INPUTS = {'study2': DEFAULT, 'study6': DEFAULT}
WHERE = DEFAULT
OUTPUT = DEFAULT
GROUP_TARGET = DEFAULT
TIME = DEFAULT
DIVERSITY = DEFAULT
FINAL_ARCHIVE_MAX_SIZE = DEFAULT
GOALS = [('component', 'components'), ('cooccurrence', 'component sets'), ('composite', 'composites')]


def figure(res, base):
    style.use('paper')
    B = res['budgets_s']
    strategies = res['strategies']
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.3), sharey=True)
    for ax, (g, title) in zip(axes, GOALS):
        for s in strategies:
            v = np.array([np.nan if x is None else x for x in res[f'{s}|{g}']])
            ax.plot(B, v, color=COL[s], lw=LW, label=SN[s])
        ax.set_xscale('log')
        ax.set_xticks([1e-2, 1, 1e2, 1e4], ['0.01', '1', '100', '10⁴'])
        ax.minorticks_off()
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('CPU time (s)')
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 50, 100])
    axes[0].set_ylabel('BBA coverage (%)')
    fig.tight_layout(w_pad=0.6)
    fig.legend([Line2D([], [], color=COL[s], lw=LW) for s in strategies], [SN[s] for s in strategies], loc='lower center', bbox_to_anchor=(0.5, 0.99),
               ncol=len(strategies))
    style.save(fig, base)


def main():
    args = cli.arguments('res_group_time_short', globals(), ('study2', 'study6'), ('GROUP_TARGET', 'TIME', 'DIVERSITY', 'FINAL_ARCHIVE_MAX_SIZE'))
    p2, p6 = args.paths['study2'], args.paths['study6']
    o = args.option
    res = groups.group_time(p2, p6, o['GROUP_TARGET'], args.filters, o['TIME'], diversity=o['DIVERSITY'], final=o['FINAL_ARCHIVE_MAX_SIZE'])
    figure(res, cli.output(args, 'res_group_time_short' + cli.suffix(args), p2, p6))


if __name__ == '__main__':
    main()
