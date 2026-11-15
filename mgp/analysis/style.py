import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mgp.analysis.stats import SN
from mgp.progress import log

INK, INK2, GRIDC = '#1f1f1e', '#52514e', '#e4e3df'
COL = {'standard': '#2a78d6', 'revisit_ic': '#eb6834', 'constructive': '#1baf7a'}
LS = {0.0: (0, (4, 2)), 0.5: '-'}
GOALS = [('component', 'components', '#2a78d6', '-', 'o'), ('cooccurrence', 'co-occurrence', '#eb6834', (0, (4, 2)), 's'),
         ('composite', 'composites', '#1baf7a', (0, (1, 1.2)), '^')]
W = 4.8
LW, MS, MS_OPEN, MEW, LEG_MS = 0.9, 2.0, 2.8, 0.8, 2.6
BASE = {'axes.axisbelow': True, 'axes.edgecolor': INK2, 'axes.grid': True, 'axes.labelcolor': INK, 'axes.labelsize': 8.0, 'axes.linewidth': 0.6,
        'axes.spines.right': False, 'axes.spines.top': False, 'axes.titlesize': 8.5, 'font.family': ['DejaVu Sans'], 'font.size': 8.0,
        'grid.color': GRIDC, 'grid.linewidth': 0.5, 'legend.fontsize': 7.5, 'legend.frameon': False, 'lines.linewidth': 1.6, 'pdf.fonttype': 42,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03, 'text.color': INK, 'xtick.color': INK2, 'xtick.labelsize': 7.5,
        'xtick.major.width': 0.6, 'ytick.color': INK2, 'ytick.labelsize': 7.5, 'ytick.major.width': 0.6}
STYLES = {
    'paper': BASE,
    'grid': {**{k: v for k, v in BASE.items() if k != 'lines.linewidth'}, 'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2},
    'workspace': {**{k: v for k, v in BASE.items() if k not in ('axes.labelsize', 'axes.linewidth', 'xtick.major.width', 'ytick.major.width')},
                  'axes.titlesize': 7.5, 'font.size': 7.0, 'lines.linewidth': 1.3, 'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0},
}


def use(name):
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(STYLES[name])


def save(fig, base):
    fig.savefig(f'{base}.pdf')
    fig.savefig(f'{base}.png', dpi=220)
    plt.close(fig)
    log('write', f'{base}.pdf')


def goal_legend(fig, y):
    h = [Line2D([], [], color=c, ls=ls, marker=mk, ms=LEG_MS, lw=LW) for _, _, c, ls, mk in GOALS]
    fig.legend(h, [lab for _, lab, _, _, _ in GOALS], loc='lower center', bbox_to_anchor=(0.5, y), ncol=3, handlelength=3.0, columnspacing=1.6)


ROWS = (('BBA', 'BA and BBA'), ('FBBA', 'FBBA'))
BOUNDED_ROWS = (('BA', 'BA'), ('BBA', 'BBA'), ('FBBA', 'FBBA'))
DIVERSITY_ROW = ('BAdiv', 'BA diversity')


def ticks(size):
    return [10 ** k for k in range(len(str(size))) if 10 ** k <= size]


def growth_figure(growth, base, diversity=False, bounded=False):
    rows = (BOUNDED_ROWS if bounded else ROWS) + ((DIVERSITY_ROW,) if diversity else ())
    use('grid')
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
    goal_legend(fig, 0.995)
    save(fig, base)
