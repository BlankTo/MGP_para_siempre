import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np
from matplotlib.lines import Line2D

from config import DEFAULT
from mgp.analysis import cli, data, style
from mgp.progress import log
from mgp.analysis.style import INK2, plt

INPUT = DEFAULT
OUTPUT = DEFAULT
OP, SEM, INC = '#6250d6', '#e34948', '#eda100'
SERIES = [('operators', OP, '-', lambda M: M[:, 2] + M[:, 3]), ('semantic objects', SEM, '-', lambda M: M[:, 0] + M[:, 1]),
          ('incomplete objects', INC, '-', lambda M: M[:, 4])]


def figure(Z, base):
    from mgp.experiments.plan import object_counts
    style.use('workspace')
    keys = [k.split('|') for k in Z]
    sizes = sorted({int(s) for s, _ in keys})
    ratios = [r for r in data.LEVEL_ORDER['ratio'] if r in {r for _, r in keys}]
    fig, axes = plt.subplots(len(sizes), len(ratios), figsize=(4.8, 3.9), sharex='row', sharey='row', squeeze=False)
    for i, size in enumerate(sizes):
        for j, ratio in enumerate(ratios):
            ax = axes[i, j]
            M = Z[f'{size}|{ratio}']
            t = np.arange(len(M))
            for _, color, ls, f in SERIES:
                ax.plot(t, f(M), color=color, ls=ls)
            n_o, n_v = object_counts(size, ratio)
            ax.text(0.97, 0.97, f'{n_o} op.\n{n_v} var.', transform=ax.transAxes, ha='right', va='top', fontsize=5.6, color=INK2)
            ax.set_ylim(0, None)
            ax.set_xticks([0, size // 2, size])
            if i == 0:
                ax.set_title(f'ratio {ratio}')
            if j == 0:
                ax.set_ylabel(f'$N_0={size}$\nobjects')
            if i == len(sizes) - 1:
                ax.set_xlabel('MERGE step')
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    fig.legend([Line2D([], [], color=c, ls=ls) for _, c, ls, _ in SERIES], [s[0] for s in SERIES], loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=3,
               handlelength=2.6, columnspacing=1.2)
    style.save(fig, base)


def main():
    args = cli.arguments('res_workspace_grid', globals(), ('workspace',))
    path = args.paths['workspace']
    Z, counts = data.workspace(path)
    log('res_workspace_grid', 'cells', derivations=','.join(map(str, sorted(set(counts.values())))))
    figure(Z, cli.output(args, 'res_workspace_grid', path))


if __name__ == '__main__':
    main()
