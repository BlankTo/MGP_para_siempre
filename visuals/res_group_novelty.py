import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np

from config import DEFAULT
from mgp.analysis import cli, groups, style
from mgp.analysis.stats import SN
from mgp.analysis.style import COL, plt


INPUT = DEFAULT
OUTPUT = DEFAULT
GROUP_TARGET = DEFAULT
NOVELTY_POOLS = DEFAULT
def figure(curves, base):
    style.use('paper')
    strategies = list(curves)
    nmax = len(curves[strategies[0]][0])
    n = np.arange(1, nmax + 1)
    fig, ax = plt.subplots(figsize=(5.2, 1.8))
    for s in strategies:
        x = np.stack(curves[s]).astype(float)
        m = x.mean(axis=0)
        h = 1.96 * x.std(axis=0, ddof=1) / np.sqrt(len(x))
        ax.fill_between(n, np.maximum(m - h, 0.5), m + h, color=COL[s], alpha=0.18, lw=0)
        ax.plot(n, m, color=COL[s], lw=0.9, label=SN[s])
    xt = [10 ** k for k in range(len(str(nmax))) if 10 ** k <= nmax]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(xt, [f'{t:,}' for t in xt])
    ax.set_yticks([1, 10, 100], ['1', '10', '100'])
    ax.minorticks_off()
    ax.set_xlim(0.85, round(1.18 * nmax, 6))
    ax.set_ylim(0.8, 200)
    ax.set_xlabel('derivation $n$ of the group')
    ax.set_ylabel('n° of new objects')
    ax.legend(loc='lower left', ncol=len(strategies), handlelength=2.4)
    fig.tight_layout()
    style.save(fig, base)


def main():
    args = cli.arguments('res_group_novelty_strategies_ci_wide', globals(), ('study2',), ('GROUP_TARGET', 'NOVELTY_POOLS'))
    path = args.paths['study2']
    curves, _ = groups.novelty(path, args.option['GROUP_TARGET'], args.option['NOVELTY_POOLS'])
    figure(curves, cli.output(args, 'res_group_novelty_strategies_ci_wide' + cli.suffix(args), path))


if __name__ == '__main__':
    main()
