import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from config import DEFAULT
from mgp.analysis import cli, style, timing
from mgp.analysis.stats import PAR, SN, fmt_val
from mgp.analysis.style import INK, W, plt

INPUT = DEFAULT
WHERE = DEFAULT
REFERENCE = DEFAULT
OUTPUT = DEFAULT
GROUP_TARGET = DEFAULT
TIME = DEFAULT
DIVERSITY = DEFAULT
CMAP = LinearSegmentedColormap.from_list('blue', ['#e8f1fb', '#9cc3ee', '#2a78d6', '#0d3f7a'])


def label(ref, f, v):
    return 'reference' if f is None else f'{PAR[f]}: {fmt_val(f, ref[f])} → {fmt_val(f, v)}'


def figure(M, strategies, cols, ref, base):
    style.use('grid')
    pos = M[M > 0] if (M > 0).any() else np.ones(1)
    norm = LogNorm(vmin=pos.min() * 0.8, vmax=pos.max() * 1.05, clip=True)
    fig, ax = plt.subplots(figsize=(W, 2.35))
    ax.imshow(M, cmap=CMAP, norm=norm, aspect='auto')
    for i in range(len(strategies)):
        for j in range(len(cols)):
            v = M[i, j]
            if np.isnan(v):
                continue
            rgb = CMAP(norm(v))[:3]
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            ax.text(j, i, f'{v:.0f}' if v >= 10 else f'{v:.1f}', ha='center', va='center', fontsize=7.5, color='white' if lum < 0.5 else INK)
    ax.set_xticks(range(len(cols)), [label(ref, f, v) for f, v in cols], fontsize=7, rotation=35, ha='left', rotation_mode='anchor')
    ax.set_yticks(range(len(strategies)), [SN[s] for s in strategies])
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(strategies), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    ax.xaxis.tick_top()
    ax.set_xlabel('mean CPU time per derivation (ms)', fontsize=7.5, labelpad=4)
    style.save(fig, base)


def main():
    args = cli.arguments('res_timing', globals(), ('study6',), ('GROUP_TARGET', 'TIME', 'DIVERSITY'))
    path = args.paths['study6']
    o = args.option
    x = timing.frame(path, o['GROUP_TARGET'], o['TIME'], args.filters, o['DIVERSITY'])
    M, strategies, cols = timing.heatmap(x, args.ref)
    figure(M, strategies, cols, args.ref, cli.output(args, 'res_timing' + cli.suffix(args), path))


if __name__ == '__main__':
    main()
