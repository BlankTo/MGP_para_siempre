import string
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from matplotlib.lines import Line2D

from config import DEFAULT
from mgp.analysis import cli, single, style
from mgp.analysis.stats import SN
from mgp.analysis.style import GOALS, INK2, LEG_MS, LS, LW, MS, W, plt

INPUT = DEFAULT
WHERE = DEFAULT
OUTPUT = DEFAULT
LINES = [('nguyen_1', 'composite', 'N1 composites'), ('nguyen_4', 'component', 'N4 components'), ('nguyen_4', 'cooccurrence', 'N4 component sets')]


def figure(res, base, lines=LINES):
    style.use('paper')
    colour = {g: (c, mk) for g, _, c, _, mk in GOALS}
    strategies, ratios, ims = res['strategies'], res['ratios'], res['im']
    lines = [x for x in lines if any(k.startswith(f'{x[0]}|{x[1]}|') for k in res)]
    fig, axes = plt.subplots(1, len(strategies), figsize=(W, 1.5), sharey=True, squeeze=False)
    for ax, s, letter in zip(axes[0], strategies, string.ascii_lowercase):
        for t, g, _ in lines:
            c, mk = colour[g]
            for im in ims:
                ax.plot(range(len(ratios)), res[f'{t}|{g}|{single.lab(s, im)}'], color=c, ls=LS.get(im, '-'), marker=mk, ms=MS, lw=LW)
        ax.set_title(f'({letter}) {SN[s]}', loc='left')
        ax.set_xticks(range(len(ratios)), ratios, fontsize=6.5)
        ax.set_xlabel('operator : variable', fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
    axes[0, 0].set_ylabel('BBA coverage\nper derivation (%)')
    fig.tight_layout(w_pad=0.5)
    h = [Line2D([], [], color=colour[g][0], marker=colour[g][1], ms=LEG_MS, lw=LW) for _, g, _ in lines]
    h += [Line2D([], [], color=INK2, ls=LS.get(im, '-'), lw=LW) for im in ims]
    fig.legend(h, [x for _, _, x in lines] + [f'$p_{{IM}}={im:g}$' for im in ims], loc='lower center', bbox_to_anchor=(0.5, 0.98),
               ncol=len(lines) + len(ims), handlelength=2.4, columnspacing=1.0, fontsize=6.6)
    style.save(fig, base)


def main():
    args = cli.arguments('res_ratio_coverage_strategies_short', globals(), ('study1',))
    path = args.paths['study1']
    figure(single.ratio_coverage(single.load(path, args.filters)), cli.output(args, 'res_ratio_coverage_strategies_short', path))


if __name__ == '__main__':
    main()
