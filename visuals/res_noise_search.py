import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from config import DEFAULT
from mgp.analysis import cli, data
from mgp.analysis.conditions import noise_table, select

INPUT = DEFAULT
WHERE = DEFAULT
OUTPUT = DEFAULT
TEMPLATE = r'''\begin{table}[!htbp]
\caption{Search sensitivity to training noise $\sigma/\mathrm{RMS}(y)$ (@N@ derivations per level). Identical: unchanged trajectories; First divergence: median first differing $MERGE$; Changed IC/elite: differing choices; Overlap: mean Jaccard similarity of evaluated expressions (\%).}
\label{tab:res-noise-search}
\centering\scriptsize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Noise & Identical & First divergence & Changed IC & Changed elite & Overlap \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}
'''


def build(res):
    out, lines = noise_table(res)
    return out, lines, TEMPLATE.replace('@N@', cli.thousands(out['pairs'])).replace('@ROWS@', '\n'.join(lines))


def main():
    args = cli.arguments('res_noise_search', globals(), ('study4',))
    path = args.paths['study4']
    _, _, text = build(select(data.study4(path), args.filters))
    cli.write_text(cli.output(args, 'res_noise_search', path), '.tex', text)


if __name__ == '__main__':
    main()
