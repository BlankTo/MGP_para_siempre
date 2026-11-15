import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from config import DEFAULT
from mgp.analysis import cli, data
from mgp.analysis.datasets import datasets_table
from mgp.analysis.stats import pct

INPUT = DEFAULT
WHERE = DEFAULT
OUTPUT = DEFAULT
TEMPLATE = r'''\begin{table}[!htbp]
\caption{Training sets (Study~3): @CONFIGS@ configurations (ratio 1:1, $e_{sel}=e_{sur}=0.5$, $N_0\in\{128,256\}$), each run on @SETS@ training sets per target with @PER@ derivations per set. Num., Symb.: recovery (\%); Range: smallest and largest numerical recovery of a set (all configurations pooled). Sets matter: configurations (of @CONFIGS@) whose recovery differs between sets more than chance (chi-square test of homogeneity) or whose BBA component coverage does (Kruskal--Wallis test); Holm correction over all configurations and targets.}
\label{tab:res-datasets}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & \multicolumn{4}{c}{Recovery} & \multicolumn{2}{c}{BBA components} \\
\cmidrule(lr){2-5}\cmidrule(l){6-7}
Target & Num. & Symb. & Range & Sets matter & Comp. & Sets matter \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}
'''


def body(recs):
    lines = []
    for r in recs:
        rng = '--' if r['set_max'] == 0 else f"{r['set_min']:.1f}--{r['set_max']:.1f}"
        eff = '--' if r['set_max'] == 0 else f"{r['rec_effect']}/{r['configs']}"
        lines.append(' & '.join([r['target'], pct(r['num']), pct(r['sym']), rng, eff, pct(r['comp']), f"{r['cov_effect']}/{r['configs']}"]) + ' \\\\')
    return lines


def build(frame):
    recs = datasets_table(frame)
    text = TEMPLATE.replace('@CONFIGS@', str(max(r['configs'] for r in recs))).replace('@SETS@', str(max(r['sets'] for r in recs)))
    text = text.replace('@PER@', str(max(r['per_config_set'] for r in recs))).replace('@ROWS@', '\n'.join(body(recs)))
    return recs, text


def main():
    args = cli.arguments('res_datasets', globals(), ('study3',))
    path = args.paths['study3']
    _, text = build(data.where(data.study3(path), args.filters))
    cli.write_text(cli.output(args, 'res_datasets', path), '.tex', text)


if __name__ == '__main__':
    main()
