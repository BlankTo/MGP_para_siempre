import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from config import DEFAULT
from mgp.analysis import cli, data
from mgp.analysis.conditions import labels_table

INPUT = DEFAULT
WHERE = DEFAULT
OUTPUT = DEFAULT
TEMPLATE = r'''\begin{table}[t]
\caption{Uninformative training (@N@ paired derivations per condition). Pure noise: $\mathrm{RMS}(y)\,z$ with $z\sim\mathcal{N}(0,1)$; random IC: random choices; inverted: reciprocal error. Best/worst select the lowest/highest error from the same derivations. Overlap: Jaccard similarity with noise-free search; coverage averaged across targets (BA/BBA: BBA shown). Values in \%.}
\label{tab:res-labels}
\centering\scriptsize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}lrrrrrrrrrr@{}}
\toprule
 & \multicolumn{3}{c}{Target (\%)} & & \multicolumn{3}{c}{BA \& BBA (\%)} & \multicolumn{3}{c}{FBBA (\%)} \\
\cmidrule(lr){2-4}\cmidrule(lr){6-8}\cmidrule(l){9-11}
Condition & Built & Num. & Symb. & Overlap & Comp. & Co-occ. & Compos. & Comp. & Co-occ. & Compos. \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}
'''


def build(frame):
    out, lines = labels_table(frame)
    n = out['clean']['derivations'] if 'clean' in out else max(o['derivations'] for o in out.values())
    return out, lines, TEMPLATE.replace('@N@', cli.thousands(n)).replace('@ROWS@', '\n'.join(lines))


def main():
    args = cli.arguments('res_labels', globals(), ('study5',))
    path = args.paths['study5']
    _, _, text = build(data.where(data.study5(path), args.filters))
    cli.write_text(cli.output(args, 'res_labels', path), '.tex', text)


if __name__ == '__main__':
    main()
