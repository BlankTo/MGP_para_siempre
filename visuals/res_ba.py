import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from config import DEFAULT
from mgp.analysis import cli, single
from mgp.analysis.stats import design_rows, short

INPUT = DEFAULT
WHERE = DEFAULT
REFERENCE = DEFAULT
OUTPUT = DEFAULT
TEMPLATE = r'''\begin{table}[t]
\caption{Single-derivation archive contents and coverage (@N@ derivations per row). Values are averaged across targets. With no archive limit, BA/BBA coincide. Bold: significant paired differences from the reference, in the same direction on every target.}
\label{tab:res-ba}
\centering\scriptsize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}lrrrrrrrrrr@{}}
\toprule
 & \multicolumn{5}{c}{BA and BBA} & \multicolumn{5}{c}{FBBA} \\
\cmidrule(lr){2-6}\cmidrule(l){7-11}
 & Ent. & Comp. & Sets & Compos. & Deg$>$3 & Ent. & Comp. & Sets & Compos. & Deg$>$3 \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}
'''


def body(T, rows):
    return [' & '.join([short(f, v)] + T['BA']['cells'][i] + T['FBBA']['cells'][i]) + ' \\\\' for i, (f, v) in enumerate(rows)]


def build(S1, ref):
    rows = design_rows(S1, ref)
    T = single.tables(S1, rows, ref)
    text = TEMPLATE.replace('@N@', cli.thousands(T['targets'] * T['repetitions'])).replace('@ROWS@', '\n'.join(body(T, rows)))
    return T, rows, text


def main():
    args = cli.arguments('res_ba', globals(), ('study1',))
    path = args.paths['study1']
    _, _, text = build(single.load(path, args.filters), args.ref)
    cli.write_text(cli.output(args, 'res_ba', path), '.tex', text)


if __name__ == '__main__':
    main()
