import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from config import DEFAULT
from mgp.analysis import cli, single
from mgp.analysis.stats import design_rows, num, short

INPUT = DEFAULT
WHERE = DEFAULT
REFERENCE = DEFAULT
OUTPUT = DEFAULT
TEMPLATE = r'''\begin{table}[t]
\caption{Target recovery in groups of @N@ derivations. Rows vary one reference parameter. Built/Num./Symb.: construction and numerical/symbolic recovery; Train/Test: mean RMSE of the group's best expression.}
\label{tab:res-group-recovery}
\centering\scriptsize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}llllrr@{}}
\toprule
 & & \multicolumn{2}{c}{Recovery} & \multicolumn{2}{c}{RMSE} \\
\cmidrule(lr){3-4}\cmidrule(l){5-6}
Configuration & Built & Num. & Symb. & Train & Test \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}
'''


def body(T, lists, rows):
    labels = ['Reference'] + [short(f, v) for f, v in rows[1:]]
    return [' & '.join([label, ', '.join(g['built']), ', '.join(g['num']), ', '.join(g['sym']), num(r['train']), num(r['test'])]) + ' \\\\'
            for label, g, r in zip(labels, lists, T['groups'])]


def build(S1, ref):
    rows = design_rows(S1, ref)
    T = single.tables(S1, rows, ref)
    lists = single.group_lists(S1, rows, ref)
    text = TEMPLATE.replace('@N@', cli.thousands(T['repetitions'])).replace('@ROWS@', '\n'.join(body(T, lists, rows)))
    return T, lists, text


def main():
    args = cli.arguments('res_group_recovery', globals(), ('study1',))
    path = args.paths['study1']
    _, _, text = build(single.load(path, args.filters), args.ref)
    cli.write_text(cli.output(args, 'res_group_recovery', path), '.tex', text)


if __name__ == '__main__':
    main()
