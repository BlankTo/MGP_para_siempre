import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from config import DEFAULT
from mgp.analysis import cli, data, groups
from mgp.analysis.stats import num

INPUT = DEFAULT
WHERE = DEFAULT
OUTPUT = DEFAULT
TEMPLATE = r'''\begin{table}[!htbp]
\caption{Groups of @SIZE@ derivations, recovery (Study~7; one configuration per strategy: $N_0=128$, ratio 1:1, $p_{IM}=0.5$, $e_{sel}=e_{sur}=0.5$ for Standard and Revisit IC and $e_{sel}=e_{sur}=0$ for Constructive; @GROUPS@ groups per target and strategy). Derivations: rates over the @TOTAL@ derivations (\%). Groups: groups (of @GROUPS@) in which at least one derivation builds the target or recovers it numerically or symbolically. RMSE: training error of the best derivation of a group and its test error, mean over the @GROUPS@ groups.}
\label{tab:res-group-recovery-5000}
\centering\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}llrrrrrrrr@{}}
\toprule
 & & \multicolumn{3}{c}{Derivations (\%)} & \multicolumn{3}{c}{Groups} & \multicolumn{2}{c}{RMSE} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-10}
Target & Strategy & Built & Num. & Symb. & Built & Num. & Symb. & Train & Test \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}
'''


def rate(x):
    return '0.0' if x == 0 else f'{x:.3f}' if x < 0.1 else f'{x:.1f}'


def body(recs):
    lines, last = [], None
    for r in recs:
        if last is not None and r['target'] != last:
            lines.append('\\midrule')
        lines.append(' & '.join([r['target'] if r['target'] != last else '', r['strategy'], rate(r['built']), rate(r['num']), rate(r['sym']),
                                 str(r['groups_built']), str(r['groups_num']), str(r['groups_sym']), num(r['best_train']), num(r['best_test'])])
                     + ' \\\\')
        last = r['target']
    return lines


def build(frame):
    recs = groups.recovery7(frame)
    size = int(frame.groupby(['target', 'strategy', 'pool']).size().max())
    count = max(r['groups'] for r in recs)
    text = TEMPLATE.replace('@SIZE@', cli.thousands(size)).replace('@GROUPS@', str(count)).replace('@TOTAL@', cli.thousands(size * count))
    return recs, text.replace('@ROWS@', '\n'.join(body(recs)))


def main():
    args = cli.arguments('res_group_recovery_5000', globals(), ('study7',))
    path = args.paths['study7']
    _, text = build(data.where(data.study7(path), args.filters))
    cli.write_text(cli.output(args, 'res_group_recovery_5000', path), '.tex', text)


if __name__ == '__main__':
    main()
