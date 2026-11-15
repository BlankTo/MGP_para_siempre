import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

from config import DEFAULT
from mgp.analysis import cli, data
from mgp.analysis.datasets import training_table
from mgp.analysis.stats import ARCH, pct

INPUT = DEFAULT
WHERE = DEFAULT
OUTPUT = DEFAULT
TEMPLATE = r'''\begin{table}[!htbp]
\caption{Training-set size. Each derivation is repeated with @SMALLER@ of its @REF@-point training set (same seeds; noise-free training data; @CONFIGS@ configurations with ratio 1:1 and $N_0\in\{128,256\}$, @TARGETS@ targets, @REPS@ repetitions: @N@ derivations per row). Numerical and symbolic recovery (\%) on the 1{,}000 test points; component coverage (\%) of the three archives.}
\label{tab:res-training}
\centering\footnotesize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Training points & Numerical & Symbolic & BA & BBA & FBBA \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}
'''


def body(table):
    ref = table['values'][0]['n']
    return [' & '.join([f'{ref} (reference)' if r['n'] == ref else f"{ref} $\\to$ {r['n']}"] + [pct(r[k]) for k in ('num', 'sym', *ARCH)]) + ' \\\\'
            for r in table['values']]


def build(frame):
    table = training_table(frame)
    sizes = [r['n'] for r in table['values']]
    text = TEMPLATE.replace('@SMALLER@', ' and '.join(f'the first {n}' for n in sizes[1:]) + ' points').replace('@REF@', str(sizes[0]))
    text = text.replace('@CONFIGS@', str(frame.config_id.nunique())).replace('@TARGETS@', str(frame.target.nunique()))
    text = text.replace('@REPS@', str(frame.rep.nunique())).replace('@N@', cli.thousands(table['values'][0]['derivations']))
    return table, text.replace('@ROWS@', '\n'.join(body(table)))


def main():
    args = cli.arguments('res_training', globals(), ('study8',))
    path = args.paths['study8']
    _, text = build(data.where(data.study8(path), args.filters))
    cli.write_text(cli.output(args, 'res_training', path), '.tex', text)


if __name__ == '__main__':
    main()
