
# DISCLAIMER: I generated this README with AI, so check it well...

# Overleaf MGP info: https://www.overleaf.com/2958542536dptjtsmkzfms#dd408b

# MGP: single derivation

Python implementation of the derivation rules from *Minimalist Genetic Programming*
by Leonardo Trujillo and the supplied `mgp_info.pdf` table.

## Run

Requires Python 3.10 or later, NumPy and Matplotlib. From the project directory:

```bash
python -m pip install -r requirements.txt
python main.py
python main_visual.py
python main_repeat.py
python main_repeat_stats.py
python main_stats.py
```

| Script | Purpose |
| --- | --- |
| `main.py` | One derivation, with the initial workspace, each MERGE and the best result |
| `main_visual.py` | One interactive derivation; press SPACE to advance the workspace trees |
| `main_repeat.py` | Restart from scratch until a solution is found or `max_derivations` is reached |
| `main_repeat_stats.py` | Complete `derivations` independent trials and print mean best RMSE and exact match count |
| `main_stats.py` | Save workspace evolution, per-derivation results, diversity and plots |

MERGE steps are separated by a visible line. The best semantic object is tracked
throughout each derivation, even if it later leaves the workspace. A derivation
continues after finding RMSE zero; `main_repeat.py` checks the result after each
completed derivation.

The run log shows two workspace views on consecutive lines: the complete MERGE
structure, then the evaluated labels. For example, merging `x1` with `-` and then
with `x2` produces the structure `{{{x1} | {-}} | {x2}}` and the evaluated expression
`(x1 - x2)`. The final and best objects also include their structure.

## Interactive trees

Run `python main_visual.py` from a desktop terminal. Its `VISUAL_CONFIG` uses the
shared defaults with a smaller initial workspace of six variables and four
operators. Change these values inside the script to explore another setup.

The window starts with all workspace atoms. Click inside it and press SPACE to
perform one MERGE. Release SPACE before advancing again. The derivation
waits between steps; it is not computed in advance. Results with infinite fitness
update the workspace and trees like any other result.

Yellow leaves are the original atoms. Blue internal nodes show the evaluated
label at that MERGE, including incomplete expressions and operator labels. The
latest result has a green root. Both operand branches remain visible even when
Minimal Search or the interface condition selects only one label. The toolbar
provides zoom, pan and image saving.

With the default settings, the workspace finishes as one tree. A MERGE limit or
survival elitism can leave several trees. The window displays the stopping reason
and the best RMSE found so far. Closing it stops the interactive run.

The viewer uses the existing Matplotlib dependency. It needs an interactive
desktop backend; it does not force a backend or change the statistics scripts.

## Configuration

`config.py` contains the shared defaults. Each repeated-run script has its own
flat configuration dictionary beginning with `**CONFIG`. Add or change a value
in that dictionary to override a shared default for that script.

| Setting | Meaning |
| --- | --- |
| `seed` | Experiment seed; `None` generates a new seed |
| `dataset_seed` | Independent seed for the sampled dataset |
| `target` | Name from `TARGETS` in `targets.py` |
| `sample_count`, `sample_min`, `sample_max` | Number and interval of uniform input samples; `None` bounds use the target defaults |
| `variable_atom_count`, `operator_count` | Initial workspace occurrence counts |
| `unary_operators_on` | Enable `sin`, `cos`, `log` and `sqrt`; default `False` |
| `binary_operators_on` | Enable the built-in binary operators; default `True` |
| `ternary_operators_on` | Reserved flag; default `False` |
| `fitness_probability` | Probability of choosing by fitness in the interface condition; otherwise random |
| `internal_merge_probability` | Probability of selecting Internal MERGE; default `0.0` |
| `perfect_fit_tolerance` | Maximum RMSE counted as a solution; default `0.0` |
| `max_merge` | Maximum MERGE operations in one derivation, including results with infinite fitness |
| `esel` | Probability of selecting the best accessible workspace semantic as the left operand |
| `esur` | Probability of preserving the selected best workspace semantic |
| `archive_enabled` | Enable recording selected semantic MERGE results |
| `archive_max_size` | Maximum archive size; `None` means unlimited |
| `archive_similarity` | Cosine similarity threshold, default `0.99` |
| `archive_atomic_objects` | Allow atomic semantic objects in the archive |

The symbols `+`, `-`, `*`, `/` are defined as `BINARY_OPERATORS`, and
`sin`, `cos`, `log`, `sqrt` as `UNARY_OPERATORS`, in `mgp/objects.py`.
They are not configuration lists.
The flags control which operators can enter a generated workspace. With both
groups enabled, each operator occurrence is sampled uniformly from the eight
symbols. `operator_count` is the total count across enabled groups.

The default keeps unaries disabled and reproduces the binary-only behavior.
To try a unary target, change these entries in `config.py` or in a script's own
configuration dictionary:

```python
"target": "sine",
"unary_operators_on": True,
```

`sine` represents `sin(x1)`. `polynomial_sine` represents
`x1 * x1 + x1 + sin(x1)`. To generate unary operators only, also set
`binary_operators_on` to `False`. If all operator groups are disabled, set
`operator_count` to zero. Ternary operators remain a stub: enabling their flag
raises `NotImplementedError`.

Direct MERGE examples create their operands explicitly, so `examples.py` shows
both arities independently of workspace-generation flags.

`InternalMerge.merge(current)` in `mgp/merge.py` is an explicit stub. The derivation
selects Internal MERGE with probability `internal_merge_probability` and External
MERGE otherwise. Selecting the internal branch currently raises
`NotImplementedError`. With the default probability of zero, only External MERGE
is selected and no extra random draw is made for this choice.

Variable names come from the target function's positional arguments. Add a target
by defining a function in `targets.py` and adding its name, function and readable
expression to `TARGETS`.

## Nguyen targets

`targets.py` includes Nguyen-1 through Nguyen-10 and Nguyen-12. The definitions
below follow Table 1 of Trujillo's *Minimalist Genetic Programming*, including
its choices of constants and coefficients. Nguyen-11, `x1**x2`, is excluded.

| Target | Expression |
| --- | --- |
| `nguyen_1` | `x1 + x1**2 + x1**3` |
| `nguyen_2` | `x1 + x1**2 + x1**3 + x1**4` |
| `nguyen_3` | `x1 + x1**2 + x1**3 + x1**4 + x1**5` |
| `nguyen_4` | `x1 + x1**2 + x1**3 + x1**4 + x1**5 + x1**6` |
| `nguyen_5` | `sin(x1**2) * cos(x1)` |
| `nguyen_6` | `sin(x1) + sin(x1 + x1**2)` |
| `nguyen_7` | `log(x1 + 1) + log(x1**2)` |
| `nguyen_8` | `sqrt(x1)` |
| `nguyen_9` | `sin(x1) + sin(x2**2)` |
| `nguyen_10` | `sin(x1) * cos(x2)` |
| `nguyen_12` | `x1**4 - x1**3 + x2**2 - x2` |

For example, set these entries in the shared or script-specific configuration:

```python
"target": "nguyen_7",
"unary_operators_on": True,
"sample_min": None,
"sample_max": None,
```

`TARGET_DOMAINS` defines the default intervals. Each bound set to `None` uses
that target's default. Explicit numeric bounds take precedence.

- Nguyen-1 to Nguyen-6, Nguyen-9, Nguyen-10 and Nguyen-12: `[-5, 5]`.
- Nguyen-7: `[1e-6, 5]`, a positive lower bound to avoid the logarithm at zero.
- Nguyen-8: `[0, 5]`, including the valid square root of zero.
- Previous targets and custom functions without a listed domain: `[-20, 20]`.

The paper lists `(0, 5]` for Nguyen-7 and Nguyen-8. The practical bounds above
make endpoint handling explicit. Dataset generation keeps the project's uniform
random sampling, using `sample_count` points and independent input variables.
The paper's equidistant grids and two-dimensional meshes are a separate
experimental protocol. Statistics save the resolved numeric bounds and the
actual dataset, so a run can be replayed even when the input settings use `None`.

## Repeated derivations

`main_repeat.py` has `max_derivations` as its stopping guard. It reports whether
it stopped after a solution or after exhausting that limit.

`main_repeat_stats.py` always runs its configured number of `derivations`. Its
mean best RMSE averages the best error reached within each trial. Exact matches
count trials with best RMSE exactly zero, once per trial. A positive tolerance
can count additional approximate solutions, but they are reported separately
from exact matches.

Both scripts generate the dataset once, then create a new workspace and empty
archive for every trial. They print the effective seeds, the best expression and
the seeds of the trial that produced it. Means use finite fitnesses only; the
number of contributing trials is printed. If none has a finite fitness, the
mean is `None`.

A match means agreement on the sampled dataset, not a proof of symbolic
identity. Repetition does not guarantee a solution within a finite attempt limit.

## Statistics files

`main_stats.py` adds these settings:

- `derivations`: number of independent derivations.
- `output_dir`: parent directory for a new timestamped results directory.
- `progress_every`: print progress and ETA every this many trials; zero disables progress.
- `max_value_pairs`: cap for output-vector distance pairs; `None` evaluates all pairs.
- `save_plots`: save the two SVG plots.

| Output | Content |
| --- | --- |
| `dataset.csv` | Actual sampled inputs and target values |
| `derivation_matrices/derivation_XXXXXX.csv` | Five counts after every MERGE of that trial |
| `derivation_stats.csv` | Trial seeds, MERGE counts, final type, best/final expressions and fitnesses, solution flags, archive size and diversity |
| `derivation_average_matrix.csv` | Mean workspace counts, including the initial workspace at row zero |
| `derivation_variance_matrix.csv` | Population variance of those counts |
| `derivation_standard_deviation_matrix.csv` | Standard deviation of those counts |
| `derivation_normalized_*_matrix.csv` | The same statistics after normalizing each trial's row by its current workspace size |
| `derivation_contributors.csv` | Number of trials contributing to each step, including step zero |
| `derivation_average_standard_deviation.svg` | Mean counts and standard deviation bands |
| `derivation_normalized_average_standard_deviation.svg` | Mean proportions and standard deviation bands |
| `summary.json` | Effective configuration, aggregate results and diversity |

The five count columns are `atomic_semantic`, `composite_semantic`,
`atomic_operator`, `composite_operator`, and `incomplete`. Each per-trial matrix
still has exactly one row per MERGE; it does not include the initial
workspace. Aggregate matrices and plots additionally include step zero.

A completed trial does not contribute to later steps. For example, if 100 trials
start but only 30 reach step 80, that step's mean and variance use those 30 trials.
`derivation_contributors.csv` records this sample count. Empty workspaces have
zero counts and zero normalized values.

Plots start at step zero and end at the last recorded MERGE, without horizontal
padding. Count axes start at zero; proportion axes run from zero to one. Bands
show mean plus/minus standard deviation, clipped to the valid display range.
The original variance remains in the CSV files. If no trial performs a
MERGE, no evolution plots are saved.

Matplotlib selects its backend automatically. The statistics script saves SVG
vector graphics without opening a window; only `main_visual.py` calls `plt.show()`.

The compact per-trial CSV omits duplicated initial/final category counts, best
location/source details, and other values already present in the matrices or
available from the main run log. The JSON omits prose definitions, environment
versions, code hashes and duplicated configuration blocks.

A final object exists only when the workspace contains exactly one object. A
workspace with multiple objects is reported as `multiple`; its final expression
and fitness are empty. A semantic object with nonfinite outputs has fitness
`inf`, which is written explicitly in the per-trial CSV. Missing values are
empty in CSV and `null` in JSON. Mean best fitness uses finite fitnesses only;
`finite_best_count` records how many trials contribute to that mean.

## Diversity

Distances are computed separately for the best objects and the final semantic
objects:

- Absolute difference between their RMSE fitnesses relative to the target.
- RMSE between their output vectors on the shared dataset.

Fitness distances use only finite fitnesses and are computed exactly from sorted
values. Output distances
use all pairs up to `max_value_pairs`; above that cap, pairs are sampled uniformly
without replacement. The summary records evaluated/available pairs, the sampling
seed and any nonfinite distances. The CSV records each object's sampled pair
count, so a mean based on few pairs is identifiable. A mean output distance
containing a nonfinite pair is unavailable, rather than silently excluding that pair.

## MERGE behavior and efficiency

Operator objects have arity one or two. A composite operator head contains
candidates of a single arity. Incomplete objects accept only binary heads.

Minimal Search takes precedence over the arity rule. For example, an atomic `+`
merged with the composite head `[sin, cos]` keeps `+`. For equally accessible
operator heads of different arities, the lower arity survives: `sin` merged with
`+` keeps `sin`. Equally accessible heads of the same arity combine their
candidate lists, retaining duplicate occurrences.

A unary head applied to a semantic object immediately produces a composite
semantic object. A composite unary head evaluates its candidates and uses the
existing fitness/random interface condition. Merging a unary operator head with
an incomplete object replaces its binary head and applies the unary candidates
to its semantic component.

The paper's unary nesting rule is implemented: merging `cos` with `sin(x1)`
produces `cos(x1)`. The outer function replaces the immediately nested unary
function, while the displayed MERGE history preserves both operator leaves.
A binary expression remains one operand, so applying `cos` to `sin(x1) + x1`
produces `cos((sin(x1) + x1))`.

`log` is the natural logarithm, with a real finite result for positive arguments.
`sqrt` returns the real square root and accepts zero and positive arguments.
Out-of-domain inputs, division by zero and overflow retain their ordinary NumPy
results, including `nan` and `inf`. If any output is nonfinite, the expression's
RMSE is `inf`. It is still a semantic object: the MERGE updates the workspace,
records a normal step and continues the derivation.

The nesting rule also applies to these operators: merging `log` with `sqrt(x1)`
produces `log(x1)`. Evaluation uses the argument after this replacement. For
example, a positive `sin(x1)` value does not make `log` valid if the original
`x1` is nonpositive. The functions use their ordinary real definitions without
absolute values or offsets.

Unary results keep a reference to their evaluated argument for this rule. Its
numeric vector is reused, without parsing expression strings or recalculating
the argument. This information survives semantic selection and incomplete
objects. When constructing composite inputs manually, build unary expressions
through `ExternalMerge` so that this argument is available.

Semantic MERGE results are composite according to the supplied table, even when
their selected expression is a single variable. An atomic operator selected by
Minimal Search remains atomic. Composite semantics remain single operands;
subtraction and division follow the current MERGE's left and right branches.

Operator occurrences are retained, including duplicates. Each distinct operator
is evaluated only once per MERGE, while duplicate occurrences retain their
probability in random selection. All candidates participate in the interface
condition. Fitness selection prefers finite fitness over `inf`; random selection
can choose either. If every candidate has infinite fitness, one is still selected.
A derivation may therefore finish with infinite fitness while preserving an
earlier, better expression as its best result.

Output vectors are reused. Each semantic object caches its latest fitness for
an owned, read-only target and output vector. Direct calls with mutable arrays
or NumPy views recompute the fitness. Workspace counts are updated incrementally,
and a workspace-wide best search runs only when elitism needs it.

Each object also has a lightweight MERGE tree containing labels and references
to child trees. It keeps the syntactic history without retaining old numeric
vectors or re-evaluating expressions. This history is used for display. Workspace
categories and selection still use the active object and its evaluated label.
An externally supplied composite object with no recorded history appears as one
label; the program does not reconstruct missing branches from expression text.

`Derivation.steps()` yields the initial workspace and then one workspace per
attempt. `Derivation.run()` consumes the same iterator to completion. Each yielded
workspace is a shallow list copy, while objects and numeric arrays are shared.
The viewer uses this iterator and keeps its drawing code in `main_visual.py`.

The archive records selected semantic results without feeding them back into
construction. It excludes nonfinite vectors, for which cosine similarity is
undefined, constant output vectors and vectors whose cosine similarity reaches
the threshold. Opposite vectors remain distinct. Results with poor fitness do
not remove earlier archived expressions or the best result. The explicit
`archive` and `input_so` arguments of `Derivation.run` can provide an initial
best; all supplied objects must use the same dataset and sample order.
