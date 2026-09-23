# MGP v15: focused study reports

Python implementation of the derivation rules from *Minimalist Genetic Programming*
by Leonardo Trujillo and the supplied `mgp_info.pdf` table.

## Reporting changes in v15

The report now contains only the requested figures, in PNG and SVG. Open
`report/index.html` to browse them. CSVs are in `report/data/`, JSON metadata in
`report/metadata/`, and guides and findings in `report/docs/`.
[PLOT_GUIDE.md](PLOT_GUIDE.md) lists every figure, definition and source table.

```text
python main_study_report.py STUDY_DIRECTORY
python main_study_report.py STUDY_DIRECTORY --output REPORT_DIRECTORY
```

New figures cover all six hyperparameter pairs, global runtime and a 3D surface
of materialized subtarget coverage. The workspace plots now use all runs grouped
by each hyperparameter value. Per-step means are weighted by the actual number
of contributing runs; undefined pair means stay missing. The curve CSV includes
both workspace and comparison contributor counts. Stopped runs are not extended.

Global summaries retain equal target weights and complete repetition seed blocks.
The effects plot shows success and runtime only. Global heatmaps still include
normalized error (RMSE / target RMS), as labelled. Probe diagnostics and paired
comparisons remain in the tables and written report. Known retired images are
removed when regenerating an old report; unrelated files are preserved.

The diversity accumulator uses compensated summation to limit cancellation drift
and bounds its reported mean to the permitted comparison range. This changes
statistics only, not selection, fitness, MERGE, archive admission or random draws.
The default absolute target/subtarget tolerance remains `1e-8`. No dependencies
were added and the result database schema is unchanged.

`--configurations` now selects only detailed step/histogram CSVs. The new workspace
curves always pool all configurations of the target. `--baseline` selects the
reference for the paired tables and the global success matrix.

**Resume older experiments with their original code/environment.** Resume checks
the full code fingerprint. v15 can report existing results or start new studies.
The working study's `study.json`, `study.sqlite3` and `datasets/` keep their paths.

## Run

Requires Python 3.10 or later, NumPy and Matplotlib. From the project directory:

```bash
python -m pip install -r requirements.txt
python main.py
python main_visual.py
python examples.py
python main_repeat.py
python main_repeat_stats.py
python main_stats.py
python main_study_example.py
python -m unittest discover -s tests
```

| Script | Purpose |
| --- | --- |
| `main.py` | One derivation, with the initial workspace, each MERGE and the best result |
| `main_visual.py` | One interactive derivation; press SPACE to advance the workspace trees |
| `examples.py` | Small examples of the MERGE rules and operand order |
| `main_repeat.py` | Restart from scratch until a solution is found or `max_derivations` is reached |
| `main_repeat_stats.py` | Complete `derivations` independent trials and print mean best RMSE and exact match count |
| `main_stats.py` | Save workspace evolution, per-derivation results, diversity and plots |
| `main_internal.py` | Small Internal MERGE example, using the same derivation engine |
| `main_hyperparameters.py` | Full EM-only experiment, or resume a saved experiment |
| `main_study_example.py` | Small study through the same runner, followed by CSV export |
| `main_study_export.py` | Load and summarize saved runs, optionally filtering target/configuration |
| `main_study_report.py` | Create English figures, paired comparisons and an interpretation report |
| `main_replay.py` | Replay a saved run in the terminal or interactive viewer |

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
| `internal_merge_probability` | Probability of attempting IM when both operations are possible; default `0.0` |
| `esur_in_IM` | Apply existing elite survival to the selected IM parent; default `False` |
| `IM_terminate_on_one` | Stop on a single workspace root; default `True` |
| `perfect_fit_tolerance` | Static maximum RMSE counted as a solution; default `1e-8` |
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

## Internal MERGE

IM chooses a top-level parent with the same selection rule as EM: `esel` selects
the best current workspace semantic, with the existing first-in-workspace tie rule.
Otherwise it selects a root uniformly. It then selects uniformly among the
accessible internal occurrences of that parent. If that parent has no candidate,
the step falls back to EM using the already selected root; it does not resample
roots until finding a composite one. The archive is never a source of IM candidates.

Accessibility is recursive through all object classes, including composite
semantics. Paths identify occurrences, so two positions containing the same Python
object remain distinct. IM forms `{selected, parent}` and uses `ExternalMerge.merge`
for construction, labeling, MS, IC, arity, operand order and evaluation. Only the
selected lower occurrence becomes inaccessible. Its descendants, siblings and the
new upper occurrence remain eligible. Original objects, saved trees and surviving
workspace roots are not mutated.

The engine retains references to child objects and cached accessible-occurrence
counts. Selecting a uniform random rank through those counts avoids creating a
large list of occurrences at every step. Numeric vectors are shared, not evaluated
again. `InternalMerge.candidates()` can still enumerate paths for small inspections.

Normally IM replaces one parent with one result. With `esur_in_IM=True`, the
existing `esur` probability preserves that parent if it is the current semantic
elite; the result is then added, increasing workspace size by one. An internal
child is never independently removed from the workspace. With the flag disabled,
IM ignores survival. EM survival is unchanged.

Every performed EM or IM counts once toward `max_merge` (default 1000). With
`IM_terminate_on_one=True`, reaching a single root ends the derivation. If the flag
is false and IM has positive probability, the remaining root can continue through
IM. At this boundary EM is impossible, so the engine uses the only available
operation; the configured IM/EM probabilities apply while both types can be
attempted. An atomic or otherwise ineligible last root ends with `no_valid_merge`.
An IM probability of zero retains EM-only stopping and random-number behavior.

Large IM trees can still be expensive to print or draw in full. The numerical
engine does not need to expand their entire representation for selection.

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
- G1 and G2 outside the dedicated study: `[-5, 5]`.
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

## Hyperparameter study

`main_hyperparameters.py` has a complete, explicit `STUDY_CONFIG`. It does not
inherit mutable defaults from `CONFIG`. This experiment uses seven targets:
G1, Nguyen-1 through Nguyen-4, G2 and Nguyen-12. G1 is `x1 + x1**2`; G2 is
`x1**2 + x2**2`. G3 is excluded. IM and unary operators are disabled; binary
operators and fitness IC are enabled. `max_merge=1000` and target RMSE tolerance
is `1e-8`. Each run starts with new syntactic objects, best state and archive.
No archive feeds later runs, and no outer evolutionary loop is implemented.

The default grid is:

| Parameter | Values |
| --- | --- |
| `esel` | 0, 0.2, 0.4, 0.6, 0.8, 1 |
| `esur` | 0, 0.2, 0.4, 0.6, 0.8, 1 |
| `variable_atom_count` | 8, 16, 32, 64, 128, 256, 512 |
| `operator_count` | 4, 8, 16, 32, 64, 128, 256 |

`variable_atom_count` counts variable occurrences, not distinct inputs. `esur`
is a probability. These are the only sweep axes of this study. The default
`repetitions=10000` gives 1764 configurations and 123,480,000 runs. The smoke
script reduces targets, repetitions, workspace sizes and MERGE limits; it calls
exactly the same `run_study` function. Do not use the full study as a quick test.

### Fixed datasets and seeds

The study builds each target dataset once. One-dimensional targets use 100
equally spaced points on [-20, 20]. Two-dimensional targets use a 10 by 10
Cartesian mesh, with `indexing="ij"` and 100 total samples. For smaller tests,
`sample_count` may change, but must be a perfect square for two-dimensional
targets. `sample_min` and `sample_max` are explicit study settings.

`dataset_seed` is resolved and recorded independently. These deterministic grids
contain no random sampling, so changing that seed does not change the grid.
Legacy scripts still use their existing random-uniform datasets.

A master `seed` generates an ordered list of distinct repetition seeds. Each
repetition seed generates separate workspace and derivation seeds. Every
configuration receives the same pair for that repetition, regardless of how many
random draws workspace construction needs. All targets reuse the same list;
paired comparisons should be made within one target. No Python `hash()` is used
for seeds. `seed=None` generates and saves a reproducible effective seed.

Execution order is repetition, target, configuration: all jobs for one seed run
before moving to the next. Finding a target records its first hit but never ends
a derivation early. Ordinary EM termination and the MERGE limit still apply.

### Run, stop and resume

From the project directory:

```bash
python main_study_example.py
python main_hyperparameters.py --max-runs 100
python main_hyperparameters.py --resume "hyperparameter_output/STUDY_DIRECTORY"
python main_hyperparameters.py --resume "hyperparameter_output/STUDY_DIRECTORY" --max-runs 100
python main_study_export.py "hyperparameter_output/STUDY_DIRECTORY"
python main_study_report.py "hyperparameter_output/STUDY_DIRECTORY"
python main_study_report.py "hyperparameter_output/STUDY_DIRECTORY" --target g1 --baseline 1 --configurations 1 2 4
python main_study_export.py "hyperparameter_output/STUDY_DIRECTORY" --target nguyen_1 --configuration 1
python main_replay.py "hyperparameter_output/STUDY_DIRECTORY" --run 1 --visual
```

Running without `--max-runs` requests the full configured study. Ctrl+C stops it.
`--max-runs` limits additional completed runs in that invocation, not the total
study size. Resume loads the saved configuration and seed assignments, including
when the original seeds were random. Editing current script defaults does not
alter a resumed experiment. Changed core code, comparison/equivalence methods,
Python/NumPy versions, metadata or saved datasets cause an explicit refusal to
mix incompatible results. Keep the project version used to start a long study.

`study.sqlite3` uses ordinary Python `sqlite3`, without additional dependencies.
One transaction saves the run, subtarget hits, archive-subtarget hits, diagnostics, step aggregates, optional trace and
explicit progress cursor. A committed row has `completed=1`. The unique key
`(target, configuration_id, repetition)` resolves to the requested full key
`(target, esel, esur, variable_atom_count, operator_count, seed)` through immutable
configuration and seed tables. The `completed_runs` view exposes that full key.
Run IDs and completion state are explicit; resume does not count CSV lines.

Interrupted transactions roll back. Completed runs are skipped; only an unfinished
run may be recomputed, with exactly its assigned seeds. Resume can be repeated.
Execution is sequential with one writer per study directory. Stop the writer
before copying a study directory; retain its dataset and metadata files with the
database. CSV export reads a consistent snapshot and does not need to hold all
runs in memory.

### Comparison and equivalence

Select `comparison_method` and `comparison_parameters` in `STUDY_CONFIG`.
`mgp/semantics.py` provides `select_comparison()` and a small `COMPARISON_METHODS`
registry. Defaults use cosine; RMSE distance is also registered. Additional
symmetric, deterministic functions taking two vectors can be registered there.
Their parameters must be JSON-serializable. Configure `histogram_edges` for the
selected method's score range; out-of-range finite scores raise an error rather
than being silently dropped. Fitness and IC always keep their RMSE behavior.

The registry is independent of statistics. Other consumers can select their own
method. For example, `Archive(comparison=select_comparison("rmse"),
comparison_is_distance=True)` uses a distance threshold; normal archives retain
their existing cosine threshold and filtering behavior. The study does not change
archive comparison when changing its statistical comparison method.

For uniqueness select `equivalence_method="rmse"` with
`equivalence_tolerance=1e-12`, or `equivalence_method="allclose"` with explicit
`equivalence_atol=1e-12` and `equivalence_rtol=1e-12`.
`select_equivalence()` and `EQUIVALENCE_METHODS` are the extension points.
Similarity and equivalence are distinct: proportional vectors can have cosine 1
while being different semantic objects.

Tolerance comparisons are not necessarily transitive; NumPy allclose can also
be asymmetric. Uniqueness uses the first matching representative in workspace
order, calling the method as `(candidate, representative)`. This deterministic
convention is stored with the experiment. Nonfinite vectors count as semantic
objects but have a separate nonfinite count and are excluded from equivalence
classes. A zero vector is finite and can be unique, but its cosine is undefined.

### Statistics and saved data

Step 0 is the initial workspace. After every successful MERGE, only top-level
objects contribute to AS, AO, CS, CO, I and ICO counts and proportions. Every
step records its contributing derivation count; ended runs do not contribute to
later steps. Recovery observes initial semantic objects and selected MERGE
results. Discarded IC candidates and semantic components hidden inside initial
incompletes do not count as discoveries.

Target and subtarget recovery use RMSE, independently of uniqueness/comparison.
`perfect_fit_tolerance` and `subtarget_tolerance` default to the static value `1e-8`. Subtargets
are all nonempty proper subsets of unsigned polynomial terms. For Nguyen-12,
`x1**3` and `x2` are positive subtarget components, while the full target keeps
its negative signs. Every found subtarget has an identifier and first-hit step;
initial hits are step 0.

Every unordered semantic pair contributes once to a step's histogram. Identical
vectors may be grouped computationally, retaining their exact pair multiplicity.
Only affected pair contributions are updated after a workspace change. Raw
pairwise scores are discarded immediately. One aggregate histogram is stored per
target/configuration/step. Its integer bin counts pool all valid pairs from all
contributing runs. CSV probabilities divide by the total valid pair count, so
they sum to 1 whenever that total is positive. Scores remain on their original
axis: cosine bins cover [-1, 1], not a rescaled [0, 1].

Undefined comparisons, including cosine involving zero/nonfinite vectors, do
not enter the mean or histogram; total, valid and undefined pair counts are
saved. The average comparison is the per-run mean over valid pairs. Across runs,
Welford updates preserve count, mean and M2; exports report population standard
deviation. Counts, proportions, unique count and workspace size use the same
stable aggregation. A pooled histogram weights runs by their valid pair counts;
it is not a mean of separately normalized per-run histograms.

| File or table | Content |
| --- | --- |
| `study.json` | Complete fixed configuration, grid, target definitions, seed procedure, methods, bins, schema and implementation identity |
| `datasets/TARGET.csv` | Actual fixed inputs and target values |
| `study.sqlite3` | Compact completed runs, seed/configuration tables, subtarget first hits, step aggregates and selected traces |
| `csv/runs.csv` | Full run key, both RNG sub-seeds, success, first hit, best/final error, MERGE counts, termination, archive size and duration |
| `csv/configurations.csv`, `csv/seeds.csv` | Resolved configurations and deterministic repetition seed plan |
| `csv/ranking.csv` | Results per target/configuration, using actual completed-run denominators |
| `csv/subtargets.csv`, `csv/subtarget_hits.csv` | Subtarget definitions and per-run first discoveries |
| `csv/subtarget_summary.csv` | Individual recovery frequencies, including subtargets never found |
| `csv/step_summary.csv` | Means, population deviations and contributor counts by step |
| `csv/histograms.csv` | Fixed score-bin boundaries, accumulated counts, valid-pair total and probabilities |
| `csv/run_steps.csv` | Optional selected traces, with raw/evaluated workspaces and per-step metrics |
| `csv/export.json` | Snapshot completion count and export filters |

CSV export is explicit, avoiding repeated rewriting of huge result files after
every run. Use `--target` and `--configuration` to export a manageable subset.
Best/final nonfinite fitness is `inf`; absent measurements are blank/null.
Means of best fitness use finite values and report their contributing count.
Durations are observational and naturally differ between reproducible runs.

`trace_run_ids=[]` disables full traces by default. Put a few stable run IDs in
that list before starting the study to save their raw/evaluated workspace history
and per-step measurements. Other runs retain compact scalar results and
incremental aggregates, not their full semantic vectors or temporary objects.
Use `main_replay.py` to reconstruct any completed run from its saved inputs and
seeds, even without a trace. Its visual mode uses the saved workspace size and
configuration, not the small defaults in `main_visual.py`.

## Automatic study reports

After the final planned run commits, `save_report=True` generates the report
in `report/`. `main_study_example.py` uses the same reporting path. A partial study
can be reported manually without repeating derivations:

```text
python main_study_report.py "hyperparameter_output/STUDY_DIRECTORY"
```

See [PLOT_GUIDE.md](PLOT_GUIDE.md) for the complete current output list and
aggregation definitions. The HTML gallery links all retained figures and the
report's data and documentation folders. English labels are concise; additional
interpretation and diagnostic details are kept in `report/docs/report.md`.


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
vector graphics without opening a window; the interactive viewer and visual replay call `plt.show()`.

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
to child trees. It keeps the syntactic history for display. Child-object references additionally retain the existing numeric
vectors needed for recursive IM; these vectors are shared without recalculation. Workspace
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

## Tests

Tests use `unittest`, with small methods grouped by topic. Each method checks one
behavior or a closely related result. The MERGE table uses independent subtests
for its 36 binary-only cells and all 64 ordered cells of the updated arity table.
Unary tests cover head selection, nesting, duplicate evaluation and operator
flags, including logarithm and square-root domains. Nguyen tests check the
published formulas, input variables, default intervals and explicit overrides.
Separate tests cover caching, numeric limits, seeds,
archives, stopping conditions, output files and plot coordinates.
Tree tests cover operand history, layout and incremental derivation. Keyboard
tests dispatch Matplotlib events to check one-step advancement and stopping.

## New modules in v12

| Module | Responsibility |
| --- | --- |
| `mgp/merge.py`, `mgp/objects.py` | Occurrence accessibility, shared IM/EM construction and cached tree counts |
| `mgp/derivation.py` | Operation selection, fallback, survival, termination and shared best/archive updates |
| `mgp/semantics.py` | Reusable comparison and equivalence registries |
| `mgp/diversity.py` | Top-level six-class composition, unique counts and incremental pair statistics |
| `mgp/recovery.py` | Unsigned polynomial subtargets and materialized first-hit tracking |
| `mgp/study.py` | Fixed datasets, paired seed plan and one shared study loop |
| `mgp/study_storage.py` | Atomic run persistence, explicit completion cursor and online aggregates |
| `mgp/study_export.py` | Streaming CSV exports and summaries |

The original main, repetition, statistics, examples and viewer remain available.
Their behavior with IM disabled is checked against v11. Focused tests cover IM
flags and recursive occurrence selection, grid datasets, unsigned subtargets,
materialized recovery, comparison methods, histogram counts and resume. The
full 123-million-run experiment is not a validation test.
