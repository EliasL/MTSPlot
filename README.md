# MTSPlot / MTS_Plotting

Publication figures for current MTS2D output. From the shared workspace, install
with `python -m pip install --config-settings editable_mode=compat -e MTSManager -e 'MTSPlot[math]'`.
Python imports use `MTS_Plotting`; the command
line entry point remains `mtsplot`. The manager is imported as `MTS_Management`.
The same repository also contains the independent model-math modules in
[MTM_Math](MTM_Math/README.md). Install `MTSPlot[math]` for their SymPy dependency.
MTSPlot is an independent Git repository; data and generated figures are ignored.

## Four visual slots

`plot_benchmark` accepts Y, X, color, gradient, and explicit fixed selections:

```python
from MTS_Plotting.benchmark import summarize, plot_benchmark

rows, audit = summarize("data/magi_benchmark_oct26")
plot_benchmark(rows, "elastic-size.pdf",
    y="elastic_minimization_s", x="size", color="preconditioner",
    gradient="threads", fixed={"reconnection": "none"})
```

Y can be `elastic_minimization_s`, `plastic_minimization_s`,
`first_minimization_s`, `tangent_preparation_s`, or `total_runtime_s`.
The interchangeable X/color/gradient properties are `size`, `threads`,
`preconditioner`, and `reconnection`. Categorical X is supported. Unassigned
properties must be fixed or constant in the selected sample; varying ones raise
an error instead of silently pooling data. Slot assignments must be distinct.
`size` requires square systems. Size on X uses a logarithmic X axis. Any view comparing sizes (on X, color,
or gradient) uses a logarithmic timing axis so smaller systems remain visible;
these views require positive mean timings. Other views use a linear timing axis.

Each point first uses a run's mean step timing and then averages across seeds,
with equal weight per seed. Error bars show sample SD across seeds, not a
confidence interval. Single-seed points have no uncertainty bar. Duplicate seeds
within a plotted point raise an error. The returned Python point table contains mean, SD, sample count and seed IDs;
no derived CSV is written. Sparse curves contain only available points; missing runs
are never imputed. Color families are blue and orange for the two comparison
categories; gradients run from light (small values) to dark (large values).
Markers distinguish the primary comparison categories.

## The 32-figure benchmark

The master call was written first in
[examples/benchmark_figures.py](examples/benchmark_figures.py). It requests:

- Two metrics: mean elastic and mean plastic minimization time.
- Preconditioner colors, separately for each reconnection setting; then
  reconnection colors, separately for each preconditioner.
- Four views per comparison: size at 16 threads; size with thread gradients;
  threads at L400; threads with size gradients.

This is `2 metrics × 2 comparisons × 2 fixed settings × 4 views = 32 PDFs`.
The master reads one snapshot, summarizes it once, and expands its small product
of slot specifications through the same renderer. Regenerate from any working
directory using the script's absolute path. From the workspace root:

```sh
MPLBACKEND=Agg .venv/bin/python MTSPlot/examples/benchmark_figures.py
```

The script overwrites the same figure paths as more data arrive. It writes only
32 PDFs and a figure index in `figures/magi_benchmark_oct26`. Supporting tables
are regenerated from the raw snapshot instead of being saved. `plot_suite` returns
its report in memory; an explicit `report_path` can save it under `plottingData/`
if needed. The CLI similarly has an optional `--save-summary` argument.
Use the explicit fetch example separately to update the snapshot; plotting never
submits, downloads or modifies simulations.

For a single figure from the command line:

```sh
mtsplot data/magi_benchmark_oct26 figures/single --metric elastic_minimization_s \
  --x size --color preconditioner --gradient threads \
  --fixed '{"reconnection":"none"}'
```

## Eligibility and timing interpretation

Only runs whose configured sample targets were met enter timing figures.
Diagnostic logging runs, partial runs and strain-limit stops remain in the audit
but are excluded. Missing means absent from the local snapshot, not absent from
the remote cluster. Malformed/truncated CSVs, inconsistent counters and identities,
and mixed executables or unintended solver settings raise errors.

Initialization is the first cumulative minimization time. Elastic/plastic step
times are differences of that cumulative counter, grouped by the recorded step
class, excluding initialization. Tangent preparation is already included in
minimization time and must not be added again. Total runtime includes initial
relaxation and the loading history needed to reach the sample target.

Incomplete seed coverage can favor faster runs. Inspect point counts and IDs;
these figures do not imply matched-seed speedups or fitted scaling exponents.
The L400 views use whatever target-complete data are present; rerun after fetching
later results.

## Other plotting

`MTS_Plotting.macros.plot_macros` plots macro responses and
`MTS_Plotting.style.figure` provides the shared compact article style. No global
style changes or external LaTeX installation are required.

```sh
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

Plot appearance is reviewed by the user. Agents must not render PDFs, inspect
plot images or run automated visual checks; they run the plotting code and fix
runtime errors only.

Figure names use `metric_X_color_[gradient]_fixed-values.pdf`. Fixed values follow
preconditioner, reconnection, size, threads order, omitting properties assigned to
slots. Sizes use L and thread selections use t. For example:
`elastic_threads_reconnection_exactTangent_L400.pdf` and
`elastic_size_preconditioner_threads_none.pdf`.
