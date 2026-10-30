# MTSPlot

MTSPlot creates scientific figures from MTS2D simulation output. It provides
response curves and performance comparisons across system sizes, thread counts,
preconditioners and mesh-reconnection settings.

The plotting package is named `MTS_Plotting`. This repository also provides
[MTM_Math](MTM_Math/README.md), which contains energy, stress, tangent and reduction
functions for the mesoscopic tensorial model.

## Installation

Use Python 3.11 or newer and place MTSManager beside this repository:

```text
MTS/
  MTSManager/
  MTSPlot/
```

From `MTSPlot/`, create or activate a Python environment and install both packages:

```sh
python3.11 -m venv ../.venv
source ../.venv/bin/activate
python -m pip install --config-settings editable_mode=compat -e ../MTSManager -e '.[math]'
```

The `math` extra installs SymPy for model calculations. Plotting uses NumPy and
Matplotlib. Editable installation lets Python and editor navigation resolve the
local source files. Unless noted otherwise, the examples below run from MTSPlot/.

## Plot a simulation response

Point `discover` at a managed run folder or a directory containing managed runs.
Each run needs its saved `config.conf`, `manifest.json` and macro CSV.

```python
from MTS_Plotting.runs import discover
from MTS_Plotting.macros import plot_macros

runs = discover("data/example")
fig, ax = plot_macros(runs)
ax.legend(loc="best", frameon=True)
fig.savefig("stress-response.pdf", bbox_inches="tight")
```

The default response plot shows mean first-Piola shear stress against load.
`plot_macros` accepts `x`, `y`, `xlabel` and `ylabel` to select other macro columns.
It returns the Matplotlib figure and axes so you can adjust the plot before saving.

## Compare simulation timings

The desktop Plot panel selects campaigns and discovers their varied parameters.
It supports response curves, scalar reductions of macro columns and the timing
summaries below. Unused varied parameters require fixed selections. Plotting
reads a selected local collection and reports missing or partial runs.

`MTS_Plotting.campaign_plot.prepare(campaign, data, selections)` returns plotted
values and warnings; `plot(campaign, data, pdf, selections)` writes a vector PDF.
Selections specify `x`, `y`, optional `color` and `gradient`, `fixed`, `seed`
(`"all"` or one integer), and `normalization` (`"none"`, `"L"` or `"N"`, where N
is rows × cols). Macro-column scalar reductions use `reduction` (`"last"`,
`"first"`, `"mean"`, `"min"` or `"max"`). Normalization precedes seed averaging.
Response means align by load, using linear interpolation only over the common
range when grids differ, with a warning. Uncertainty is sample standard
deviation across seeds. No extrapolation or missing-seed imputation is applied.
For parametric response plots, X is the seed mean and uncertainty is shown on Y.
Independent `xscale` and `yscale` selections accept `"linear"` or `"log"`;
logarithmic axes reject nonpositive data. `legend_style` accepts `"compact"`,
`"full"` or `"custom"`, with custom parameter labels in `legend_aliases`.
Compact defaults include L for size, t for threads, R for reconnection and C for
preconditioner. Direct PNG previews use the same drawing function as PDF export.
Reduced numerical data is cached separately from appearance settings and
invalidated when configuration, manifest, campaign or macro data files change.

Timing plots assign a measured quantity to Y and simulation properties to X,
color and an optional color gradient. For example, compare elastic minimization
time across square system sizes, using colors for preconditioners and shades for
thread counts:

```python
from MTS_Plotting.benchmark import summarize, plot_benchmark

rows, audit = summarize("data/example")
points = plot_benchmark(
    rows, "elastic-size.pdf",
    y="elastic_minimization_s",
    x="size", color="preconditioner", gradient="threads",
    fixed={"reconnection": "none"},
    legend={"loc": "best", "ncols": 2},
)
```

Available Y metrics are `elastic_minimization_s`, `plastic_minimization_s`,
`first_minimization_s`, `tangent_preparation_s` and `total_runtime_s`.
The X, color and gradient slots accept `size`, `threads`, `preconditioner` and
`reconnection`. Other varying properties must be fixed explicitly, preventing
accidental averaging across different simulation settings. `size` requires square
systems. Comparisons across sizes use logarithmic timing axes; size on X also
uses a logarithmic X axis.

Each point averages per-seed timings with equal weight per seed. Error bars show
sample standard deviation across seeds, not a confidence interval; a single-seed
point has no uncertainty bar. The returned `points` list includes means, standard
deviations, seed counts and seed IDs.

Only runs that reached their configured sample targets contribute to timing
figures. Partial runs, strain-limit stops and diagnostic-logging runs are excluded
and counted in `audit`. Comparisons reject inconsistent identities, malformed
tables, mixed executables and unintended changes to solver settings. Missing
means absent from the supplied data directory, not necessarily absent remotely.
Incomplete seed coverage can favor faster runs, so inspect counts before drawing
speedup or scaling conclusions.

Elastic and plastic step timings come from differences in the cumulative
minimization counter, excluding initialization. Tangent preparation is already
included in minimization time. Total runtime also includes initialization and the
loading history needed to reach the sample target.

## Example: an overview of a performance campaign

[examples/benchmark_figures.py](examples/benchmark_figures.py) demonstrates how to
combine several timing comparisons in a single `plot_suite` call. It uses a saved
campaign from a performance study on the MAGI computing cluster, described in
[MTSManager's campaign notes](../MTSManager/campaigns/magi_benchmark_oct26/README.md).
The example compares elastic and plastic timings across sizes, thread counts,
preconditioners and reconnection settings.

To use it, configure SSH access and the `magi` site in MTSManager's `sites.toml`,
then download completed results:

```sh
python examples/fetch_benchmark.py ../MTSManager/sites.toml \
  ../MTSManager/campaigns/magi_benchmark_oct26/simulations.json data/magi_benchmark_oct26
MPLBACKEND=Agg python examples/benchmark_figures.py
```

The plotting script reads the local snapshot and writes PDFs and an index under
`figures/magi_benchmark_oct26/`. Adapt its paths, metrics and view specifications
for your own campaign. Rerun after fetching more completed runs to update figures.
Plotting functions themselves do not download data or change simulations.

`plot_suite` returns a report in memory. If you need a saved report, supply an
explicit `report_path`, for example under `plottingData/`. Derived CSV files are
not saved automatically.

## Adjust legends and axes

Legends use Matplotlib's standard framed box with location `best` by default.
Pass `legend={"loc": "upper right", "ncols": 2}`, `xlim=(...)` or `ylim=(...)` to
`plot_benchmark`. For a suite, set a common `legend` argument or include these
options in individual view specifications. Adjust placement and limits after
reviewing the resulting PDFs.

The `mtsplot` command exposes the same selections; use `mtsplot --help` for options.
For example:

```sh
mtsplot data/example figures/example --metric elastic_minimization_s \
  --x size --color preconditioner --gradient threads \
  --fixed '{"reconnection":"none"}' --legend-ncols 2
```

## Development checks

```sh
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

Tests focus on timing calculations, data selection and model mathematics. The
simulator-comparison checks also require an MTSManager build of MTS2D. Plot
appearance is reviewed manually.
