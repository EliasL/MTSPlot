"""Per-seed timings from cumulative counters, separated by step class."""

from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics
from datetime import datetime, timezone

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl

from .runs import discover
from .style import STYLE, figure

METRICS = {
    "first_minimization_s": "Initial minimization (s)",
    "elastic_minimization_s": "Mean elastic minimization (s)",
    "plastic_minimization_s": "Mean plastic minimization (s)",
    "tangent_preparation_s": "Tangent preparation (s)",
    "total_runtime_s": "Total runtime (s)",
}


def seconds(value):
    """Parse the simulator's d/h/m/s duration strings, without guessing formats."""
    tokens = value.split()
    if not tokens or any(not re.fullmatch(r"\d+(?:\.\d+)?[dhms]", t) for t in tokens):
        raise ValueError(f"Invalid simulator duration: {value!r}")
    units = [t[-1] for t in tokens]
    if len(set(units)) != len(units) or units != sorted(units, key="dhms".index):
        raise ValueError(f"Invalid duration unit order: {value!r}")
    return sum(float(t[:-1]) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[t[-1]]
               for t in tokens)


def timing_summary(run):
    c = run.config
    diagnostic = c["logDuringMinimization"] or c["fullMinimizationLogging"]
    term = run.termination
    status = "partial" if term is None else term["stopReason"]
    result = dict(folder=str(run.folder), config_hash=c.config_hash,
                  executable_sha256=run.manifest["executable_sha256"],
                  rows=c["rows"], cols=c["cols"], threads=c["nrThreads"],
                  preconditioner=c["preconditioner"], reconnection=c["reconnectionMethod"],
                  seed=c["seed"], diagnostic=diagnostic, status=status,
                  eligible=False, plastic_steps=None, completed_steps=None,
                  **{metric: None for metric in METRICS})
    path = run.folder / "macroData.csv"
    if not path.exists():
        if term is not None:
            raise ValueError(f"Terminated run has no macro table: {run.folder}")
        return result
    rows = run.macros()
    if not rows:
        if term is not None:
            raise ValueError(f"Terminated run has an empty macro table: {run.folder}")
        return result
    steps = [int(row["completed_steps"]) for row in rows]
    if steps != list(range(len(rows))):
        raise ValueError(f"Expected initialization followed by contiguous steps: {run.folder}")
    times = [seconds(row["minimization_time"]) for row in rows]
    tangent = [float(row["tangent_preparation_time_s"]) for row in rows]
    for values in (times, tangent):
        if any(not math.isfinite(x) or x < 0 for x in values) or any(
                b < a for a, b in zip(values, values[1:])):
            raise ValueError(f"Invalid cumulative timing: {run.folder}")
    by_class = defaultdict(list)
    for previous, current, row in zip(times, times[1:], rows[1:]):
        kind = row["step_class"]
        if kind not in {"elastic", "plastic"}:
            raise ValueError(f"Unknown loading step class: {kind}")
        by_class[kind].append(current - previous)
    last = rows[-1]
    plastic = int(last["plastic_steps"])
    if plastic != len(by_class["plastic"]) or int(last["elastic_steps"]) != len(by_class["elastic"]):
        raise ValueError(f"Step counters disagree with classifications: {run.folder}")
    if term is not None and (term["steps"] != steps[-1] or term["plasticSteps"] != plastic):
        raise ValueError(f"Termination and macro table disagree: {run.folder}")
    eligible = bool(term and term["targetsMet"] and not diagnostic)
    result.update(eligible=eligible, plastic_steps=plastic, completed_steps=steps[-1],
                  first_minimization_s=times[0], tangent_preparation_s=tangent[-1],
                  total_runtime_s=seconds(last["run_time"]))
    for kind in ("elastic", "plastic"):
        if by_class[kind]:
            result[f"{kind}_minimization_s"] = statistics.mean(by_class[kind])
    return result


def summarize(root, *, campaign=None):
    """Return run summaries and an audit of expected versus downloaded runs."""
    runs = discover(root)
    expected = None
    if campaign:
        # Share campaign parsing too; it validates complete settings and duplicates.
        from MTS_Management.campaign import campaign_entries
        name, entries = campaign_entries(campaign)
        expected = {c.config_hash for c, _ in entries}
        unexpected = {r.config.config_hash for r in runs} - expected
        if unexpected or any(r.manifest["campaign"] != name for r in runs):
            raise ValueError("Data root contains runs outside the requested campaign")
    summaries = [timing_summary(run) for run in runs]
    eligible = [run for run, row in zip(runs, summaries) if row["eligible"]]
    if eligible:
        # A thread comparison must not silently vary other physical/solver settings.
        varied = {"rows", "cols", "nrThreads", "seed", "preconditioner", "reconnectionMethod"}
        signatures = {json.dumps({k: v for k, v in r.config.as_dict().items() if k not in varied},
                                 sort_keys=True) for r in eligible}
        if len(signatures) > 1 or len({r.manifest["executable_sha256"] for r in eligible}) > 1:
            raise ValueError("Benchmark comparisons require identical executable and non-sweep settings")
    audit = dict(expected=len(expected) if expected is not None else None,
                 available=len(runs), missing=len(expected - {r.config.config_hash for r in runs})
                 if expected is not None else None,
                 eligible=sum(row["eligible"] for row in summaries),
                 partial=sum(row["status"] == "partial" for row in summaries),
                 diagnostic=sum(row["diagnostic"] for row in summaries))
    return summaries, audit


DIMENSIONS = {"size": "System size L", "threads": "Threads",
              "preconditioner": "Preconditioner", "reconnection": "Reconnection"}


def dimension(row, key):
    if key == "size":
        if row["rows"] != row["cols"]:
            raise ValueError("Size comparisons require square systems")
        return row["rows"]
    return row[key]


def figure_name(y, *, x, color, gradient=None, fixed=None):
    """Y_X_color_[gradient]_fixed-values; size/thread selections use L/t prefixes."""
    parts = [y.removesuffix("_minimization_s").removesuffix("_s"), x, color]
    if gradient:
        parts.append(gradient)
    for key in ("preconditioner", "reconnection", "size", "threads"):
        if key in (fixed or {}):
            value = fixed[key]
            parts.append(f"L{value}" if key == "size" else
                         f"t{value}" if key == "threads" else str(value))
    return "_".join(parts) + ".pdf"


def plot_benchmark(rows, path, *, y, x, color, gradient=None, fixed=None):
    """Assign benchmark properties to visual slots; never pool an unassigned sweep.

    Each point is the mean of per-seed timings, with sample SD across seeds.
    A single-seed point has no uncertainty bar. Return the plotted point table.
    """
    METRICS[y]  # Fail before writing anything for an unknown metric.
    fixed = dict(fixed or {})
    slots = [x, color] + ([gradient] if gradient else [])
    if (len(set(slots)) != len(slots) or set(slots) & fixed.keys() or
            (set(slots) | fixed.keys()) - DIMENSIONS.keys()):
        raise ValueError("Visual slots and fixed properties must be distinct benchmark dimensions")
    selected = [r for r in rows if r["eligible"] and r[y] is not None and
                all(dimension(r, k) == v for k, v in fixed.items())]
    if not selected:
        raise ValueError(f"No eligible {y} data for {fixed}")
    for key in DIMENSIONS.keys() - set(slots) - fixed.keys():
        if len({dimension(r, key) for r in selected}) > 1:
            raise ValueError(f"Assign {key} to a visual slot or fix it explicitly")
    groups = defaultdict(list)
    for row in selected:
        value = row[y]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {y}: {value}")
        groups[tuple(dimension(row, k) if k else None for k in (x, color, gradient))].append(row)
    points = []
    for (position, category, shade), samples in sorted(groups.items()):
        seeds = [r["seed"] for r in samples]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seeds in a plotted point; an unassigned property may vary")
        values = [r[y] for r in samples]
        points.append(dict(x=position, color=category, gradient=shade,
                           mean=statistics.mean(values),
                           sd=statistics.stdev(values) if len(values) > 1 else None,
                           seeds=len(seeds), seed_ids=sorted(seeds)))
    categories = sorted({p["color"] for p in points})
    shades = sorted({p["gradient"] for p in points})
    positions = sorted({p["x"] for p in points})
    numeric_x = all(type(v) in (int, float) for v in positions)
    coordinates = {v: v if numeric_x else i for i, v in enumerate(positions)}
    palettes = ("Blues", "Oranges", "Greens", "Purples", "Reds", "Greys")
    color_levels = sorted({dimension(r, color) for r in rows if r["eligible"]})
    if len(color_levels) > len(palettes):
        raise ValueError("Too many color categories; use gradients or a fixed selection")
    inks = dict(zip(color_levels, palettes))
    log_y = "size" in slots
    if log_y and any(p["mean"] <= 0 for p in points):
        raise ValueError("Logarithmic size comparisons require positive mean timings")
    path = Path(path)
    if path.suffix != ".pdf":
        raise ValueError("Benchmark figures must use a .pdf path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(STYLE):
        fig, ax = figure(width=4.8 if gradient else 3.4, height=3.1 if gradient else 2.5)
        try:
            for i, category in enumerate(categories):
                cmap = mpl.colormaps[inks[category]]
                for j, shade in enumerate(shades):
                    curve = [p for p in points if p["color"] == category and p["gradient"] == shade]
                    if not curve:
                        continue
                    tint = .8 if not gradient else .45 + .5 * j / max(1, len(shades)-1)
                    ink = cmap(tint)
                    ax.plot([coordinates[p["x"]] for p in curve], [p["mean"] for p in curve],
                            color=ink, marker=("o", "s", "^", "D", "v", "P")[i],
                            markerfacecolor="none")
                    replicated = [p for p in curve if p["sd"] is not None]
                    if replicated:
                        ax.errorbar([coordinates[p["x"]] for p in replicated],
                                    [p["mean"] for p in replicated],
                                    yerr=[p["sd"] for p in replicated], fmt="none",
                                    capsize=2, color=ink, linewidth=.7)
            ax.set(xlabel=DIMENSIONS[x], ylabel=METRICS[y])
            if x == "size":
                ax.set_xscale("log", base=2)
            if log_y:
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
            ax.set_xticks([coordinates[v] for v in positions], labels=[str(v) for v in positions])
            handles = [Line2D([], [], color=mpl.colormaps[inks[value]](.8),
                              marker=("o", "s", "^", "D", "v", "P")[i],
                              markerfacecolor="none", label=str(value))
                       for i, value in enumerate(categories)]
            if gradient:
                handles += [Line2D([], [], linestyle="none", label="\n" + DIMENSIONS[gradient])]
                handles += [Line2D([], [], color=mpl.colormaps["Greys"](
                    .45 + .5*j/max(1, len(shades)-1)), label=str(value))
                           for j, value in enumerate(shades)]
                ax.legend(handles=handles, title=DIMENSIONS[color], title_fontsize=8,
                          frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
            else:
                ax.legend(handles=handles, frameon=False)
            fig.savefig(path, bbox_inches="tight")
        finally:
            plt.close(fig)
    return points


def plot_suite(rows, destination, *, metrics, comparisons, views, audit=None, report_path=None):
    """A small product of explicit slot assignments, rendered by one plot function."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    figures = []
    for metric in metrics:
        for comparison in comparisons:
            color, split = comparison["color"], comparison["split"]
            if split == color or any(split in v.get("fixed", {}) for v in views):
                raise ValueError("A comparison's split must remain separate from its visual slots and filters")
            levels = sorted({dimension(r, split) for r in rows if r["eligible"]})
            for level in levels:
                for view in views:
                    fixed = {split: level} | view.get("fixed", {})
                    slots = dict(y=metric, x=view["x"], color=color,
                                 gradient=view.get("gradient"), fixed=fixed)
                    filename = figure_name(**slots)
                    if any(f["file"] == filename for f in figures):
                        raise ValueError(f"Duplicate figure specification: {filename}")
                    points = plot_benchmark(rows, destination / filename, **slots)
                    figures.append(dict(file=filename, points=len(points), **slots))
    report = dict(audit=audit, created_at=datetime.now(timezone.utc).isoformat(), figures=figures)
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    (destination / "README.md").write_text(
        "# Benchmark figures\n\nMean per-seed timings; error bars show sample SD. "
        "Missing runs are not imputed. Supporting data can be regenerated from the snapshot.\n\n" +
        "\n".join(f"- [{f['file']}]({f['file']})" for f in figures) + "\n")
    return report


