"""Campaign-selected response curves and scalar comparisons from local runs."""

import csv
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from MTS_Management.campaign import campaign_entries
from MTS_Management.config import DEFAULTS
from .runs import Run

LABELS = {
    "size": "System size L", "nrThreads": "Threads", "preconditioner": "Preconditioner",
    "reconnectionMethod": "Reconnection", "seed": "Seed", "load": "Load",
    "avg_energy": "Mean energy density", "total_energy": "Total energy",
    "avg_sigma12": "Mean shear stress σ12", "avg_sigma11": "Mean normal stress σ11",
    "avg_sigma22": "Mean normal stress σ22", "avg_P12": "Mean first-Piola stress P12",
    "run_time": "Cumulative runtime (s)", "minimization_time": "Cumulative minimization time (s)",
}
REDUCTIONS = {"last": "Final value", "first": "Initial value", "mean": "Mean over saved rows",
              "min": "Minimum", "max": "Maximum"}
ABBREVIATIONS = {"size": "L", "nrThreads": "t", "reconnectionMethod": "R", "preconditioner": "C",
                 "seed": "s", "tangentBackend": "backend", "logDuringMinimization": "diag"}
COMMON_Y = ["avg_sigma12", "avg_energy", "total_energy", "avg_sigma11", "avg_sigma22",
            "elastic_minimization_s", "plastic_minimization_s", "first_minimization_s",
            "tangent_preparation_s", "total_runtime_s"]
APPEARANCE = {"ncols", "legend_loc", "legend_style", "legend_aliases", "xlim", "ylim", "xscale", "yscale", "show_all", "xlabel", "ylabel"}


def definition(path):
    name, entries = campaign_entries(path)
    square = all(c["rows"] == c["cols"] for c, _ in entries)
    settings = {c.config_hash: parameters(c, square) for c, _ in entries}
    values = {k: sorted({s[k] for s in settings.values()}) for k in next(iter(settings.values()))}
    varied = {k: v for k, v in values.items() if len(v) > 1}
    return name, settings, varied


def parameters(config, square):
    values = {k: config[k] for k in DEFAULTS}
    if square:
        values["size"] = values.pop("rows")
        values.pop("cols")
    return values


def matching_folders(root, hashes):
    root = Path(root)
    manifests = [root / "manifest.json"] if (root / "manifest.json").exists() else sorted(root.glob("*/manifest.json"))
    return [p.parent for p in manifests if json.loads(p.read_text())["config_hash"] in hashes]


def number(value, key):
    if key in {"run_time", "minimization_time"}:
        from .benchmark import seconds
        return seconds(value)
    return float(value)


def describe(campaign, sources, root=None):
    """Inspect saved definitions and CSV headers/first rows, without remote reads."""
    from .benchmark import METRICS
    name, expected, varied = definition(campaign)
    collections = [{"path": str(Path(p).resolve()), "runs": len(matching_folders(p, expected))}
                   for p in dict.fromkeys(map(str, sources)) if Path(p).is_dir()]
    collections = sorted((c for c in collections if c["runs"]), key=lambda c: (-c["runs"], c["path"]))
    root = str(Path(root).expanduser().resolve()) if root else (collections[0]["path"] if collections else "")
    folders = matching_folders(root, expected) if root else []
    available, headers = set(), set()
    for folder in folders:
        path = folder / "macroData.csv"
        if not path.exists():
            continue
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "load" not in reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f"Invalid macro CSV header: {path}")
            headers.update(reader.fieldnames or [])
            first = next(reader, {})
            if None in first or any(v is None for v in first.values()):
                raise ValueError(f"Malformed macro CSV: {path}")
            for key, value in first.items():
                try:
                    if np.isfinite(number(value, key)):
                        available.add(key)
                except ValueError:
                    pass  # Text columns are not quantitative axes.
    quantities = {k: {"label": LABELS.get(k, k), "kind": "macro", "available": k in available}
                  for k in available | {"load", "total_energy", "avg_energy", "avg_sigma12", "avg_sigma11", "avg_sigma22"}}
    for key, label in METRICS.items():
        quantities[key] = {"label": label, "kind": "metric", "available":
                           {"minimization_time", "tangent_preparation_time_s", "completed_steps", "step_class"} <= headers}
    parameter_order = [k for k in ("size", "rows", "cols", "nrThreads", "seed", "preconditioner", "reconnectionMethod") if k in varied]
    parameter_order += sorted(varied.keys() - set(parameter_order))
    return {"campaign": name, "parameters": varied, "seeds": sorted({s["seed"] for s in expected.values()}),
            "labels": {k: LABELS.get(k, k) for k in varied}, "quantities": quantities,
            "sources": collections, "data": root, "expected": len(expected), "available": len(folders),
            "square": all("size" in s for s in expected.values()), "reductions": REDUCTIONS,
            "common_axes": {"x": ["load", "accumulated_load", "completed_steps", *parameter_order], "y": COMMON_Y},
            "abbreviations": {k: ABBREVIATIONS.get(k, k) for k in varied}}


def compatibility(schema, spec):
    """Return selection errors shared by export and data preparation."""
    x, y = spec["x"], spec["y"]
    params, quantities = schema["parameters"], schema["quantities"]
    errors = []
    if x not in params and (x not in quantities or quantities[x]["kind"] != "macro"):
        errors.append("X must be a varied parameter or a saved response column.")
    if y not in quantities:
        errors.append("Y must be a measured quantity.")
    if x == y:
        errors.append("X and Y must be different quantities.")
    if x not in params and y in quantities and quantities[y]["kind"] != "macro":
        errors.append("A timing summary requires a campaign parameter on X.")
    for key in (x, y):
        if key in quantities and not quantities[key]["available"]:
            errors.append(f"{quantities[key]['label']} is unavailable in this data collection.")
    slots = [key for key in (x if x in params else None, spec.get("color"), spec.get("gradient")) if key]
    if len(set(slots)) != len(slots) or any(key not in params for key in slots):
        errors.append("X, color and gradient must use distinct campaign parameters.")
    if spec.get("normalization", "none") not in {"none", "L", "N"}:
        errors.append("Unknown system-size normalization.")
    if spec.get("normalization") == "L" and not schema["square"]:
        errors.append("Normalization by L requires square systems.")
    return errors


def align_seeds(samples):
    """Align each seed on load; interpolate only inside the shared load range."""
    loads = [s[0] for s in samples]
    matched = all(np.array_equal(loads[0], v) for v in loads[1:])
    if matched:
        xs, ys = np.array([s[1] for s in samples]), np.array([s[2] for s in samples])
    else:
        ordered = []
        for load, x, y in samples:
            if np.all(np.diff(load) < 0):
                load, x, y = load[::-1], x[::-1], y[::-1]
            if len(load) < 2 or not np.all(np.diff(load) > 0):
                raise ValueError("Mismatched load points must be strictly monotonic for interpolation. Select an individual seed.")
            ordered.append((load, x, y))
        lo, hi = max(s[0][0] for s in ordered), min(s[0][-1] for s in ordered)
        if lo > hi:
            raise ValueError("Seeds have no common load range.")
        grid = np.unique(np.concatenate([s[0][(s[0] >= lo) & (s[0] <= hi)] for s in ordered]))
        xs = np.array([np.interp(grid, load, x) for load, x, _ in ordered])
        ys = np.array([np.interp(grid, load, y) for load, _, y in ordered])
    return {"x": xs.mean(axis=0).tolist(), "mean": ys.mean(axis=0).tolist(),
            "sd": ys.std(axis=0, ddof=1).tolist() if len(samples) > 1 else None,
            "interpolated": not matched}


def prepare(campaign, root, spec, schema=None):
    from .benchmark import METRICS, timing_summary
    schema = schema or describe(campaign, [root], root)
    errors = compatibility(schema, spec)
    if errors:
        raise ValueError(" ".join(errors))
    _, expected, varied = definition(campaign)
    fixed = spec.get("fixed", {})
    slots = {spec["x"], spec.get("color"), spec.get("gradient")}
    if set(fixed) - varied.keys() or set(fixed) & slots or "seed" in fixed:
        raise ValueError("Fixed selections must be unused varied campaign parameters; select seeds separately.")
    if any(v not in varied[k] for k, v in fixed.items()):
        raise ValueError("A fixed value is absent from the campaign.")
    seed = spec.get("seed", "all")
    if seed != "all" and (type(seed) is not int or seed not in schema["seeds"]):
        raise ValueError("Select an existing seed or all seeds.")
    selected = {h: s for h, s in expected.items() if all(s[k] == v for k, v in fixed.items())
                and (seed == "all" or s["seed"] == seed)}
    for key in varied.keys() - slots - fixed.keys() - {"seed"}:
        if len({s[key] for s in selected.values()}) > 1:
            raise ValueError(f"Assign {key} to a plotting role or hold it fixed.")
    runs = [Run.read(p) for p in matching_folders(root, selected)]
    if len({r.config.config_hash for r in runs}) != len(runs):
        raise ValueError("Duplicate configurations; choose one local copy per run.")
    if len({r.manifest["executable_sha256"] for r in runs}) > 1:
        raise ValueError("Selected runs use different executables; choose a consistent data collection.")
    curve = spec["x"] not in varied
    groups, warnings, excluded, partial = defaultdict(list), [], 0, 0
    for run in runs:
        values = selected[run.config.config_hash]
        divisor = {"none": 1, "L": run.config["rows"], "N": run.config["rows"] * run.config["cols"]}[spec.get("normalization", "none")]
        key = tuple(values.get(spec.get(slot)) for slot in ("color", "gradient"))
        if not (run.folder / "macroData.csv").exists():
            excluded += 1
            continue
        if spec["y"] in METRICS:
            summary = timing_summary(run)
            value = summary[spec["y"]]
            if not summary["eligible"] or value is None:
                excluded += 1
                continue
            y = np.array([value / divisor])
        else:
            rows = run.macros()
            if not rows:
                excluded += 1
                continue
            def column(key):
                if key not in rows[0]:
                    raise ValueError(f"Column {key} is unavailable: {run.folder}")
                result = np.array([number(row[key], key) for row in rows])
                if not np.all(np.isfinite(result)):
                    raise ValueError(f"Non-finite {key}: {run.folder}")
                return result
            y = column(spec["y"]) / divisor
        partial += run.termination is None
        if curve:
            sample = (column("load"), column(spec["x"]), y)
        else:
            reduction = spec.get("reduction", "last")
            if reduction not in REDUCTIONS:
                raise ValueError("Unknown scalar reduction.")
            value = {"last": lambda: y[-1], "first": lambda: y[0], "mean": y.mean,
                     "min": y.min, "max": y.max}[reduction]()
            key += (values[spec["x"]],)
            sample = float(value)
        groups[key].append((values["seed"], sample))
    series = []
    for key, samples in sorted(groups.items(), key=lambda item: json.dumps(item[0])):
        seeds = [s[0] for s in samples]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seeds in a plot group; an unassigned configuration property varies.")
        item = dict(color=key[0], gradient=key[1], seed_ids=sorted(seeds), seeds=len(seeds))
        if curve:
            item.update(align_seeds([s[1] for s in samples]))
        else:
            values = [s[1] for s in samples]
            item.update(x=key[2], mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if len(values) > 1 else None)
        series.append(item)
    if not series:
        raise ValueError("No usable data matches this selection. Timing summaries require completed sample targets without diagnostic logging.")
    if any(s.get("interpolated") for s in series):
        warnings.append("Seeds have mismatched load points. The mean and sample standard deviation use interpolation within their common load range; no extrapolation is used.")
    if seed == "all" and any(s["seeds"] == 1 for s in series):
        warnings.append("Groups with only one available seed have no sample standard deviation or uncertainty band.")
    if len(runs) < len(selected):
        warnings.append(f"{len(selected) - len(runs)} of {len(selected)} selected campaign runs are absent from this local collection; available seeds may be incomplete.")
    if excluded:
        warnings.append(f"{excluded} local runs were excluded because the required data or eligible timing samples were unavailable.")
    if partial:
        warnings.append(f"{partial} included runs have no termination record; the plotted data is partial.")
    return {"mode": "curve" if curve else "scalar", "series": series, "warnings": warnings,
            "runs": sum(s["seeds"] for s in series), "expected": len(selected)}


def prepared_plot(campaign, root, spec):
    # Cache only reduced numerical data, never the much larger raw CSV tables.
    campaign, root = Path(campaign).resolve(), Path(root).resolve()
    _, expected, _ = definition(campaign)
    paths = [campaign, campaign.with_name("metadata.json")]
    paths += [folder / name for folder in matching_folders(root, expected)
              for name in ("config.conf", "manifest.json", "macroData.csv", "termination.json")]
    stamps = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else (str(p), None, None) for p in paths)
    selection = json.dumps({k: v for k, v in spec.items() if k not in APPEARANCE and k not in {"campaign", "data"}}, sort_keys=True)
    return _prepared(str(campaign), str(root), selection, stamps)


@lru_cache(maxsize=4)
def _prepared(campaign, root, selection, stamps):
    schema = describe(campaign, [root], root)
    return schema, prepare(campaign, root, json.loads(selection), schema)


def validate_axes(result, spec):
    warnings = []
    for axis, field in (("x", "x"), ("y", "mean")):
        scale = spec.get(axis + "scale", "linear")
        if scale not in {"linear", "log"}:
            raise ValueError(f"Unknown {axis.upper()} axis scale.")
        values = [v for s in result["series"] for v in (s[field] if isinstance(s[field], list) else [s[field]])]
        if scale == "log" and any(type(v) not in (int, float) or v <= 0 for v in values):
            raise ValueError(f"Logarithmic {axis.upper()} requires strictly positive numeric values.")
        limits = spec.get(axis + "lim")
        if limits is not None and (len(limits) != 2 or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]
                                   or (scale == "log" and limits[0] <= 0)):
            raise ValueError(f"Invalid {axis.upper()} limits for the selected scale.")
    if spec.get("yscale") == "log" and any(s["sd"] is not None and np.any(np.array(s["mean"]) - s["sd"] <= 0) for s in result["series"]):
        warnings.append("Nonpositive uncertainty bounds are not visible on logarithmic Y.")
    return warnings


def legend_label(schema, spec, category, shade):
    style = spec.get("legend_style", "compact")
    if style not in {"compact", "full", "custom"}:
        raise ValueError("Unknown legend label style.")
    labels = schema["labels"] if style == "full" else schema["abbreviations"] | (spec.get("legend_aliases", {}) if style == "custom" else {})
    return ", ".join((f"{labels[spec[k]]}={v}" if labels[spec[k]] else str(v))
                     for k, v in (("color", category), ("gradient", shade)) if spec.get(k)) or schema["quantities"][spec["y"]]["label"]


def plot(campaign, root, path, spec):
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from .style import STYLE, figure
    schema, cached = prepared_plot(campaign, root, spec)
    result = cached | {"warnings": cached["warnings"] + validate_axes(cached, spec)}
    path = Path(path)
    if path.suffix not in {".pdf", ".png"}:
        raise ValueError("Figures require PDF or PNG output.")
    categories = sorted({s["color"] for s in result["series"]})
    shades = sorted({s["gradient"] for s in result["series"]})
    positions = sorted({s["x"] for s in result["series"]}) if result["mode"] == "scalar" else []
    categorical = any(type(x) not in (int, float) for x in positions)
    coordinates = {x: i if categorical else x for i, x in enumerate(positions)}
    with mpl.rc_context(STYLE):
        fig, ax = figure(width=4.8, height=3.1)
        try:
            for i, category in enumerate(categories):
                for j, shade in enumerate(shades):
                    members = [s for s in result["series"] if s["color"] == category and s["gradient"] == shade]
                    if not members:
                        continue
                    ink = np.array((mpl.colormaps["tab10"](i) if len(categories) <= 10 else
                                    mpl.colormaps["turbo"](i / (len(categories) - 1)))[:3])
                    tint = .9 if len(shades) == 1 else .4 + .6 * j / (len(shades) - 1)
                    ink = 1 - tint * (1 - ink)
                    label = legend_label(schema, spec, category, shade)
                    if result["mode"] == "curve":
                        item = members[0]
                        ax.plot(item["x"], item["mean"], color=ink, label=label)
                        if item["sd"] is not None:
                            mean, sd = np.array(item["mean"]), np.array(item["sd"])
                            ax.fill_between(item["x"], mean - sd, mean + sd, color=ink, alpha=.18, linewidth=0)
                    else:
                        members = sorted(members, key=lambda s: coordinates[s["x"]])
                        xs = [coordinates[s["x"]] for s in members]
                        ax.plot(xs, [s["mean"] for s in members], marker="o", markerfacecolor="none", color=ink, label=label)
                        replicated = [s for s in members if s["sd"] is not None]
                        if replicated:
                            ax.errorbar([coordinates[s["x"]] for s in replicated], [s["mean"] for s in replicated],
                                        yerr=[s["sd"] for s in replicated], fmt="none", color=ink, capsize=2, linewidth=.7)
            if categorical:
                ax.set_xticks(range(len(positions)), [str(x) for x in positions])
            ylabel = schema["quantities"][spec["y"]]["label"]
            if result["mode"] == "scalar" and schema["quantities"][spec["y"]]["kind"] == "macro":
                ylabel = REDUCTIONS[spec.get("reduction", "last")] + ": " + ylabel
            normalization = spec.get("normalization", "none")
            if normalization != "none":
                ylabel += " / " + ("L" if normalization == "L" else "(rows × cols)")
            xlabel = schema["labels"].get(spec["x"], schema["quantities"].get(spec["x"], {}).get("label", spec["x"]))
            ax.set(xlabel=xlabel if spec.get("xlabel") is None else spec["xlabel"],
                   ylabel=ylabel if spec.get("ylabel") is None else spec["ylabel"])
            for axis in ("x", "y"):
                scale = spec.get(axis + "scale", "linear")
                getattr(ax, "set_" + axis + "scale")(scale, **({"nonpositive": "mask"} if scale == "log" else {}))
            for key in ("xlim", "ylim"):
                if spec.get(key) is not None:
                    getattr(ax, "set_" + key)(spec[key])
            ax.legend(loc=spec.get("legend_loc", "best"), ncols=spec.get("ncols", 1), frameon=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight", dpi=120)
        finally:
            plt.close(fig)
    return result


def preview(campaign, root, path, spec):
    import os
    import time
    started = time.monotonic()
    path = Path(path)
    temporary = path.with_name(f".preview-{os.getpid()}.png")
    result = plot(campaign, root, temporary, spec)
    temporary.replace(path)
    return {k: result[k] for k in ("warnings", "runs", "expected", "mode")} | {"seconds": time.monotonic() - started}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("options", help="JSON plot selections")
    args = parser.parse_args()
    result = plot(args.campaign, args.data, args.output, json.loads(args.options))
    print(json.dumps({k: result[k] for k in ("warnings", "runs", "expected")}))
