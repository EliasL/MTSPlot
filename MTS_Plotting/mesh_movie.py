"""Render native MTS2D mesh snapshots directly to MP4."""

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from itertools import islice
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from MTS_Management.data import MESH_SNAPSHOT, macro_reader, mesh_snapshots, read_vtu
from MTS_Management.errors import EXPECTED_ERROR_EXIT, EXPECTED_ERROR_FORMAT, ExpectedError

from .dislocations import DEFAULT_BOND_TOLERANCE, delaunay_dislocations, dislocation_candidates, dislocation_segments
from .mesh_geometry import (integer_shears, periodic_basis, poincare_coordinates,
                            reference_cell, visible_periodic_images)
from .render_capacity import automatic_workers, peak_memory
from .runs import Run

MATRICES = ("T", "T_p", "F", "F_E", "F_P")
STRESS_COMPONENTS = {"11": "xx", "12": "xy", "22": "yy"}
HEATMAPS = ("inferno", "viridis", "magma", "coolwarm")
VIDEO_DPI = (100, 200, 300)
NODE_COLOURS = {"Fixed": "#2a3857", "Free": "#d24646", "Displaced": "#d69220"}
ENERGY_RANGE_PROGRESS = "Animation energy range: "


def frame_schedule(frames, *, playback, fps, seconds_per_load, minimum_seconds):
    if not frames:
        raise ExpectedError("No mesh snapshots are available. Choose a run with saved mesh output.")
    loads = np.array([frame["load"] for frame in frames])
    if playback not in {"load", "saved"}:
        raise ValueError("Unknown playback mode.")
    if (not np.all(np.isfinite(loads)) or type(fps) is not int or not 1 <= fps <= 60 or
            not np.isfinite(seconds_per_load) or seconds_per_load <= 0 or
            not np.isfinite(minimum_seconds) or minimum_seconds <= 0):
        raise ValueError("Use finite playback settings, a positive duration and 1–60 frames per second.")
    distance = np.r_[0., np.cumsum(np.abs(np.diff(loads)))]
    if playback == "saved" or distance[-1] == 0:
        count = max(len(frames), math.ceil(minimum_seconds * fps))
        return np.floor(np.arange(count) * len(frames) / count).astype(int)
    count = max(2, math.ceil(max(minimum_seconds, seconds_per_load * distance[-1]) * fps))
    targets = np.linspace(0., distance[-1], count)
    right = np.searchsorted(distance, targets).clip(1, len(frames) - 1)
    left = right - 1
    indices = np.where(targets - distance[left] <= distance[right] - targets, left, right)
    indices[0], indices[-1] = 0, len(frames) - 1
    return indices


def preview_frame_index(schedule, position):
    offsets = {"first": 0, "middle": len(schedule) // 2, "last": -1}
    if position not in offsets:
        raise ValueError("Choose the first, middle, or last animation frame.")
    return schedule[offsets[position]]


def encoder_path(preferred=None):
    search = os.pathsep.join((os.environ.get("PATH", os.defpath), "/opt/homebrew/bin", "/usr/local/bin"))
    if not (path := shutil.which(preferred or os.environ.get("MTS_FFMPEG") or "ffmpeg", path=search)):
        raise ExpectedError("Install FFmpeg and reopen MTS to export MP4 animations.")
    try:
        result = subprocess.run([path, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ExpectedError(f"Could not start FFmpeg at {path}. Check the installation.") from error
    if result.returncode or not any(line.split()[1:2] == ["libx264"] for line in result.stdout.splitlines()):
        detail = next((line for line in result.stderr.splitlines() if line.strip()), "H.264 encoder unavailable.")
        raise ExpectedError(f"Repair the FFmpeg installation or configure a working H.264 encoder. {detail}")
    return path


def read_frame(folder, frame, kind, matrix, config, *, stress_component="12"):
    mesh = read_vtu(folder / "data" / frame["name"])
    points = np.asarray(mesh.mesh.points[:, :2], dtype=float)
    triangles = points[mesh.cells]
    if not len(triangles) or not np.all(np.isfinite(triangles)):
        raise ValueError(f"Invalid mesh coordinates in {frame['name']}.")
    if kind == "energy":
        values = mesh.field("energy_field")[:, None]
    elif kind == "stress":
        field = "sigma" + stress_component
        if field not in mesh.mesh.cell_data:
            raise ExpectedError(f"{frame['name']} lacks Cauchy stress σ{STRESS_COMPONENTS[stress_component]}. "
                                "Choose a component available in the saved mesh fields.")
        values = mesh.field(field)[:, None]
    elif kind == "shear":
        branch = "F_P" if config["reconnectionMethod"] == "none" else "T_p"
        values = integer_shears(mesh.matrix(branch))
    elif kind == "matrix":
        values = mesh.matrix(matrix).reshape(-1, 4)
    elif kind == "disk":
        values = poincare_coordinates(mesh.matrix(matrix))
    else:
        values = None
    if values is not None and (values.shape[0] != len(triangles) or not np.all(np.isfinite(values))):
        raise ValueError(f"Invalid animation field in {frame['name']}.")
    return mesh, points, triangles, values


def triangle_area(triangles):
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2


def node_classes(point_data):
    """Displaced nodes are prescribed (fixed) nodes moved from their reference positions."""
    if "fixed" not in point_data or "displacement" not in point_data:
        raise ExpectedError("Node colours require snapshots with fixed-node and displacement fields.")
    fixed = np.asarray(point_data["fixed"]).reshape(-1)
    displacement = np.asarray(point_data["displacement"])
    if (displacement.shape != (len(fixed), 3) or not np.all(np.isfinite(displacement))
            or not np.all(np.isin(fixed, (0, 1)))):
        raise ValueError("Invalid fixed-node or displacement field.")
    moved = np.any(np.abs(displacement) > 1e-10, axis=1)
    return np.where(fixed == 1, np.where(moved, "Displaced", "Fixed"), "Free")


def frame_labels(frame, index, mesh, areas):
    energy = mesh.field("energy_field")
    stress = np.average(mesh.field("sigma12"), weights=areas)
    plastic = int(MESH_SNAPSHOT.fullmatch(frame["name"])["plastic"])
    return [rf"$\gamma$: {frame['load']:.5f}", rf"$E$: {energy.sum():.4g}",
            rf"$\langle E_i \rangle$: {energy.mean():.3e}", rf"$\langle\sigma_{{12}}\rangle$: {stress:.3f}",
            rf"$N_p$: {plastic}", f"Step: {frame['step']}", f"f: {index + 1}"]


def frame_crop(fig, axes, header):
    """Crop the fixed artist extents in pixel space, with even H.264 dimensions."""
    renderer = fig.canvas.get_renderer()
    if header:
        boxes = [text.get_window_extent(renderer).transformed(fig.transFigure.inverted()) for text in header]
        widths = [box.width for box in boxes]
        centre = (min(ax.get_position().x0 for ax in axes) + max(ax.get_position().x1 for ax in axes)) / 2
        y = max(ax.get_position().y1 for ax in axes) + .014
        split = [slice(None)] if sum(widths) + .09 <= .94 else [slice(4, None), slice(0, 4)]
        for row in split:
            gap = .015
            width = sum(widths[row]) + gap * (len(widths[row]) - 1)
            x = min(max(centre - width / 2, .01), .99 - width)
            for text, size in zip(header[row], widths[row], strict=True):
                text.set_position((x + size / 2, y))
                x += size + gap
            y += max(box.height for box in boxes[row]) + .012
    box = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    width, height = fig.canvas.get_width_height()
    pad = max(4, round(.04 * fig.dpi))
    x0, y0 = np.maximum(0, np.floor(box.min).astype(int) - pad)
    x1, y1 = np.minimum([width, height], np.ceil(box.max).astype(int) + pad)
    return int(x0), int(height - y1), int((x1 - x0) // 2 * 2), int((y1 - y0) // 2 * 2)


def opacity_values(positions, opacity, curve):
    if curve is None:
        return np.full_like(positions, opacity, dtype=float)
    points = np.asarray(curve, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or not len(points) or
            not np.all((0 <= points) & (points <= 1)) or np.any(np.diff(points[:, 0]) <= 0)):
        raise ValueError("Opacity nodes require increasing energy positions and values between zero and one.")
    return np.interp(positions, points[:, 0], points[:, 1])


def energy_limits(minimum, maximum, lower_limit, upper_limit):
    lower = min(0., minimum) if lower_limit is None else lower_limit
    upper = maximum if upper_limit is None else upper_limit
    if upper_limit is None and upper <= lower and (lower_limit is None or lower == 0):
        upper = lower + 1.
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ExpectedError("Choose an upper energy limit greater than the lower limit.")
    return lower, upper


def csv_display_limits(folder, frames, *, bounds_needed, energy_max_needed):
    """Calculate fixed display limits from matching accepted-state rows."""
    if not bounds_needed and not energy_max_needed:
        return None, None
    selected = {}
    for frame in frames:
        step = frame["step"]
        selected[step] = None if step in selected else frame["load"]
    data = {}
    coordinates = ("minX", "minY", "maxX", "maxY")
    try:
        with (folder / "macroData.csv").open(newline="") as stream:
            reader = macro_reader(stream, allow_incomplete_tail=True)
            columns = reader.fieldnames or ()
            if len(set(columns)) != len(columns) or not {"load_step", "load"} <= set(columns):
                return None, None
            bounds_needed = bounds_needed and set(coordinates) <= set(columns)
            energy_max_needed = energy_max_needed and "max_energy" in columns
            names = (coordinates if bounds_needed else ()) + (("max_energy",) if energy_max_needed else ())
            if not names:
                return None, None
            for row in reader:
                step = int(row["load_step"])
                if step not in selected:
                    continue
                if step in data or selected[step] is None or not math.isclose(float(row["load"]), selected[step],
                                                    rel_tol=1e-10, abs_tol=5.01e-7):
                    data[step] = None
                else:
                    data[step] = [float(row[name]) for name in names]
    except (FileNotFoundError, TypeError, ValueError):
        return None, None
    matched = [row for row in data.values() if row is not None]
    bounds = None
    if bounds_needed:
        matches = [row[:4] for row in matched if all(math.isfinite(value) for value in row[:4])
                   and row[2] >= row[0] and row[3] >= row[1]]
        if matches:
            values = np.array(matches)
            bounds = np.array([values[:, :2].min(axis=0), values[:, 2:4].max(axis=0)])
            # Expand the CSV's rounded bounds so zero padding keeps the mesh visible.
            bounds += np.array([[-1.], [1.]]) * (1e-10 * np.abs(bounds))
    maximum = None
    # The native accumulator uses zero as an upper bound for nonpositive energies.
    if energy_max_needed:
        maximum = max((row[-1] for row in matched if math.isfinite(row[-1]) and row[-1] >= 0),
                      default=None)
    return bounds, maximum


def tiled_display(config, kind, tiling):
    return kind != "disk" and (tiling == "tile" or (tiling == "auto" and config["boundary"] == "periodic"))


def display_limit_sources(folder, frames, *, kind, tiled, limit, lower_limit, range_from_last_frame=False):
    bounds_needed = kind != "disk" and not tiled
    energy_max_needed = kind == "energy" and limit is None
    bounds, maximum = csv_display_limits(folder, frames, bounds_needed=bounds_needed,
                                        energy_max_needed=energy_max_needed and not range_from_last_frame)
    return bounds, maximum, (
        bounds_needed and bounds is None,
        kind == "energy" and lower_limit is None,
        energy_max_needed and maximum is None,
        kind in {"matrix", "shear", "stress"} and limit is None,
    )


def display_scan_frames(frames, scans, *, range_from_last_frame):
    scan_bounds, scan_min, scan_max, scan_signed = scans
    if not scan_bounds and range_from_last_frame:
        frames = frames[-1:]
    for index, frame in enumerate(frames):
        colours = (scan_min or scan_max or scan_signed) and (
            not range_from_last_frame or index == len(frames) - 1)
        if scan_bounds or colours:
            yield frame, colours


def frame_renderer(*, folder, config, kind, matrix, cell, bounds, tiled, coloured_mesh,
                   with_nodes, with_edges, node_scale, node_colours, cmap, limit, minimum, maximum,
                   dpi, show_legend, legend_scale, show_border, show_title, title_size,
                   defect_lattice, defect_method, bond_tolerance, dislocation_scale, dislocation_width,
                   energy_range, opacity, opacity_curve, stress_component):
    from matplotlib import colormaps
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    rows, cols = {"matrix": (2, 2), "shear": (1, 2)}.get(kind, (1, 1))
    if kind != "disk" and (not np.all(np.isfinite(bounds)) or np.any(bounds[1] <= bounds[0])):
        raise ValueError("The selected mesh has invalid display bounds.")
    fig, axes = plt.subplots(rows, cols, figsize=(9.6, 5.4), dpi=dpi, layout="constrained")
    placeholders = [r"$\gamma$: -999.99999", "$E$: -1.234e+308", r"$\langle E_i\rangle$: -1.234e+308",
                    r"$\langle\sigma_{12}\rangle$: -999.999", r"$N_p$: 999999", "Step: 99999999", "f: 999999"]
    header = [fig.text(0, .97, label, ha="center", va="bottom", fontsize=title_size)
              for label in placeholders] if show_title else []
    fig.get_layout_engine().set(rect=(0, 0, 1, .85 if show_title else 1))
    axes = np.asarray(axes).reshape(-1)
    artists = []
    colourbars = []
    if kind == "disk":
        ax = axes[0]
        ax.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02))
        ax.add_patch(Circle((0, 0), 1, fill=False, linewidth=.8))
        if config["energyFunction"] == "contiSquare":
            from MTM_Math.energy import ContiEnergy

            x, y = np.meshgrid(*[np.linspace(-1, 1, 161)] * 2)
            r2 = x * x + y * y
            mask = r2 < .97 ** 2
            d = 1 - r2[mask]
            background = np.full(x.shape, np.nan)
            background[mask] = ContiEnergy.energy_from_C_components(
                (1 + r2[mask] + 2 * x[mask]) / d, (1 + r2[mask] - 2 * x[mask]) / d,
                2 * y[mask] / d, K=config["bulkModulus"])
            ax.contour(x, y, background, levels=(.01, .03, .06, .1), colors=".78", linewidths=.55)
        artist = ax.scatter([], [], s=2, linewidths=0, color="#23608e", alpha=.3)
    elif coloured_mesh:
        magnitude = limit or max(abs(minimum), abs(maximum)) or 1.
        norm = Normalize(*energy_range) if kind == "energy" else Normalize(-magnitude, magnitude)
        if opacity_curve is not None or opacity != 1:
            positions = np.linspace(0, 1, 1024)
            colours = colormaps[cmap](positions)
            colours[:, 3] = opacity_values(positions, opacity, opacity_curve)
            cmap = ListedColormap(colours)
        if kind == "shear":
            magnitude = int(np.ceil(magnitude))
            step = max(1, int(np.ceil(magnitude / 63)))
            norm = BoundaryNorm(np.arange(-magnitude - .5, magnitude + step + .5, step), 256)
        for ax in axes:
            collection = PolyCollection([], cmap=cmap, norm=norm, edgecolors="none", antialiaseds=False)
            ax.add_collection(collection)
            artists.append(collection)
        if show_legend and kind == "shear":
            for ax, collection, direction in zip(axes, artists, ("h", "v"), strict=True):
                colourbars.append(fig.colorbar(collection, ax=ax, shrink=.76, label=rf"$n_{direction}$"))
        elif show_legend:
            label = {"energy": r"$E_i$", "stress": rf"$\sigma_{{{STRESS_COMPONENTS[stress_component]}}}$"}.get(
                kind, rf"$({matrix})_{{ij}}$")
            colourbars.append(fig.colorbar(artists[0], ax=axes.tolist(), orientation="horizontal", shrink=.5, label=label))
    edge_artists, node_artists, defect_artists = [], [], []
    node_radius = 3 / 7 * node_scale if with_nodes else 0.
    for ax in axes:
        ax.set(aspect="equal", xticks=[], yticks=[])
        if kind != "disk":
            display_bounds = bounds + np.array([[-node_radius], [node_radius]])
            ax.set(xlim=display_bounds[:, 0], ylim=display_bounds[:, 1])
            for spine in ax.spines.values():
                spine.set_visible(show_border)
                spine.set_edgecolor("black")
        if with_edges:
            lines = PolyCollection([], facecolors="none", edgecolors="#303642", linewidths=.35, zorder=2)
            ax.add_collection(lines)
            edge_artists.append(lines)
        if with_nodes:
            node_artists.append(ax.scatter([], [], linewidths=0, zorder=3))
        if defect_lattice is not None:
            from matplotlib.patheffects import Normal, Stroke

            symbols = LineCollection([], colors="#101010", linewidths=dislocation_width, zorder=4,
                                     path_effects=[Stroke(linewidth=2 * dislocation_width, foreground="white"), Normal()])
            ax.add_collection(symbols)
            defect_artists.append(symbols)
    if with_nodes and show_legend:
        handles = [Line2D([], [], marker="o", linestyle="none", color=colour, label=label)
                   for label, colour in node_colours.items()]
        axes[0].legend(handles=handles, loc="upper right", fontsize=5 * legend_scale,
                       markerscale=legend_scale)
    for bar in colourbars:
        bar.ax.tick_params(labelsize=5 * legend_scale)
        label = bar.ax.xaxis.label if bar.orientation == "horizontal" else bar.ax.yaxis.label
        label.set_fontsize(5 * legend_scale)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    x0, y0, width, height = frame_crop(fig, axes, header)
    for ax, nodes in zip(axes, node_artists):
        # One reference spacing covers two radii plus a gap of one third of a radius.
        pixels = np.linalg.norm(ax.transData.transform((1, 0)) - ax.transData.transform((0, 0)))
        diameter = 2 * node_radius * pixels * 72 / fig.dpi
        nodes.set_sizes([diameter ** 2])

    def draw(index, frame):
        mesh, points, polygons, values = read_frame(folder, frame, kind, matrix, config,
                                                    stress_component=stress_component)
        areas = triangle_area(polygons)
        if header:
            for text, label in zip(header, frame_labels(frame, index, mesh, areas), strict=True):
                text.set_text(label)
        if kind == "disk":
            artist.set_offsets(values)
        else:
            indices = slice(None)
            periodic = config["boundary"] in {"periodic", "periodicAnchored"}
            if tiled or (defect_artists and periodic):
                area = np.linalg.det(cell)
                if config["deformation"] in {"compression", "tension"}:
                    area = areas.sum()
                ids = mesh.mesh.point_data.get("refIndex")
                if ids is None:
                    raise ExpectedError("The selected snapshots lack periodic reference node IDs.")
                basis = periodic_basis(points, ids, area)
            if tiled and (coloured_mesh or edge_artists):
                polygons, indices = visible_periodic_images(polygons, basis, display_bounds)
            if defect_artists:
                periodic_cell = (basis, ids) if periodic else None
                if defect_method == "delaunay":
                    positions, burgers = delaunay_dislocations(points, mesh.cells, defect_lattice,
                                                              periodic=periodic_cell, tiled=tiled, tolerance=bond_tolerance)
                else:
                    try:
                        elastic = mesh.matrix("F_E")
                    except KeyError as error:
                        detail = " Select Delaunay to use node positions." if config["reconnectionMethod"] == "none" else ""
                        raise ExpectedError("Native dislocation detection requires the saved elastic distortion field." + detail) from error
                    positions, burgers = dislocation_candidates(points, mesh.cells, elastic, defect_lattice,
                                                               periodic=periodic_cell)
                symbols = dislocation_segments(positions, burgers, dislocation_scale)
                if tiled and len(symbols):
                    symbols, _ = visible_periodic_images(symbols, basis, bounds)
                for collection in defect_artists:
                    collection.set_segments(symbols)
            for column, collection in enumerate(artists):
                collection.set_verts(polygons)
                collection.set_array(values[indices, column])
            for edges in edge_artists:
                edges.set_verts(polygons)
            if node_artists:
                classes = node_classes(mesh.mesh.point_data)
                if len(classes) != len(points):
                    raise ValueError("Node roles do not match the snapshot coordinates.")
                if tiled:
                    representatives = np.unique(ids, return_index=True)[1]
                    images, selected = visible_periodic_images(points[representatives, None, :], basis, bounds)
                    points, classes = images[:, 0], classes[representatives[selected]]
                colours = [node_colours[label] for label in classes]
                for nodes in node_artists:
                    nodes.set_offsets(points)
                    nodes.set_facecolors(colours)
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())
        return pixels[y0:y0 + height, x0:x0 + width].tobytes()

    return draw, lambda: plt.close(fig), (width, height)


def init_frame_worker(settings):
    global _draw_frame
    _draw_frame, _, _ = frame_renderer(**settings)


def render_frame(job):
    index, frame = job
    return _draw_frame(index, frame)


def rendered_frames(settings, jobs, workers, memory, count, renderer):
    jobs = iter(jobs)
    draw, close, _ = renderer
    if count == 1:
        workers = 1
    if workers is None and memory:
        workers = min(count, automatic_workers(memory))
        print(f"Using {workers} worker(s) from the preview memory estimate.", file=sys.stderr, flush=True)
    if workers is None:
        try:
            pixels = draw(*next(jobs))
            workers = min(count - 1, automatic_workers(peak_memory()))
            print(f"First frame measured. Using {max(1, workers)} worker(s) for the remaining frames.",
                  file=sys.stderr, flush=True)
            yield pixels
            if workers <= 1:
                for job in jobs:
                    yield draw(*job)
                return
        finally:
            close()
    if workers == 1:
        try:
            for index, frame in jobs:
                yield draw(index, frame)
        finally:
            close()
        return

    close()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                             initializer=init_frame_worker, initargs=(settings,)) as pool:
        jobs = iter(jobs)
        # Keep at most one outstanding frame per worker and preserve playback order.
        pending = deque(pool.submit(render_frame, job) for job in islice(jobs, workers))
        while pending:
            pixels = pending.popleft().result()
            if (job := next(jobs, None)) is not None:
                pending.append(pool.submit(render_frame, job))
            yield pixels


def render(folder, output, spec, frames=None):
    preview = spec.get("preview", False)
    ffmpeg = None if preview else encoder_path()
    run = Run.read(folder)
    folder = run.folder
    config = {key: run.config[key] for key in ("rows", "cols", "boundary", "energyFunction",
                                               "bulkModulus", "reconnectionMethod", "deformation")}
    kind, matrix, tiling = spec["kind"], spec["matrix"], spec["tiling"]
    stress_component = spec.get("stress_component", "12")
    if kind not in {"none", "energy", "shear", "matrix", "disk", "stress"} or matrix not in MATRICES:
        raise ValueError("Unknown animation type or matrix.")
    if stress_component not in STRESS_COMPONENTS:
        raise ValueError("Choose a Cauchy stress component: xx, xy or yy.")
    coloured_mesh = kind in {"energy", "shear", "matrix", "stress"}
    with_nodes = kind != "disk" and spec.get("show_nodes", False)
    with_edges = kind != "disk" and spec.get("show_mesh_lines", False)
    dislocations = kind != "disk" and spec.get("dislocations", False)
    if kind == "none" and not (with_nodes or with_edges or dislocations):
        raise ExpectedError("Enable nodes, mesh lines or dislocations when no field is selected.")
    node_scale = spec.get("node_size", 100) / 200
    legend_scale = spec.get("legend_size", 100) / 50
    if with_nodes and (not np.isfinite(node_scale) or node_scale <= 0):
        raise ValueError("Node size must be a finite, positive percentage.")
    if not np.isfinite(legend_scale) or legend_scale <= 0:
        raise ValueError("Legend size must be a finite, positive percentage.")
    workers = spec.get("workers")
    if workers is not None and (type(workers) is not int or not 1 <= workers <= 8):
        raise ValueError("Choose between one and eight frame workers.")
    cmap = spec.get("heatmap") or ("inferno" if kind == "energy" else "coolwarm")
    dpi = spec.get("dpi", 200)
    show_title, title_size = spec.get("show_title", True), spec.get("title_size", 8)
    show_border = spec.get("show_border", True)
    show_legend = spec.get("show_legend", True)
    dislocation_scale = spec.get("dislocation_scale", 100) / 100
    dislocation_thickness = spec.get("dislocation_thickness", 100) / 100
    lattice = spec.get("dislocation_lattice", "auto")
    if lattice not in {"auto", "square", "triangular"}:
        raise ValueError("Choose a square or triangular dislocation lattice.")
    defect_method = spec.get("dislocation_method", "native")
    if defect_method not in {"native", "delaunay"}:
        raise ValueError("Choose the native mesh or Delaunay dislocation detector.")
    if dislocations and defect_method == "delaunay" and config["reconnectionMethod"] != "none":
        raise ExpectedError("Choose the native mesh detector for a run with reconnection.")
    triangular = lattice == "triangular" or lattice == "auto" and config["energyFunction"] == "contiTriangular"
    defect_lattice = (np.array([[1, .5], [0, np.sqrt(3) / 2]]) if triangular else np.eye(2)) if dislocations else None
    node_colours = spec.get("node_colours", NODE_COLOURS)
    if (cmap not in HEATMAPS or dpi not in VIDEO_DPI or not 4 <= title_size <= 16
            or not 0 < dislocation_scale <= 10 or not 0 < dislocation_thickness <= 10
            or node_colours.keys() != NODE_COLOURS.keys()
            or any(not re.fullmatch(r"#[0-9a-fA-F]{6}", colour) for colour in node_colours.values())):
        raise ValueError("Invalid DPI, title size, dislocation scale, heatmap or node colours.")
    if kind == "shear" and config["energyFunction"] != "contiSquare":
        raise ExpectedError("Integer shear is defined for the square energy model.")
    periodic = config["boundary"] in {"periodic", "periodicAnchored"}
    if tiling not in {"auto", "tile", "cell"}:
        raise ValueError("Unknown tiling option.")
    if kind != "disk" and tiling == "tile" and not periodic:
        raise ExpectedError("Tiling requires periodic boundary conditions.")
    tiled = tiled_display(config, kind, tiling)
    frames = mesh_snapshots(folder) if frames is None else frames
    fps = spec["fps"]
    schedule = frame_schedule(frames, playback=spec["playback"], fps=fps,
                              seconds_per_load=spec["seconds_per_load"], minimum_seconds=spec["minimum_seconds"])
    scheduled = [frames[index] for index in dict.fromkeys(schedule)]
    counts = Counter(schedule)
    if preview:
        counts = {preview_frame_index(schedule, spec.get("preview_frame", "first")): 1}
    limit = spec.get("limit")
    range_from_last_frame = spec.get("range_from_last_frame", True)
    lower_limit = spec.get("energy_min", 0.) if kind == "energy" else None
    if limit is not None and (not np.isfinite(limit) or kind != "energy" and limit <= 0):
        raise ValueError("The colour limit must be finite, and positive for a signed field.")
    opacity = spec.get("opacity", 100) / 100
    opacity_curve = spec.get("opacity_curve") if kind == "energy" else None
    if not 0 <= opacity <= 1:
        raise ValueError("Choose an opacity between zero and 100%.")
    opacity_values(np.array([0.]), opacity, opacity_curve)
    cell = reference_cell(config)
    corners = np.array([[0, 0], cell[:, 0], cell[:, 1], cell.sum(axis=1)])
    bounds = np.array([corners.min(axis=0), corners.max(axis=0)]) if tiled else np.array([[np.inf] * 2, [-np.inf] * 2])
    minimum, maximum = np.inf, -np.inf
    csv_bounds, csv_maximum, scans = display_limit_sources(
        folder, scheduled, kind=kind, tiled=tiled, limit=limit, lower_limit=lower_limit,
        range_from_last_frame=range_from_last_frame)
    if csv_bounds is not None:
        bounds = csv_bounds
    if csv_maximum is not None:
        maximum = csv_maximum
    scan_bounds, scan_min, scan_max, scan_signed = scans
    scan_frames = list(display_scan_frames(scheduled, scans, range_from_last_frame=range_from_last_frame))
    if scan_frames:
        signed_field = {"matrix": f"{matrix} matrix components", "shear": "integer shear",
                        "stress": f"Cauchy stress σ{STRESS_COMPONENTS[stress_component]}"}
        quantities = [label for needed, label in (
            (scan_bounds, "mesh bounds"), (scan_min, "minimum element energy"),
            (scan_max, "maximum element energy"),
            (scan_signed, f"maximum absolute {signed_field.get(kind, '')}"),
        ) if needed]
        print(f"Scanning {len(scan_frames)} mesh states for fixed display limits ({', '.join(quantities)}).",
              file=sys.stderr, flush=True)
        for frame, scan_colours in scan_frames:
            _, _, polygons, values = read_frame(folder, frame, kind if scan_colours else "none", matrix, config,
                                                stress_component=stress_component)
            if scan_bounds:
                bounds[0] = np.minimum(bounds[0], polygons.min(axis=(0, 1)))
                bounds[1] = np.maximum(bounds[1], polygons.max(axis=(0, 1)))
            if scan_colours and (scan_min or scan_signed):
                minimum = min(minimum, values.min())
            if scan_colours and (scan_max or scan_signed):
                maximum = max(maximum, values.max())
    if kind != "disk" and show_border:
        bounds += np.array([[-.02], [.02]]) * (bounds[1] - bounds[0])

    energy_range = energy_limits(minimum, maximum, lower_limit, limit) if kind == "energy" else None
    if energy_range is not None and (lower_limit is None or limit is None):
        scanned = [value if override is None else None
                   for value, override in zip(energy_range, (lower_limit, limit), strict=True)]
        print(ENERGY_RANGE_PROGRESS + json.dumps(scanned), file=sys.stderr, flush=True)
    settings = dict(folder=folder, config=config, kind=kind, matrix=matrix, cell=cell, bounds=bounds,
                    tiled=tiled, coloured_mesh=coloured_mesh, with_nodes=with_nodes, with_edges=with_edges,
                    node_scale=node_scale, node_colours=node_colours, cmap=cmap, limit=limit,
                    minimum=minimum, maximum=maximum, dpi=dpi, show_legend=show_legend,
                    legend_scale=legend_scale, show_border=show_border,
                    show_title=show_title, title_size=title_size,
                    defect_lattice=defect_lattice, defect_method=defect_method,
                    bond_tolerance=spec.get("dislocation_bond_tolerance", DEFAULT_BOND_TOLERANCE),
                    dislocation_scale=.5 * dislocation_scale,
                    dislocation_width=1.25 * 1.3 * dislocation_thickness,
                    energy_range=energy_range, opacity=opacity, opacity_curve=opacity_curve,
                    stress_component=stress_component)
    if workers is not None:
        workers = min(workers, len(counts), os.cpu_count() or 1)
    jobs = ((index, frames[index]) for index in counts)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = frame_renderer(**settings)
    width, height = renderer[2]
    if preview:
        from matplotlib.image import imsave

        with closing(rendered_frames(settings, jobs, 1, None, 1, renderer)) as rendered:
            pixels = next(rendered)
            imsave(output, np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4))
        return {"frame_memory": peak_memory(), "energy_range": energy_range}
    temporary = output.with_name(output.stem + ".partial.mp4")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgba",
               "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "-", "-an",
               "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder, \
                closing(rendered_frames(settings, jobs, workers, spec.get("frame_memory"), len(counts), renderer)) as rendered:
            try:
                for number, (pixels, copies) in enumerate(zip(rendered, counts.values(), strict=True), 1):
                    if len(pixels) != width * height * 4:
                        raise ValueError("A frame worker returned an inconsistent crop.")
                    for _ in range(copies):
                        encoder.stdin.write(pixels)
                    if number == 1 or number % 25 == 0 or number == len(counts):
                        print(f"Rendered {number}/{len(counts)} mesh states.", file=sys.stderr, flush=True)
            except BrokenPipeError as error:
                raise ExpectedError("FFmpeg stopped before all frames were encoded. See the terminal for details.") from error
            finally:
                try:
                    encoder.stdin.close()
                except BrokenPipeError:
                    pass
            if encoder.wait():
                raise RuntimeError("FFmpeg could not encode the animation.")
        temporary.replace(output)
    finally:
        renderer[1]()
        temporary.unlink(missing_ok=True)
    return {"frames": len(schedule), "states": len(counts), "seconds": len(schedule) / fps,
            "energy_range": energy_range}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("options", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.options.read_text())
    try:
        result = render(args.folder, args.output, spec, spec.pop("frames", None))
    except ExpectedError as error:
        if os.environ.get(EXPECTED_ERROR_FORMAT) != "json":
            raise
        print(json.dumps({"expected_error": str(error)}))
        return EXPECTED_ERROR_EXIT
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
