"""Simple numeric column plots, using explicit labels rather than name parsing."""

import matplotlib as mpl

from .style import STYLE, figure


def plot_macros(runs, *, x="load", y="avg_P12", xlabel="Load", ylabel=r"$\langle P_{12}\rangle$"):
    """Return (fig, ax); the caller chooses labels, limits and where to save."""
    with mpl.rc_context(STYLE):
        fig, ax = figure()
        for run in runs:
            rows = run.macros()
            if not rows:
                raise ValueError(f"Empty macro table: {run.folder}")
            ax.plot([float(row[x]) for row in rows], [float(row[y]) for row in rows],
                    label=run.folder.name.split("[")[0])
        ax.set(xlabel=xlabel, ylabel=ylabel)
    return fig, ax
