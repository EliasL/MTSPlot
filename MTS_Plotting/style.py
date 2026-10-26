"""Small figures; no global rcParams changes or external LaTeX installation."""

import matplotlib as mpl
from matplotlib import pyplot as plt

STYLE = {
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
    "axes.labelsize": 11, "legend.fontsize": 8, "legend.frameon": True,
    "legend.loc": "best", "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "lines.linewidth": 1.1, "lines.markersize": 5,
    "xtick.direction": "in", "ytick.direction": "in", "axes.grid": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
}


def figure(*, width=3.4, height=2.5):
    """Return (fig, ax), sized in inches for a journal column."""
    with mpl.rc_context(STYLE):
        return plt.subplots(figsize=(width, height), layout="constrained")
