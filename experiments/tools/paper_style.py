#!/usr/bin/env python3
"""Shared figure style for the manuscript in ``docs/paper``.

Layout and styling follow the group's reference plotting implementation.  The
one deliberate departure is scale: figures are drawn at the manuscript text
width, so ``\\includegraphics[width=\\linewidth]`` neither enlarges nor shrinks
them and the type sizes set here are the sizes that reach the page.  A tight
bounding box is not used, because its size depends on the tick labels and would
give sibling figures different widths, different scale factors on the page, and
different effective type sizes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402


#: The manuscript is set by siamltex with \textwidth = 370.38 pt.
TEXT_WIDTH_IN = 370.38374 / 72.27


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(which="major", color="0.82", linewidth=0.65, alpha=0.75)
    axis.grid(which="minor", color="0.90", linewidth=0.45, alpha=0.45)
    axis.tick_params(direction="in", top=True, right=True, width=0.7, length=2.8)
    axis.tick_params(which="minor", direction="in", top=True, right=True, length=1.6)
    axis.set_axisbelow(True)


def thin_log_ticks(axis: plt.Axes, max_labels: int = 4) -> None:
    """Keep decade labels sparse enough for a panel a third of the text wide."""
    axis.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=max_labels))
    axis.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1))
    )
    axis.yaxis.set_minor_formatter(NullFormatter())


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    """Write one figure as PDF, PNG, and SVG with reproducible metadata."""
    outputs: list[Path] = []
    for extension, kwargs in (
        ("pdf", {"metadata": {"CreationDate": None}}),
        ("png", {"dpi": 300}),
        ("svg", {"metadata": {"Date": None}}),
    ):
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, **kwargs)
        outputs.append(path)
    return outputs


def check_width(path: Path, tolerance_pt: float = 1.5) -> None:
    """Fail loudly if a saved PDF would be rescaled by \\includegraphics."""
    target = TEXT_WIDTH_IN * 72.0
    blob = path.open("rb").read(4096)
    marker = b"/MediaBox [ "
    start = blob.index(marker) + len(marker)
    width = float(blob[start:blob.index(b"]", start)].split()[2])
    if abs(width - target) > tolerance_pt:
        raise ValueError(
            f"{path.name} is {width:.1f} pt wide but the text block is "
            f"{target:.1f} pt; adjust the figure width"
        )
