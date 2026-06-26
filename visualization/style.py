"""
IEEE Access–compatible Matplotlib style configuration.

Fonts, colors, DPI, figure sizes, and rcParams are centralized here
so every figure in the suite uses identical typographic and visual conventions.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# IEEE Access figure sizing (column widths in inches)
# Single-column: 3.5 in   Double-column: 7.16 in   Full page: 9.5 × 11.69
# ---------------------------------------------------------------------------
COL1 = 3.5          # single column
COL2 = 7.16         # double column (text width)
PAGE_H = 9.0        # max page height (leave margin)

EXPORT_DPI = 300    # ≥ 300 dpi for print; 600 available but very large on screen

# ---------------------------------------------------------------------------
# Colorblind-safe palettes
# ---------------------------------------------------------------------------
# Okabe-Ito (colorblind-safe, 8 colors)
PALETTE_OI = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#000000",  # black
]

# Deformation-type canonical colors
DEFORMATION_COLORS = {
    "compression":  "#0072B2",   # blue
    "shearing":     "#D55E00",   # vermillion
    "stretching":   "#009E73",   # green
    "dissolution":  "#CC79A7",   # pink
}
DEFORMATION_LABELS = {
    "compression":  "Compression",
    "shearing":     "Shearing",
    "stretching":   "Stretching",
    "dissolution":  "Dissolution",
}

# Sequential colormap for intensity plots
CMAP_INTENSITY = "viridis"
CMAP_GREY      = "gray"
CMAP_HEAT      = "inferno"

# ---------------------------------------------------------------------------
# Specimen display names
# ---------------------------------------------------------------------------
SPECIMEN_DISPLAY = {
    "araripesaurus__BSPG-1982-I-90":       "Araripesaurus\n(BSPG-1982-I-90)",
    "archaeopteryx_london__BMNH-37001":    "Archaeopteryx\n(BMNH-37001)",
    "halszkaraptor__MPC-D-102-109":        "Halszkaraptor\n(MPC-D-102-109)",
    "hesperornis__YPM-1206-A":             "Hesperornis\n(YPM-1206-A)",
    "ichthyornis__YPM-1460":               "Ichthyornis\n(YPM-1460)",
    "ichthyornis__YPM-1775":               "Ichthyornis\n(YPM-1775)",
    "titanosaur_embryo__MCF-PVPH-874":     "Titanosaur Embryo\n(MCF-PVPH-874)",
    "tropeognathus__BSPG-1987-I-46":       "Tropeognathus\n(BSPG-1987-I-46)",
}
SPECIMEN_SHORT = {
    "araripesaurus__BSPG-1982-I-90":       "Araripesaurus",
    "archaeopteryx_london__BMNH-37001":    "Archaeopteryx",
    "halszkaraptor__MPC-D-102-109":        "Halszkaraptor",
    "hesperornis__YPM-1206-A":             "Hesperornis",
    "ichthyornis__YPM-1460":               "Ichthyornis (1460)",
    "ichthyornis__YPM-1775":               "Ichthyornis (1775)",
    "titanosaur_embryo__MCF-PVPH-874":     "Titanosaur Emb.",
    "tropeognathus__BSPG-1987-I-46":       "Tropeognathus",
}

VOXEL_SIZES = {
    "archaeopteryx_london__BMNH-37001":    13.0,
    "halszkaraptor__MPC-D-102-109":        107.16,
    "tropeognathus__BSPG-1987-I-46":       45.92,
    "araripesaurus__BSPG-1982-I-90":       45.92,
    "ichthyornis__YPM-1460":               1.28,
    "ichthyornis__YPM-1775":               1.28,
    "titanosaur_embryo__MCF-PVPH-874":     14.92,
    "hesperornis__YPM-1206-A":             3.50,
}

# ---------------------------------------------------------------------------
# Global rcParams for IEEE Access look
# ---------------------------------------------------------------------------
IEEE_RC = {
    # Font
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":            9,
    "axes.titlesize":       10,
    "axes.labelsize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    "figure.titlesize":     11,
    # Lines & markers
    "lines.linewidth":      1.2,
    "lines.markersize":     4,
    "patch.linewidth":      0.8,
    # Axes
    "axes.linewidth":       0.8,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.linewidth":       0.5,
    "grid.alpha":           0.4,
    "grid.color":           "#cccccc",
    # Figure
    "figure.dpi":           150,      # screen render; export uses EXPORT_DPI
    "savefig.dpi":          EXPORT_DPI,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    # PDF/SVG text
    "pdf.fonttype":         42,        # TrueType in PDF → selectable text
    "svg.fonttype":         "none",    # system fonts in SVG
}


def apply_ieee_style() -> None:
    """Apply IEEE Access–compatible rcParams globally."""
    mpl.rcParams.update(IEEE_RC)


def new_fig(width: float = COL2, height: float | None = None,
            nrows: int = 1, ncols: int = 1,
            **kwargs) -> tuple:
    """
    Create a new figure with IEEE sizing.

    Parameters
    ----------
    width  : figure width in inches (default: double-column = 7.16 in)
    height : figure height in inches (auto if None: golden ratio × width / ncols)
    nrows, ncols : subplot grid
    **kwargs : forwarded to plt.subplots
    """
    if height is None:
        height = width / ncols * nrows / 1.618
        height = min(height, PAGE_H)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width, height),
                             **kwargs)
    return fig, axes


def despine(ax: mpl.axes.Axes, *, top: bool = True, right: bool = True) -> None:
    """Remove spines for a cleaner academic look."""
    if top:
        ax.spines["top"].set_visible(False)
    if right:
        ax.spines["right"].set_visible(False)
