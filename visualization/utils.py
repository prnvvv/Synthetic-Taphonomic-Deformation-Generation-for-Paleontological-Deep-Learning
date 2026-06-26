"""
Export utilities shared across all figure generators.

Handles:
  - Multi-format figure export (PNG, PDF, SVG) at publication DPI
  - Table export (CSV, XLSX, Markdown, LaTeX)
  - LaTeX figure/table snippet generation
  - Output directory management
"""
from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from style import EXPORT_DPI

# ---------------------------------------------------------------------------
# Directory roots (resolved relative to this file → always correct)
# ---------------------------------------------------------------------------
VIZ_DIR     = Path(__file__).resolve().parent
GEN_DIR     = VIZ_DIR / "generated"
TABLE_DIR   = VIZ_DIR / "tables"
LATEX_DIR   = VIZ_DIR / "latex"

# Project root
PROJECT_DIR = VIZ_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"

# Ensure directories exist
for _d in (GEN_DIR, TABLE_DIR, LATEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Figure export
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure,
                name: str,
                formats: tuple[str, ...] = ("png", "pdf", "svg"),
                dpi: int = EXPORT_DPI,
                transparent: bool = False) -> dict[str, Path]:
    """
    Save *fig* to GEN_DIR/<name>.<fmt> for every format in *formats*.

    Returns a dict mapping format → saved path.
    """
    paths: dict[str, Path] = {}
    for fmt in formats:
        out = GEN_DIR / f"{name}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight",
                    pad_inches=0.05, transparent=transparent,
                    format=fmt)
        paths[fmt] = out
    print(f"  Saved: {name}  [{', '.join(formats)}]")
    return paths


# ---------------------------------------------------------------------------
# Table export
# ---------------------------------------------------------------------------

def save_table(df: pd.DataFrame,
               name: str,
               caption: str = "",
               label: str = "") -> dict[str, Path]:
    """
    Export *df* to TABLE_DIR/<name>.{csv, xlsx, md, tex}.

    Returns a dict mapping format → saved path.
    """
    paths: dict[str, Path] = {}

    # CSV
    csv_path = TABLE_DIR / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    # Excel
    xlsx_path = TABLE_DIR / f"{name}.xlsx"
    df.to_excel(xlsx_path, index=False)
    paths["xlsx"] = xlsx_path

    # Markdown
    md_path = TABLE_DIR / f"{name}.md"
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    paths["md"] = md_path

    # LaTeX
    tex_path = TABLE_DIR / f"{name}.tex"
    label_str = label or f"tab:{name}"
    caption_str = caption or name.replace("_", " ").title()

    n_cols = len(df.columns)
    col_fmt = "l" + "r" * (n_cols - 1)
    latex_body = df.to_latex(index=False, escape=True,
                              column_format=col_fmt,
                              caption=caption_str, label=label_str,
                              position="htbp")
    tex_path.write_text(latex_body, encoding="utf-8")
    paths["tex"] = tex_path

    print(f"  Table: {name}  [csv, xlsx, md, tex]")
    return paths


# ---------------------------------------------------------------------------
# LaTeX snippet builders
# ---------------------------------------------------------------------------

def latex_figure_snippet(name: str,
                          caption: str,
                          label: str,
                          width: str = r"\columnwidth") -> str:
    """Return a complete IEEE-style LaTeX figure environment string."""
    return textwrap.dedent(fr"""
    \begin{{figure}}[htbp]
      \centering
      \includegraphics[width={width}]{{figures/{name}.pdf}}
      \caption{{{caption}}}
      \label{{{label}}}
    \end{{figure}}
    """).strip()


def latex_table_snippet(name: str,
                        caption: str,
                        label: str,
                        df: pd.DataFrame | None = None,
                        col_fmt: str | None = None) -> str:
    """Return a complete IEEE-style LaTeX table environment string."""
    if df is not None:
        n = len(df.columns)
        col_fmt = col_fmt or ("l" + "r" * (n - 1))
        body = df.to_latex(index=False, escape=True, column_format=col_fmt)
        # Remove the auto wrapping \begin{table} added by pandas ≥2.0
        body = body.replace("\\begin{table}\n", "").replace("\\end{table}\n", "")
    else:
        body = "% (table body generated externally)\n"

    return textwrap.dedent(fr"""
    \begin{{table}}[htbp]
      \centering
      \caption{{{caption}}}
      \label{{{label}}}
    {textwrap.indent(body.strip(), '  ')}
    \end{{table}}
    """).strip()


# ---------------------------------------------------------------------------
# Registry — collect all snippet pairs during a run then flush to latex/
# ---------------------------------------------------------------------------

_figure_snippets: list[str] = []
_table_snippets:  list[str] = []
_captions:        dict[str, str] = {}
_labels:          dict[str, str] = {}


def register_figure(name: str, caption: str, label: str,
                    width: str = r"\columnwidth") -> None:
    _figure_snippets.append(latex_figure_snippet(name, caption, label, width))
    _captions[name] = caption
    _labels[name]   = label


def register_table(name: str, caption: str, label: str,
                   df: pd.DataFrame | None = None) -> None:
    _table_snippets.append(latex_table_snippet(name, caption, label, df))
    _captions[name] = caption
    _labels[name]   = label


def flush_latex() -> None:
    """Write accumulated LaTeX snippets to latex/*.tex files."""
    (LATEX_DIR / "figures.tex").write_text(
        "% IEEE Access — figure includes\n% Generated automatically\n\n"
        + "\n\n".join(_figure_snippets) + "\n",
        encoding="utf-8",
    )
    (LATEX_DIR / "tables.tex").write_text(
        "% IEEE Access — table environments\n% Generated automatically\n\n"
        + "\n\n".join(_table_snippets) + "\n",
        encoding="utf-8",
    )
    cap_lines = "\n".join(f"% {k}: {v}" for k, v in _captions.items())
    (LATEX_DIR / "captions.tex").write_text(
        "% IEEE Access — figure/table captions\n\n" + cap_lines + "\n",
        encoding="utf-8",
    )
    lbl_lines = "\n".join(fr"\newcommand{{\ref{k}}}{{\ref{{{v}}}}}"
                          for k, v in _labels.items())
    (LATEX_DIR / "labels.tex").write_text(
        "% IEEE Access — cross-reference labels\n\n" + lbl_lines + "\n",
        encoding="utf-8",
    )
    print(f"\n  LaTeX snippets → {LATEX_DIR}")
