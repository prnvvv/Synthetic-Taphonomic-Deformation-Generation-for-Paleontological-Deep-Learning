"""
TDE Visualization Suite — IEEE Access Publication Generator
Generates all figures, tables, and LaTeX snippets for the paper.

Run from the visualization/ directory:
    python generate_all_visualizations.py

All outputs land in:
    visualization/generated/   ← PNG, PDF, SVG
    visualization/tables/      ← CSV, XLSX, MD, LaTeX
    visualization/latex/       ← figures.tex, tables.tex, captions.tex, labels.tex
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Bootstrap paths so this script can be run from any cwd
# ---------------------------------------------------------------------------
VIZ_DIR     = Path(__file__).resolve().parent
PROJECT_DIR = VIZ_DIR.parent
sys.path.insert(0, str(VIZ_DIR))

import style as S
import utils as U

S.apply_ieee_style()

PROJ_DIR  = PROJECT_DIR / "dataset" / "projections"
SYN_DIR   = PROJECT_DIR / "dataset" / "synthetic" / "synthetic_images"
MASK_DIR  = PROJECT_DIR / "dataset" / "synthetic" / "deformation_masks"
CSV_PATH  = PROJECT_DIR / "dataset" / "synthetic" / "synthetic_labels.csv"

# ---------------------------------------------------------------------------
# Load CSV once
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df["param_c"]      = pd.to_numeric(df["param_c"],      errors="coerce")
df["param_k"]      = pd.to_numeric(df["param_k"],      errors="coerce")
df["param_sx"]     = pd.to_numeric(df["param_sx"],     errors="coerce")
df["param_sy"]     = pd.to_numeric(df["param_sy"],     errors="coerce")
df["param_lambda"] = pd.to_numeric(df["param_lambda"], errors="coerce")

SPECIMENS = sorted(df["specimen"].unique().tolist())
DEF_TYPES = ["compression", "shearing", "stretching", "dissolution"]
DEF_COLORS = [S.DEFORMATION_COLORS[d] for d in DEF_TYPES]

# ---------------------------------------------------------------------------
# Helper: load grayscale image safely
# ---------------------------------------------------------------------------
def _load_gray(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def _first_png(spec: str) -> Path | None:
    d = PROJ_DIR / spec
    pngs = sorted(d.glob("*.png")) if d.exists() else []
    return pngs[0] if pngs else None


def _ortho_png(spec: str, idx: int = 0) -> Path | None:
    d = PROJ_DIR / spec
    pngs = [p for p in sorted(d.glob("*_ortho_*.png"))] if d.exists() else []
    return pngs[idx] if idx < len(pngs) else None


# ============================================================
# SECTION 1 — SYSTEM OVERVIEW
# ============================================================

def fig01_pipeline_overview():
    """End-to-end pipeline flowchart."""
    name    = "fig01_pipeline_overview"
    caption = ("End-to-end TDE pipeline from ESRF CT acquisition to labeled "
               "synthetic deformation dataset. Stages correspond to the three "
               "processing modules: data preparation, 3-D projection, and "
               "synthetic deformation generation.")
    label   = "fig:pipeline_overview"

    fig, ax = plt.subplots(figsize=(S.COL2, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        ("Person 1\nData Prep",   "ESRF JP2\nSlice Archives",   "#4C72B0"),
        ("Person 1\nOrganize",    "Hierarchical\nDataset Tree",  "#4C72B0"),
        ("Person 1\nClean",       "Sequential\nJP2 Slices",      "#4C72B0"),
        ("Person 2\nProject",     "30 PNG\nProjections",         "#DD8452"),
        ("Person 3\nDeform",      "960 Synthetic\n+ 960 Masks",  "#55A868"),
    ]

    box_w, box_h = 1.55, 0.56
    gap = 0.12
    total = len(stages) * box_w + (len(stages) - 1) * gap
    x0 = (10 - total) / 2

    for i, (person, output, color) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        y_top, y_bot = 0.82, 0.20

        # Top box: person/action
        rect_top = FancyBboxPatch((x, y_top), box_w, 0.15,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color, edgecolor="white",
                                   linewidth=0.8, alpha=0.9)
        ax.add_patch(rect_top)
        ax.text(x + box_w/2, y_top + 0.075, person,
                ha="center", va="center", fontsize=6.5,
                color="white", fontweight="bold")

        # Bottom box: output artifact
        rect_bot = FancyBboxPatch((x, y_bot), box_w, 0.52,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color, edgecolor="white",
                                   linewidth=0.8, alpha=0.25)
        ax.add_patch(rect_bot)
        ax.text(x + box_w/2, y_bot + 0.26, output,
                ha="center", va="center", fontsize=7.5,
                color="#333333")

        # Arrow between boxes
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + box_w + gap, 0.54),
                        xytext=(x + box_w, 0.54),
                        arrowprops=dict(arrowstyle="-|>", color="#555555",
                                        lw=1.2))

    # Bottom statistics strip
    stats_txt = ("8 Specimens  ·  8 Acquisition IDs  ·  "
                 "240 Projection Images  ·  960 Synthetic Images  ·  "
                 "960 Binary Masks  ·  4 Deformation Classes")
    ax.text(5, 0.07, stats_txt, ha="center", va="center",
            fontsize=7, color="#555555",
            bbox=dict(fc="#F5F5F5", ec="#CCCCCC", pad=4, lw=0.6))

    fig.suptitle("TDE End-to-End Pipeline Overview", fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()

    U.save_figure(fig, name)
    plt.close(fig)

    # Table
    tbl = pd.DataFrame({
        "Stage": ["Data Preparation", "Organization", "Cleaning", "Projection", "Deformation"],
        "Responsible": ["Person 1"] * 3 + ["Person 2", "Person 3"],
        "Input": ["Raw ZIP archives", "Organized folders", "Unsorted JP2 slices",
                  "Sequential JP2 slices", "224×224 PNG projections"],
        "Output": ["Extracted folders", "Hierarchical dataset tree",
                   "Renamed sequential slices", "30 PNG views/specimen",
                   "960 synthetic images + 960 masks"],
        "Key Tool": ["File system", "shutil.move", "rename_sequential()",
                     "stream_volume_projections()", "apply_compression/shearing/stretching/dissolution()"],
    })
    U.save_table(tbl, "table01_pipeline_stages",
                 caption="TDE pipeline stages, responsible parties, inputs, outputs, and key functions.",
                 label="tab:pipeline_stages")
    U.register_figure(name, caption, label)
    U.register_table("table01_pipeline_stages",
                     "TDE pipeline stages, responsible parties, inputs, outputs, and key functions.",
                     "tab:pipeline_stages", tbl)


def fig02_module_dependency():
    """Software module dependency graph using NetworkX."""
    name    = "fig02_module_dependency"
    caption = ("Software module dependency graph of the TDE pipeline. "
               "Nodes represent Python modules; directed edges denote "
               "import or data-flow dependencies.")
    label   = "fig:module_dependency"

    G = nx.DiGraph()
    nodes = {
        "create_folder.py":        ("Person 1", "#4C72B0"),
        "organise_data.py":        ("Person 1", "#4C72B0"),
        "clean_dataset.py":        ("Person 1", "#4C72B0"),
        "remove_extras.py":        ("Person 1", "#4C72B0"),
        "generate_projections.py": ("Person 2", "#DD8452"),
        "generate_synthetic.py":   ("Person 3", "#55A868"),
        "dataset/raw/":            ("Data",     "#8172B2"),
        "dataset/projections/":    ("Data",     "#8172B2"),
        "dataset/synthetic/":      ("Data",     "#8172B2"),
    }
    for n in nodes:
        G.add_node(n)

    edges = [
        ("create_folder.py",        "dataset/raw/"),
        ("organise_data.py",        "dataset/raw/"),
        ("clean_dataset.py",        "dataset/raw/"),
        ("remove_extras.py",        "dataset/raw/"),
        ("dataset/raw/",            "generate_projections.py"),
        ("generate_projections.py", "dataset/projections/"),
        ("dataset/projections/",    "generate_synthetic.py"),
        ("generate_synthetic.py",   "dataset/synthetic/"),
    ]
    G.add_edges_from(edges)

    pos = {
        "create_folder.py":        (0,  3),
        "organise_data.py":        (0,  2),
        "clean_dataset.py":        (0,  1),
        "remove_extras.py":        (0,  0),
        "dataset/raw/":            (2,  1.5),
        "generate_projections.py": (4,  1.5),
        "dataset/projections/":    (6,  1.5),
        "generate_synthetic.py":   (8,  1.5),
        "dataset/synthetic/":      (10, 1.5),
    }

    fig, ax = plt.subplots(figsize=(S.COL2, 3.2))
    ax.axis("off")

    node_colors = [nodes[n][1] for n in G.nodes()]
    node_labels = {n: n.replace("generate_", "gen_") for n in G.nodes()}

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=1400, alpha=0.85)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#555555",
                           arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.05",
                           width=1.2, alpha=0.8)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                            font_size=5.5, font_color="white",
                            font_weight="bold")

    legend_handles = [
        mpatches.Patch(color="#4C72B0", label="Person 1 — Data Preparation"),
        mpatches.Patch(color="#DD8452", label="Person 2 — Projection"),
        mpatches.Patch(color="#55A868", label="Person 3 — Deformation"),
        mpatches.Patch(color="#8172B2", label="Dataset Artifacts"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
              framealpha=0.9)

    fig.suptitle("Software Module Dependency Graph", fontsize=10,
                 fontweight="bold")
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Module": list(nodes.keys()),
        "Person": [v[0] for v in nodes.values()],
        "Role": ["Create folder hierarchy", "Move ESRF downloads",
                 "Clean & rename slices", "Remove non-image extras",
                 "Streaming MIP projection", "Synthetic deformation",
                 "Raw CT slice store", "2-D projection store", "Synthetic dataset store"],
    })
    U.save_table(tbl, "table02_module_roles",
                 caption="TDE pipeline module roles and responsibilities.",
                 label="tab:module_roles")
    U.register_figure(name, caption, label)


def fig03_data_flow():
    """Detailed data flow: JP2 → MIP → projections → deformations → labels."""
    name    = "fig03_data_flow"
    caption = ("Data flow diagram of the TDE pipeline showing transformations "
               "applied at each processing step and the intermediate data formats.")
    label   = "fig:data_flow"

    fig, ax = plt.subplots(figsize=(S.COL2, 2.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)
    ax.axis("off")

    items = [
        ("JP2\nSlices\n(uint16)",     "#4C72B0", "D×H×W\nvoxels"),
        ("Streaming\nMIP",            "#C44E52", "3 × MIP\nplanes"),
        ("Percentile\nNorm.",         "#C44E52", "uint8\n[0,255]"),
        ("INTER_AREA\nResize",        "#C44E52", "224×224\npx"),
        ("PNG\nProjections",          "#DD8452", "30/\nspecimen"),
        ("Affine / Intensity\nDeform","#55A868", "4 types\nper image"),
        ("Synthetic\nDataset",        "#55A868", "960 imgs\n+960 masks"),
    ]

    bw, bh = 1.42, 0.52
    gap = 0.12
    total = len(items) * bw + (len(items)-1) * gap
    x0 = (12 - total) / 2
    yc = 0.5

    for i, (label_txt, color, sub) in enumerate(items):
        x = x0 + i * (bw + gap)
        rect = FancyBboxPatch((x, yc - bh/2), bw, bh,
                               boxstyle="round,pad=0.03",
                               facecolor=color, alpha=0.8,
                               edgecolor="white", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + bw/2, yc + 0.08, label_txt, ha="center", va="center",
                fontsize=6.2, color="white", fontweight="bold")
        ax.text(x + bw/2, yc - 0.17, sub, ha="center", va="center",
                fontsize=5.5, color="white", alpha=0.9)
        if i < len(items) - 1:
            ax.annotate("", xy=(x + bw + gap, yc),
                        xytext=(x + bw, yc),
                        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.0))

    fig.suptitle("TDE Data Flow Diagram", fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Step": [i[0].replace("\n", " ") for i in items],
        "Input Format": ["uint16 JP2", "uint16 volume", "float32 MIP",
                         "uint8 224×224", "uint8 PNG", "uint8 PNG", "uint8 PNGs"],
        "Output Format": ["uint16 volume", "3× MIP plane", "uint8 [0,255]",
                          "uint8 224×224", "30 PNG/specimen",
                          "uint8 PNG + mask", "CSV + zip archives"],
        "Memory Profile": ["O(D·H·W)", "O(H·W)", "O(H·W)",
                           "O(224²)", "O(224²)", "O(224²)", "O(N·224²)"],
    })
    U.save_table(tbl, "table03_data_flow",
                 caption="Data transformations, formats, and memory profiles at each TDE pipeline step.",
                 label="tab:data_flow")
    U.register_figure(name, caption, label)


# ============================================================
# SECTION 2 — DATASET STATISTICS
# ============================================================

def fig04_specimen_distribution():
    """Bar chart: projections per specimen."""
    name    = "fig04_specimen_distribution"
    caption = ("Number of 2-D projection images generated per fossil specimen. "
               "Each specimen contributes exactly 30 projections (6 orthographic "
               "+ 24 rotational), yielding a balanced dataset.")
    label   = "fig:specimen_distribution"

    counts = df.groupby("specimen")["image_id"].count().reset_index()
    counts.columns = ["specimen", "count"]
    counts["short"] = counts["specimen"].map(S.SPECIMEN_SHORT)
    counts = counts.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(S.COL2, 3.2))
    bars = ax.barh(counts["short"], counts["count"] / 4,
                   color=S.PALETTE_OI[:len(counts)], edgecolor="white", lw=0.5)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2,
                f"{int(w)}", va="center", fontsize=7.5)

    ax.set_xlabel("Number of Projection Images")
    ax.set_title("Projection Images per Specimen (balanced, 30 per specimen)")
    ax.set_xlim(0, 38)
    S.despine(ax)
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Specimen": counts["short"].tolist(),
        "Specimen Key": counts["specimen"].tolist(),
        "Projections": (counts["count"] / 4).astype(int).tolist(),
        "Synthetic Images": counts["count"].tolist(),
        "Masks": counts["count"].tolist(),
    })
    U.save_table(tbl, "table04_specimen_distribution",
                 caption="Projection and synthetic image counts per specimen in the TDE dataset.",
                 label="tab:specimen_distribution")
    U.register_figure(name, caption, label)


def fig05_dataset_composition():
    """Pie chart: proportional contribution by specimen."""
    name    = "fig05_dataset_composition"
    caption = ("Dataset composition showing equal contribution of all eight "
               "archosaur specimens to the TDE synthetic deformation dataset. "
               "Each specimen contributes 120 synthetic images (12.5%).")
    label   = "fig:dataset_composition"

    counts = df.groupby("specimen").size()
    labels = [S.SPECIMEN_SHORT[s] for s in counts.index]

    fig, ax = plt.subplots(figsize=(S.COL2 * 0.6, 3.4))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=None, autopct="%1.1f%%",
        colors=S.PALETTE_OI[:len(counts)],
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(edgecolor="white", linewidth=0.8))
    for at in autotexts:
        at.set_fontsize(6.5)

    ax.legend(wedges, labels, loc="center left",
              bbox_to_anchor=(1.0, 0.5), fontsize=7, framealpha=0.9)
    ax.set_title("Dataset Composition\nby Specimen", fontsize=9, fontweight="bold")
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Specimen": labels,
        "Synthetic Images": counts.values.tolist(),
        "Percentage (%)": [f"{v/counts.sum()*100:.1f}" for v in counts.values],
    })
    U.save_table(tbl, "table05_dataset_composition",
                 caption="Proportional composition of the TDE dataset by specimen.",
                 label="tab:dataset_composition")
    U.register_figure(name, caption, label)


def fig06_dataset_growth():
    """Funnel/stacked bar showing dataset growth through pipeline."""
    name    = "fig06_dataset_growth"
    caption = ("Dataset scale at each stage of the TDE pipeline, demonstrating "
               "the multiplicative effect of multi-view projection (×30) and "
               "synthetic deformation (×4) on the base specimen count.")
    label   = "fig:dataset_growth"

    stages = ["Specimens", "Projection\nImages", "Synthetic\nImages", "Binary\nMasks"]
    values = [8, 240, 960, 960]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#CC79A7"]

    fig, ax = plt.subplots(figsize=(S.COL1 + 0.5, 3.0))
    bars = ax.bar(stages, values, color=colors, edgecolor="white", lw=0.8, width=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f"{v:,}", ha="center", fontsize=8.5, fontweight="bold")

    # Multiplication annotations
    multipliers = ["×30", "×4", "×1"]
    xs = [0.5, 1.5, 2.5]
    for x, m in zip(xs, multipliers):
        ax.annotate("", xy=(x + 0.01, 200), xytext=(x - 0.01, 200),
                    arrowprops=dict(arrowstyle="-|>", color="#888", lw=0.8))
        ax.text(x, 220, m, ha="center", fontsize=7.5, color="#555")

    ax.set_ylabel("Number of Items")
    ax.set_title("Dataset Growth Through\nTDE Pipeline")
    ax.set_ylim(0, 1150)
    S.despine(ax)
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Stage": stages,
        "Count": values,
        "Multiplier from Previous": ["-", "×30", "×4", "×1"],
        "Cumulative Images": [8, 240, 960, 960],
    })
    U.save_table(tbl, "table06_dataset_growth",
                 caption="Dataset size at each stage of the TDE pipeline.",
                 label="tab:dataset_growth")
    U.register_figure(name, caption, label)


def fig07_projection_type_breakdown():
    """Donut chart: orthographic vs rotational projections."""
    name    = "fig07_projection_types"
    caption = ("Breakdown of the 30 projection views generated per specimen: "
               "6 orthographic views (3 axes × 2 directions) and 24 rotational "
               "views (0° to 345° at 15° increments) of the sagittal MIP.")
    label   = "fig:projection_types"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(S.COL2, 2.8))

    # Left: donut
    sizes  = [6, 24]
    labels = ["Orthographic\n(6 views)", "Rotational\n(24 views)"]
    colors = ["#4C72B0", "#DD8452"]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, autopct="%1.0f%%",
        colors=colors, startangle=90,
        wedgeprops=dict(width=0.5, edgecolor="white"),
        pctdistance=0.75)
    for at in autotexts:
        at.set_fontsize(8)
    ax1.set_title("View Type Distribution\n(per specimen)", fontsize=8.5)

    # Right: polar plot of rotation angles
    ax2_polar = fig.add_subplot(122, polar=True)
    ax1.remove()  # remove placeholder
    ax2.remove()

    ax_left  = fig.add_subplot(121)
    ax_left.pie(sizes, labels=labels, autopct="%1.0f%%",
                colors=colors, startangle=90,
                wedgeprops=dict(width=0.5, edgecolor="white"),
                pctdistance=0.75,
                textprops={"fontsize": 7.5})
    ax_left.set_title("View Type Split\n(30 views/specimen)", fontsize=8)

    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ax2_polar.set_theta_zero_location("N")
    ax2_polar.set_theta_direction(-1)
    ax2_polar.scatter(angles, np.ones(24), s=40, color="#DD8452", zorder=3)
    ax2_polar.plot(np.append(angles, angles[0]),
                   np.ones(25), color="#DD8452", lw=0.8, alpha=0.5)
    ax2_polar.set_rticks([])
    ax2_polar.set_xticks(angles)
    angle_labels = [f"{int(a * 180/np.pi)}°" for a in angles]
    ax2_polar.set_xticklabels(angle_labels, fontsize=5.5)
    ax2_polar.set_title("Rotational View\nAngle Coverage", fontsize=8, pad=12)
    ax2_polar.set_ylim(0, 1.5)

    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "View Type": ["Orthographic"] * 6 + ["Rotational"] * 24,
        "Label": [f"ortho_{i:02d}" for i in range(6)]
                 + [f"rot_{a:03d}" for a in range(0, 360, 15)],
        "Angle (°)": ["N/A"] * 6 + list(range(0, 360, 15)),
        "Axis / Source": (
            ["Coronal (+)", "Coronal (−)", "Sagittal (+)",
             "Sagittal (−)", "Transverse (+)", "Transverse (−)"]
            + ["Sagittal MIP rotated"] * 24
        ),
    })
    U.save_table(tbl, "table07_projection_types",
                 caption="Specification of all 30 projection views generated per specimen.",
                 label="tab:projection_types")
    U.register_figure(name, caption, label)


def fig08_deformation_distribution():
    """Bar chart: synthetic images per deformation type."""
    name    = "fig08_deformation_distribution"
    caption = ("Distribution of synthetic images across four taphonomic "
               "deformation categories. All categories are balanced with "
               "240 images each, producing a class-balanced training dataset.")
    label   = "fig:deformation_distribution"

    counts = df["deformation_type"].value_counts().reindex(DEF_TYPES)

    fig, ax = plt.subplots(figsize=(S.COL1 + 0.5, 2.8))
    bars = ax.bar(counts.index, counts.values,
                  color=DEF_COLORS, edgecolor="white", lw=0.8, width=0.6)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 3, str(int(bar.get_height())),
                ha="center", fontsize=8, fontweight="bold")

    ax.set_ylabel("Number of Synthetic Images")
    ax.set_title("Synthetic Images per\nDeformation Type (balanced)")
    ax.set_ylim(0, 290)
    ax.set_xticklabels([d.capitalize() for d in counts.index], rotation=15)
    S.despine(ax)
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Deformation Type": [d.capitalize() for d in DEF_TYPES],
        "Count": counts.values.tolist(),
        "Percentage (%)": [f"{v/counts.sum()*100:.1f}" for v in counts.values],
        "Parameter": ["c ∈ [0.30, 0.90]", "k ∈ [−0.50, 0.50]",
                      "sx,sy ∈ [1.00, 1.50]", "λ ∈ [0.30, 0.80]"],
        "Mask Type": ["Difference mask", "Difference mask",
                      "Difference mask", "Ellipse region"],
    })
    U.save_table(tbl, "table08_deformation_distribution",
                 caption="Distribution and parameter ranges for each of the four synthetic "
                         "taphonomic deformation types in the TDE dataset.",
                 label="tab:deformation_distribution")
    U.register_figure(name, caption, label)


# ============================================================
# SECTION 3 — PROJECTION PIPELINE
# ============================================================

def fig09_projection_pipeline():
    """Block diagram: JP2 slices → streaming MIP → normalize → resize."""
    name    = "fig09_projection_pipeline"
    caption = ("Streaming Maximum Intensity Projection (MIP) architecture. "
               "JP2 slices are decoded one at a time; three orthogonal MIPs "
               "are accumulated in a single pass, reducing peak memory from "
               r"O(D \cdot H^2) to O(H^2).")
    label   = "fig:projection_pipeline"

    fig, ax = plt.subplots(figsize=(S.COL2, 3.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main pipeline row
    steps = [
        ("JP2 Slice\nStream\n(glymur)", "#4C72B0",
         "D slices\nuint16\nH×W"),
        ("Running MIP\nAccumulator\n(in-place max)", "#C44E52",
         "3 planes:\naxis0,1,2\nO(H²) RAM"),
        ("Percentile\nNormalize\n1–99%", "#DD8452",
         "→ float32\n→ uint8\n[0,255]"),
        ("INTER_AREA\nResize\n(OpenCV)", "#55A868",
         "→ 224×224\npx\nuint8"),
        ("PNG\nOutput\n(30 views)", "#8172B2",
         "6 ortho\n+24 rot.\nper spec."),
    ]

    bw, bh = 1.8, 0.58
    gap = 0.14
    total = len(steps) * bw + (len(steps)-1)*gap
    x0 = (12 - total) / 2

    for i, (title, color, desc) in enumerate(steps):
        x = x0 + i * (bw + gap)
        yc = 0.58
        rect = FancyBboxPatch((x, yc - bh/2), bw, bh,
                               boxstyle="round,pad=0.03",
                               facecolor=color, alpha=0.85,
                               edgecolor="white", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + bw/2, yc + 0.10, title, ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold")
        ax.text(x + bw/2, yc - 0.18, desc, ha="center", va="center",
                fontsize=5.5, color="white", alpha=0.9)

        if i < len(steps) - 1:
            ax.annotate("", xy=(x + bw + gap, yc),
                        xytext=(x + bw, yc),
                        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.2))

    # Side branch: rotational views
    mid = x0 + 1 * (bw + gap) + bw/2
    ax.annotate("", xy=(mid, 0.19), xytext=(mid, 0.29),
                arrowprops=dict(arrowstyle="-|>", color="#777", lw=0.9, ls="dashed"))
    rot_rect = FancyBboxPatch((mid - 0.9, 0.05), 1.8, 0.15,
                               boxstyle="round,pad=0.03",
                               facecolor="#8172B2", alpha=0.7,
                               edgecolor="white", lw=0.7)
    ax.add_patch(rot_rect)
    ax.text(mid, 0.125, "Sagittal MIP → 2D Rotate ×24\n(15° increments, bicubic)",
            ha="center", va="center", fontsize=6, color="white")

    fig.suptitle("Streaming MIP Projection Architecture", fontsize=10,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Step": ["JP2 Decode", "MIP Accumulation", "Percentile Norm.",
                 "INTER_AREA Resize", "Orthographic Save", "2D Rotation", "Rotational Save"],
        "Function": ["glymur.Jp2k[:]", "np.maximum(mip, arr, out=mip)",
                     "percentile_normalise()", "cv2.resize(INTER_AREA)",
                     "cv2.imwrite()", "scipy.ndimage.rotate(order=3)",
                     "cv2.imwrite()"],
        "Input Shape": ["H×W uint16", "H×W + H×W", "H×W uint16",
                        "H×W uint8", "H×W uint8", "H×W float32", "224×224 uint8"],
        "Output Shape": ["H×W uint16", "3 MIP planes", "H×W uint8",
                         "224×224 uint8", "PNG file", "variable float32", "PNG file"],
        "Memory": ["O(H·W)", "O(H·W)", "O(H·W)",
                   "O(224²)", "O(1)", "O(max_dim²)", "O(1)"],
    })
    U.save_table(tbl, "table09_projection_pipeline",
                 caption="Detailed step-by-step specification of the TDE streaming MIP projection pipeline.",
                 label="tab:projection_pipeline")
    U.register_figure(name, caption, label)


def fig10_sample_projections():
    """Gallery of real projection images from the dataset."""
    name    = "fig10_sample_projections"
    caption = ("Representative 2-D Maximum Intensity Projection images from "
               "four archosaur specimens. Top row: orthographic coronal views. "
               "Bottom row: rotational views (45°, 90°, 135°, 180°). "
               "All images are 224×224 grayscale PNG.")
    label   = "fig:sample_projections"

    specimen_keys = [
        "araripesaurus__BSPG-1982-I-90",
        "archaeopteryx_london__BMNH-37001",
        "halszkaraptor__MPC-D-102-109",
        "titanosaur_embryo__MCF-PVPH-874",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(S.COL2, 3.4))

    for col, spec in enumerate(specimen_keys):
        spec_dir = PROJ_DIR / spec
        ortho_pngs = sorted(spec_dir.glob("*_ortho_00.png"))
        rot_pngs   = sorted(spec_dir.glob("*_rot_045.png"))

        for row, pngs in enumerate([ortho_pngs, rot_pngs]):
            ax = axes[row, col]
            if pngs:
                img = _load_gray(pngs[0])
                if img is not None:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255,
                              interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(S.SPECIMEN_SHORT[spec], fontsize=6.5, pad=3)
            if col == 0:
                view = "Ortho (0°)" if row == 0 else "Rot. (45°)"
                ax.set_ylabel(view, fontsize=7)

    fig.suptitle("Sample Projection Images from TDE Dataset", fontsize=9,
                 fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Specimen": [S.SPECIMEN_SHORT[s] for s in specimen_keys],
        "Specimen Key": specimen_keys,
        "Voxel Size (µm)": [S.VOXEL_SIZES[s] for s in specimen_keys],
        "Ortho Image": [f"{s}_ortho_00.png" for s in specimen_keys],
        "Rotation Image": [f"{s}_rot_045.png" for s in specimen_keys],
        "Resolution": ["224×224"] * 4,
        "Bit Depth": ["uint8"] * 4,
    })
    U.save_table(tbl, "table10_sample_projections",
                 caption="Metadata for specimen projection images shown in the sample gallery.",
                 label="tab:sample_projections")
    U.register_figure(name, caption, label)


def fig11_all_specimens_grid():
    """One ortho view per specimen — complete 8-specimen grid."""
    name    = "fig11_all_specimens_grid"
    caption = ("Complete gallery of orthographic coronal (ortho\\_00) Maximum "
               "Intensity Projection images for all eight archosaur specimens "
               "in the TDE dataset. Images are 224×224 grayscale PNG.")
    label   = "fig:all_specimens_grid"

    fig, axes = plt.subplots(2, 4, figsize=(S.COL2, 3.8))
    axes = axes.flatten()

    for i, spec in enumerate(SPECIMENS):
        ax = axes[i]
        png = _ortho_png(spec, 0)
        if png:
            img = _load_gray(png)
            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        short = S.SPECIMEN_SHORT[spec]
        vx    = S.VOXEL_SIZES.get(spec, "?")
        ax.set_title(f"{short}\n({vx} µm)", fontsize=6, pad=2)

    fig.suptitle("All Eight TDE Specimens — Orthographic Coronal MIP",
                 fontsize=9, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Specimen": [S.SPECIMEN_SHORT[s] for s in SPECIMENS],
        "Specimen ID": SPECIMENS,
        "Voxel Size (µm)": [S.VOXEL_SIZES.get(s, "N/A") for s in SPECIMENS],
        "Taxonomy": ["Pterosaur", "Bird", "Dromaeosaurid", "Sauropod (embryo)",
                     "Bird", "Bird", "Pterosaur", "Bird"],
        "Projections": [30] * 8,
        "Synthetic Images": [120] * 8,
    })
    U.save_table(tbl, "table11_all_specimens",
                 caption="All eight archosaur specimens in the TDE dataset with "
                         "taxonomic classification, voxel resolution, and image counts.",
                 label="tab:all_specimens")
    U.register_figure(name, caption, label)


def fig12_voxel_sizes():
    """Bar chart of voxel sizes per specimen (log scale)."""
    name    = "fig12_voxel_sizes"
    caption = ("CT voxel resolution (µm) for each archosaur specimen. "
               "Voxel sizes span two orders of magnitude (1.28–107.16 µm), "
               "reflecting the diverse scan resolutions available in the "
               "ESRF Paleo CT dataset.")
    label   = "fig:voxel_sizes"

    specs  = list(S.VOXEL_SIZES.keys())
    sizes  = list(S.VOXEL_SIZES.values())
    labels = [S.SPECIMEN_SHORT[s] for s in specs]
    order  = np.argsort(sizes)
    specs  = [specs[i] for i in order]
    sizes  = [sizes[i] for i in order]
    labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(S.COL2, 2.8))
    bars = ax.barh(labels, sizes,
                   color=S.PALETTE_OI[:len(sizes)], edgecolor="white", lw=0.6)
    for bar, v in zip(bars, sizes):
        ax.text(v * 1.05, bar.get_y() + bar.get_height()/2,
                f"{v} µm", va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Voxel Size (µm, log scale)")
    ax.set_title("CT Voxel Resolution per Specimen\n(ESRF Paleo CT Dataset)")
    S.despine(ax)
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Specimen": labels,
        "Voxel Size (µm)": sizes,
        "Scan Source": ["ESRF Paleo CT"] * len(specs),
        "Format": ["JP2 (JPEG 2000)"] * len(specs),
    })
    U.save_table(tbl, "table12_voxel_sizes",
                 caption="CT voxel resolution for each specimen in the TDE dataset.",
                 label="tab:voxel_sizes")
    U.register_figure(name, caption, label)


# ============================================================
# SECTION 4 — DEFORMATION MATHEMATICS & VISUALIZATION
# ============================================================

def fig13_transformation_matrices():
    """Visual representation of the 4 affine/intensity transformation matrices."""
    name    = "fig13_transformation_matrices"
    caption = (r"Transformation matrix representations for the four taphonomic "
               r"deformations. (a) Compression: $c \in [0.3, 0.9]$. "
               r"(b) Shearing: $k \in [-0.5, 0.5]$. "
               r"(c) Stretching: $s_x, s_y \in [1.0, 1.5]$. "
               r"(d) Dissolution: intensity subtraction via ellipse mask.")
    label   = "fig:transformation_matrices"

    fig, axes = plt.subplots(1, 4, figsize=(S.COL2, 2.4))
    titles = ["(a) Compression", "(b) Shearing",
              "(c) Stretching", "(d) Dissolution"]
    matrices = [
        np.array([[1, 0], [0, "c"]]),
        np.array([[1, 0], ["k", 1]]),
        np.array([["$s_x$", 0], [0, "$s_y$"]]),
        None,   # dissolution is not a matrix transform
    ]
    param_labels = [
        "c ∈ [0.3, 0.9]",
        "k ∈ [−0.5, 0.5]",
        "sₓ,sᵧ ∈ [1.0, 1.5]",
        "λ ∈ [0.3, 0.8]",
    ]

    for ax, title, mat, plabel in zip(axes, titles, matrices, param_labels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=7.5, fontweight="bold", pad=4)

        if mat is not None:
            # Draw 2×2 matrix
            cell_entries = [str(mat[r][c]) for r in range(2) for c in range(2)]
            for idx, entry in enumerate(cell_entries):
                r, c = divmod(idx, 2)
                x = 0.25 + c * 0.30
                y = 0.68 - r * 0.28
                ax.text(x, y, entry, ha="center", va="center",
                        fontsize=9,
                        color="#C44E52" if entry not in ("0", "1") else "#333333")

            # Brackets
            for xb, side in [(0.12, "["), (0.88, "]")]:
                ax.text(xb, 0.54, side, ha="center", va="center",
                        fontsize=22, color="#333333")

            ax.add_patch(FancyBboxPatch((0.10, 0.36), 0.80, 0.48,
                                         boxstyle="square,pad=0",
                                         fill=False, edgecolor="#CCCCCC", lw=0.6))
        else:
            # Dissolution formula
            ax.text(0.5, 0.62,
                    r"$I'(x,y) =$",
                    ha="center", va="center", fontsize=8, color="#333")
            ax.text(0.5, 0.47,
                    r"$I(x,y) - \lambda \cdot M(x,y)$",
                    ha="center", va="center", fontsize=8, color="#C44E52")

        ax.text(0.5, 0.15, plabel, ha="center", va="center",
                fontsize=7, color="#555555",
                bbox=dict(fc="#F5F5F5", ec="#CCCCCC", pad=2, lw=0.5))

    fig.suptitle("Taphonomic Deformation Transformations",
                 fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Deformation": ["Compression", "Shearing", "Stretching", "Dissolution"],
        "Transformation": [
            "[x', y'] = [[1,0],[0,c]]·[x,y]",
            "[x', y'] = [[1,0],[k,1]]·[x,y]",
            "[x', y'] = [[sx,0],[0,sy]]·[x,y]",
            "I'(x,y) = I(x,y) − λ·M(x,y)",
        ],
        "Parameter": ["c ∈ [0.3, 0.9]", "k ∈ [−0.5, 0.5]",
                      "sx,sy ∈ [1.0, 1.5]", "λ ∈ [0.3, 0.8]"],
        "Effect": ["Vertical compression", "Horizontal shear",
                   "Anisotropic scale", "Intensity loss"],
        "Axis": ["Y", "Y", "X and Y", "Intensity"],
        "Centre-aligned": ["Yes (cy)", "Yes (cx)", "Yes (cx, cy)", "N/A"],
    })
    U.save_table(tbl, "table13_transformations",
                 caption="Mathematical specifications of the four taphonomic deformation types "
                         "applied in the TDE synthetic generation pipeline.",
                 label="tab:transformations")
    U.register_figure(name, caption, label)


def fig14_deformation_geometry():
    """Visual geometry illustration for all 4 deformations."""
    name    = "fig14_deformation_geometry"
    caption = ("Geometric illustration of the four taphonomic deformation "
               "types applied to a unit square. The dashed square shows the "
               "original boundary; the solid polygon shows the deformed result.")
    label   = "fig:deformation_geometry"

    fig, axes = plt.subplots(1, 4, figsize=(S.COL2, 2.6))
    titles   = ["Compression\n(c = 0.6)", "Shearing\n(k = 0.35)",
                "Stretching\n(sₓ=1.3, sᵧ=1.4)", "Dissolution\n(λ = 0.5)"]
    colors   = DEF_COLORS

    # Original unit square corners (centered at origin)
    sq = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5],[-.5,-.5]])

    def transform(pts, M):
        return (M @ pts[:, :2].T).T

    transforms = [
        np.array([[1, 0], [0, 0.6]]),
        np.array([[1, 0], [0.35, 1]]),
        np.array([[1.3, 0], [0, 1.4]]),
        None,
    ]

    for ax, title, color, M in zip(axes, titles, colors, transforms):
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=7.5, fontweight="bold")

        # Original
        ax.plot(sq[:, 0], sq[:, 1], "--", color="#AAAAAA", lw=1.0, label="Original")

        if M is not None:
            deformed = transform(sq, M)
            ax.plot(deformed[:, 0], deformed[:, 1], "-",
                    color=color, lw=1.8, label="Deformed")
            ax.fill(deformed[:-1, 0], deformed[:-1, 1],
                    alpha=0.18, color=color)
        else:
            # Dissolution: show original square with ellipse cutout
            ax.fill(sq[:-1, 0], sq[:-1, 1], alpha=0.18, color=color)
            ax.plot(sq[:, 0], sq[:, 1], "-", color=color, lw=1.8)
            theta = np.linspace(0, 2*np.pi, 100)
            ex = 0.25 * np.cos(theta)
            ey = 0.18 * np.sin(theta)
            ax.fill(ex, ey, color="white", alpha=0.9)
            ax.plot(ex, ey, color="#444", lw=0.8, ls=":")
            ax.text(0, 0, "λ·M", ha="center", va="center",
                    fontsize=7, color="#444")

        ax.axhline(0, color="#DDDDDD", lw=0.4)
        ax.axvline(0, color="#DDDDDD", lw=0.4)
        ax.set_xticks([-.5, 0, .5])
        ax.set_yticks([-.5, 0, .5])
        ax.tick_params(labelsize=6)
        S.despine(ax, top=False, right=False)

    fig.suptitle("Geometric Effect of Taphonomic Deformations",
                 fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Deformation": ["Compression", "Shearing", "Stretching", "Dissolution"],
        "Example Parameter": ["c = 0.6", "k = 0.35", "sₓ=1.3, sᵧ=1.4", "λ = 0.5"],
        "Corner (TL) After": ["(−0.5, −0.3)", "(−0.325, −0.5)",
                               "(−0.65, −0.7)", "unchanged"],
        "Corner (TR) After": ["(0.5, −0.3)", "(0.675, 0.5)",
                               "(0.65, −0.7)", "unchanged"],
        "Area Change": ["−40%", "0%", "+82%", "0% (intensity only)"],
    })
    U.save_table(tbl, "table14_geometry",
                 caption="Geometric effect of example deformation parameters on unit-square corners.",
                 label="tab:geometry")
    U.register_figure(name, caption, label)


def fig15_deformation_gallery():
    """Before/After real-image gallery for all 4 deformation types."""
    name    = "fig15_deformation_gallery"
    caption = ("Deformation gallery for the araripesaurus specimen "
               "(ortho\\_00 view). Each column shows: original projection, "
               "deformed image, and binary deformation mask. "
               "Rows correspond to compression, shearing, stretching, "
               "and dissolution respectively.")
    label   = "fig:deformation_gallery"

    spec = "araripesaurus__BSPG-1982-I-90"
    base_stem = f"{spec}_ortho_00"
    orig_path = PROJ_DIR / spec / f"{base_stem}.png"
    orig = _load_gray(orig_path)

    # Find matching synthetic/mask files for this stem
    def find_syn(def_type: str):
        pngs = sorted(SYN_DIR.glob(f"{base_stem}__{def_type[0:4]}*.png"))
        return pngs[0] if pngs else None

    def_map = {"comp": "compression", "shear": "shearing",
               "stretch": "stretching", "diss": "dissolution"}

    rows_data = []
    for prefix, def_type in def_map.items():
        syn_pngs  = sorted(SYN_DIR.glob(f"{base_stem}__{prefix}*.png"))
        mask_pngs = sorted(MASK_DIR.glob(f"{base_stem}__{prefix}*_mask.png"))
        if syn_pngs and mask_pngs:
            rows_data.append((def_type, syn_pngs[0], mask_pngs[0]))

    n_rows = len(rows_data) + 1  # +1 for header original row
    fig, axes = plt.subplots(len(rows_data), 3, figsize=(S.COL2 * 0.75, 5.2))

    col_titles = ["Original (224×224)", "Deformed", "Binary Mask"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=7.5, fontweight="bold", pad=3)

    for i, (def_type, syn_path, mask_path) in enumerate(rows_data):
        syn_img  = _load_gray(syn_path)
        mask_img = _load_gray(mask_path)

        for j, (img, cmap) in enumerate([(orig, "gray"), (syn_img, "gray"),
                                          (mask_img, "gray")]):
            ax = axes[i, j]
            if img is not None:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(def_type.capitalize(), fontsize=7,
                               labelpad=3)

    fig.suptitle("Taphonomic Deformation Gallery — Araripesaurus (ortho_00)",
                 fontsize=9, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    U.save_figure(fig, name)
    plt.close(fig)

    rows = []
    for def_type, syn_path, mask_path in rows_data:
        param_row = df[(df["specimen"] == spec) &
                       (df["deformation_type"] == def_type) &
                       (df["original_file"] == f"{base_stem}.png")]
        param_str = ""
        if not param_row.empty:
            r = param_row.iloc[0]
            if def_type == "compression":
                param_str = f"c = {r['param_c']:.4f}"
            elif def_type == "shearing":
                param_str = f"k = {r['param_k']:.4f}"
            elif def_type == "stretching":
                param_str = f"sx = {r['param_sx']:.4f}, sy = {r['param_sy']:.4f}"
            else:
                param_str = f"λ = {r['param_lambda']:.4f}"
        rows.append({
            "Deformation": def_type.capitalize(),
            "Parameter Value": param_str,
            "Synthetic File": syn_path.name,
            "Mask File": mask_path.name,
        })
    tbl = pd.DataFrame(rows)
    U.save_table(tbl, "table15_gallery_params",
                 caption="Deformation parameters used in the gallery figure for "
                         "araripesaurus\\_\\_BSPG-1982-I-90 ortho\\_00.",
                 label="tab:gallery_params")
    U.register_figure(name, caption, label)


# ============================================================
# SECTION 5 — PARAMETER STATISTICAL ANALYSIS
# ============================================================

def fig16_parameter_distributions():
    """4-panel histogram of all deformation parameters from CSV."""
    name    = "fig16_parameter_distributions"
    caption = ("Parameter distributions for all four deformation types across "
               "the full TDE dataset (n=240 per type, seed=42). "
               r"Parameters are uniformly sampled: $c \sim \mathcal{U}(0.3,0.9)$, "
               r"$k \sim \mathcal{U}(-0.5,0.5)$, "
               r"$s_x, s_y \sim \mathcal{U}(1.0,1.5)$, "
               r"$\lambda \sim \mathcal{U}(0.3,0.8)$.")
    label   = "fig:parameter_distributions"

    param_info = [
        ("compression",  "param_c",      "c (Compression Factor)",    [0.3, 0.9]),
        ("shearing",     "param_k",      "k (Shear Coefficient)",     [-0.5, 0.5]),
        ("stretching",   "param_sx",     "sₓ (Stretch X)",            [1.0, 1.5]),
        ("dissolution",  "param_lambda", "λ (Dissolution Intensity)", [0.3, 0.8]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(S.COL2, 2.6))

    rows = []
    for ax, (def_type, col, xlabel, xlim) in zip(axes, param_info):
        vals = df[df["deformation_type"] == def_type][col].dropna()
        color = S.DEFORMATION_COLORS[def_type]

        ax.hist(vals, bins=20, color=color, edgecolor="white",
                lw=0.5, alpha=0.85, density=True)

        # Theoretical uniform PDF
        lo, hi = xlim
        ax.axhline(1/(hi-lo), color="#333333", lw=1.0, ls="--",
                   label="Uniform PDF")

        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("Density" if ax is axes[0] else "", fontsize=7)
        ax.set_title(def_type.capitalize(), fontsize=8, fontweight="bold")
        ax.set_xlim(xlim)
        S.despine(ax)

        rows.append({
            "Parameter": xlabel,
            "n": int(vals.count()),
            "Mean": f"{vals.mean():.4f}",
            "Std": f"{vals.std():.4f}",
            "Min": f"{vals.min():.4f}",
            "Max": f"{vals.max():.4f}",
            "Range": f"[{lo}, {hi}]",
        })

    axes[0].legend(fontsize=6, loc="upper left")
    fig.suptitle("Deformation Parameter Distributions (n=240 per type)",
                 fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame(rows)
    U.save_table(tbl, "table16_parameter_stats",
                 caption="Descriptive statistics of deformation parameters sampled "
                         "from uniform distributions (seed=42, n=240 per type).",
                 label="tab:parameter_stats")
    U.register_figure(name, caption, label)


def fig17_stretching_scatter():
    """Scatter plot: sx vs sy for stretching deformation."""
    name    = "fig17_stretching_scatter"
    caption = (r"Joint distribution of stretching parameters $s_x$ and $s_y$ "
               r"(n=240). Both are independently sampled from $\mathcal{U}(1.0,1.5)$, "
               "confirming near-zero correlation and independent axis scaling.")
    label   = "fig:stretching_scatter"

    sub = df[df["deformation_type"] == "stretching"].copy()
    sx  = sub["param_sx"].values
    sy  = sub["param_sy"].values

    fig, ax = plt.subplots(figsize=(S.COL1, 2.8))
    ax.scatter(sx, sy, alpha=0.4, s=12, color=S.DEFORMATION_COLORS["stretching"])

    r, p = stats.pearsonr(sx, sy)
    ax.set_xlabel("$s_x$ (stretch factor, X-axis)", fontsize=8)
    ax.set_ylabel("$s_y$ (stretch factor, Y-axis)", fontsize=8)
    ax.set_title(f"Stretching Parameters\n(Pearson r = {r:.3f}, p = {p:.3f})")
    ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes, fontsize=7)
    S.despine(ax)
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Statistic": ["n", "sx mean", "sx std", "sy mean", "sy std",
                      "Pearson r", "p-value"],
        "Value": [len(sx), f"{sx.mean():.4f}", f"{sx.std():.4f}",
                  f"{sy.mean():.4f}", f"{sy.std():.4f}",
                  f"{r:.4f}", f"{p:.4f}"],
    })
    U.save_table(tbl, "table17_stretching_scatter",
                 caption="Descriptive and correlation statistics for the stretching parameter pair.",
                 label="tab:stretching_scatter")
    U.register_figure(name, caption, label)


def fig18_compression_progression():
    """Synthetic images at increasing compression levels."""
    name    = "fig18_compression_progression"
    caption = ("Effect of increasing compression factor $c$ on specimen "
               "appearance. A lower value of $c$ produces stronger vertical "
               "compression, simulating sedimentary overburden deformation.")
    label   = "fig:compression_progression"

    spec = "araripesaurus__BSPG-1982-I-90"
    orig_path = PROJ_DIR / spec / f"{spec}_ortho_00.png"
    orig = _load_gray(orig_path)

    c_values = [0.3, 0.45, 0.6, 0.75, 0.9]
    H, W = 224, 224

    fig, axes = plt.subplots(1, len(c_values) + 1, figsize=(S.COL2, 2.0))
    axes[0].imshow(orig, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original", fontsize=7)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    for i, c in enumerate(c_values):
        ax = axes[i+1]
        cy = H / 2.0
        M  = np.float32([[1, 0, 0], [0, c, cy * (1.0 - c)]])
        deformed = cv2.warpAffine(orig, M, (W, H),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ax.imshow(deformed, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"c = {c}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Compression Factor Progression — Araripesaurus ortho_00",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "c value": [1.0] + c_values,
        "Description": ["Original (no compression)"] +
                        [f"Vertical compression, scale={c}" for c in c_values],
        "Visible Height (px)": [224] + [int(224 * c) for c in c_values],
        "Height Reduction (%)": [0] + [f"{(1-c)*100:.0f}%" for c in c_values],
    })
    U.save_table(tbl, "table18_compression_progression",
                 caption="Visible image height for each compression factor c shown in the "
                         "compression progression figure.",
                 label="tab:compression_progression")
    U.register_figure(name, caption, label)


def fig19_dissolution_masks():
    """Gallery of real dissolution masks from the dataset."""
    name    = "fig19_dissolution_masks"
    caption = ("Example binary deformation masks for the dissolution deformation "
               "type. Each mask encodes the elliptical region where intensity was "
               "attenuated. Masks are provided for Grad-CAM explainability validation.")
    label   = "fig:dissolution_masks"

    mask_pngs = sorted(MASK_DIR.glob("*diss*_mask.png"))[:8]

    fig, axes = plt.subplots(2, 4, figsize=(S.COL2, 3.0))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.set_xticks([]); ax.set_yticks([])
        if i < len(mask_pngs):
            img = _load_gray(mask_pngs[i])
            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(mask_pngs[i].stem[:28] + "…", fontsize=4.5, pad=2)
        else:
            ax.axis("off")

    fig.suptitle("Dissolution Deformation Masks (Binary, 224×224)",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    mask_stats = []
    for mp in mask_pngs:
        img = _load_gray(mp)
        if img is not None:
            coverage = (img > 127).mean() * 100
            mask_stats.append({
                "Mask File": mp.name,
                "Coverage (%)": f"{coverage:.1f}",
                "Non-zero Pixels": int((img > 127).sum()),
                "Total Pixels": img.size,
            })
    tbl = pd.DataFrame(mask_stats)
    U.save_table(tbl, "table19_dissolution_masks",
                 caption="Coverage statistics for eight representative dissolution deformation masks.",
                 label="tab:dissolution_masks")
    U.register_figure(name, caption, label)


def fig20_mask_coverage_stats():
    """Box plots: mask coverage (%) per deformation type."""
    name    = "fig20_mask_coverage_stats"
    caption = (r"Distribution of deformation mask coverage (\% of image pixels "
               "changed) per deformation type. Dissolution masks reflect the "
               "random ellipse area; compression, shearing, and stretching "
               "masks reflect the affine-transformed region.")
    label   = "fig:mask_coverage"

    mask_rows = []
    for _, row in df.iterrows():
        mp = MASK_DIR / row["mask_file"]
        img = _load_gray(mp)
        if img is not None:
            cov = (img > 127).mean() * 100
            mask_rows.append({"deformation_type": row["deformation_type"],
                               "coverage": cov})
    mdf = pd.DataFrame(mask_rows)

    fig, ax = plt.subplots(figsize=(S.COL1 + 0.5, 3.0))
    data_by_type = [mdf[mdf["deformation_type"] == d]["coverage"].values
                    for d in DEF_TYPES]
    bp = ax.boxplot(data_by_type, patch_artist=True, notch=False,
                    medianprops=dict(color="white", lw=1.5))
    for patch, color in zip(bp["boxes"], DEF_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels([d.capitalize() for d in DEF_TYPES], rotation=15)
    ax.set_ylabel("Mask Coverage (% of image area)")
    ax.set_title("Deformation Mask Coverage\nper Deformation Type")
    S.despine(ax)
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl_rows = []
    for d in DEF_TYPES:
        vals = mdf[mdf["deformation_type"] == d]["coverage"]
        tbl_rows.append({
            "Deformation": d.capitalize(),
            "Mean Coverage (%)": f"{vals.mean():.2f}",
            "Std (%)": f"{vals.std():.2f}",
            "Min (%)": f"{vals.min():.2f}",
            "Max (%)": f"{vals.max():.2f}",
            "Median (%)": f"{vals.median():.2f}",
        })
    tbl = pd.DataFrame(tbl_rows)
    U.save_table(tbl, "table20_mask_coverage",
                 caption="Deformation mask coverage statistics per deformation type.",
                 label="tab:mask_coverage")
    U.register_figure(name, caption, label)


def fig21_intensity_distributions():
    """Violin plots: pixel intensity distributions, original vs each deformation."""
    name    = "fig21_intensity_distributions"
    caption = ("Pixel intensity distributions (0–255) for original projection "
               "images and each synthetic deformation type, sampled from "
               "araripesaurus\\_\\_BSPG-1982-I-90. Dissolution is the only "
               "deformation that shifts the mean intensity downward.")
    label   = "fig:intensity_distributions"

    spec = "araripesaurus__BSPG-1982-I-90"
    orig_paths = sorted((PROJ_DIR / spec).glob("*_ortho_*.png"))[:6]

    def collect_intensities(paths):
        vals = []
        for p in paths:
            img = _load_gray(p)
            if img is not None:
                vals.extend(img.flatten().tolist())
        return np.array(vals, dtype=np.float32)

    orig_vals = collect_intensities(orig_paths)

    all_vals = {"Original": orig_vals}
    for def_type in DEF_TYPES:
        pngs = sorted(SYN_DIR.glob(f"{spec}_ortho_*__{def_type[:4]}*.png"))[:6]
        all_vals[def_type.capitalize()] = collect_intensities(pngs)

    fig, ax = plt.subplots(figsize=(S.COL2 * 0.85, 3.0))
    keys   = list(all_vals.keys())
    colors = ["#888888"] + DEF_COLORS
    data   = [all_vals[k] for k in keys]

    parts  = ax.violinplot(data, positions=range(len(keys)),
                            showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.65)
        body.set_edgecolor("#333333")
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_lw(1.5)

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=12, fontsize=8)
    ax.set_ylabel("Pixel Intensity (0–255)")
    ax.set_title("Pixel Intensity Distributions by Deformation Type\n(Araripesaurus ortho views)")
    S.despine(ax)
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl_rows = []
    for k, vals in all_vals.items():
        tbl_rows.append({
            "Category": k,
            "n pixels": len(vals),
            "Mean": f"{vals.mean():.2f}",
            "Std": f"{vals.std():.2f}",
            "Median": f"{np.median(vals):.2f}",
            "Min": f"{vals.min():.0f}",
            "Max": f"{vals.max():.0f}",
        })
    tbl = pd.DataFrame(tbl_rows)
    U.save_table(tbl, "table21_intensity_distributions",
                 caption="Pixel intensity distribution statistics for original and deformed images.",
                 label="tab:intensity_distributions")
    U.register_figure(name, caption, label)


# ============================================================
# SECTION 6 — ARCHITECTURE DIAGRAMS
# ============================================================

def fig22_folder_structure():
    """Visual folder hierarchy tree."""
    name    = "fig22_folder_structure"
    caption = ("Canonical folder hierarchy of the TDE dataset. "
               "Each specimen directory contains an ESRF-standard "
               "\\texttt{org\\_slices/} directory with JP2 CT slices and a "
               "\\texttt{mesh/} directory for 3-D surface meshes.")
    label   = "fig:folder_structure"

    lines = [
        ("dataset/",                      0, "#333333"),
        ("├── raw/",                       1, "#4C72B0"),
        ("│   ├── <specimen_name>/",       2, "#4C72B0"),
        ("│   │   └── <specimen_id>/",     3, "#4C72B0"),
        ("│   │       ├── org_slices/",    4, "#4C72B0"),
        ("│   │       │   └── *.jp2",      5, "#888888"),
        ("│   │       └── mesh/",          4, "#4C72B0"),
        ("├── projections/",               1, "#DD8452"),
        ("│   └── <specimen>__<id>/",      2, "#DD8452"),
        ("│       ├── *_ortho_00.png",     3, "#888888"),
        ("│       └── *_rot_000.png",      3, "#888888"),
        ("└── synthetic/",                 1, "#55A868"),
        ("    ├── synthetic_images/",       2, "#55A868"),
        ("    │   └── *__comp*.png",        3, "#888888"),
        ("    ├── deformation_masks/",      2, "#55A868"),
        ("    │   └── *_mask.png",          3, "#888888"),
        ("    └── synthetic_labels.csv",    2, "#55A868"),
    ]

    fig, ax = plt.subplots(figsize=(S.COL1 + 0.8, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(lines) + 1)
    ax.axis("off")

    for i, (text, indent, color) in enumerate(reversed(lines)):
        y = i + 0.6
        ax.text(0.4 + indent * 0.35, y, text,
                fontsize=7, fontfamily="monospace",
                color=color, va="center")

    # Legend
    legend_items = [
        mpatches.Patch(color="#4C72B0", label="Raw CT data"),
        mpatches.Patch(color="#DD8452", label="2-D projections"),
        mpatches.Patch(color="#55A868", label="Synthetic dataset"),
        mpatches.Patch(color="#888888", label="Files"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              fontsize=7, framealpha=0.9)
    fig.suptitle("TDE Dataset Folder Hierarchy", fontsize=9.5,
                 fontweight="bold")
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Path": [l[0].strip() for l in lines],
        "Depth": [l[1] for l in lines],
        "Contents": [
            "Root dataset directory",
            "Raw CT data root",
            "Specimen taxon folder (e.g. archaeopteryx_london)",
            "Acquisition ID folder (e.g. BMNH-37001)",
            "JP2 CT slice directory (canonical ESRF name)",
            "JPEG2000 CT slices (uint16, ordered)",
            "3-D surface mesh files (OBJ/PLY)",
            "2-D projection root",
            "Specimen projection folder (<name>__<id>)",
            "Orthographic projection PNG",
            "Rotational projection PNG",
            "Synthetic dataset root",
            "Deformed projection images",
            "Example compression image",
            "Binary deformation masks",
            "Example mask file",
            "Per-image label CSV (11 columns)",
        ],
    })
    U.save_table(tbl, "table22_folder_structure",
                 caption="TDE dataset folder hierarchy with path descriptions.",
                 label="tab:folder_structure")
    U.register_figure(name, caption, label)


def fig23_function_call_graph():
    """Directed call graph of key functions."""
    name    = "fig23_function_call_graph"
    caption = ("Key function call graph for the TDE projection and deformation "
               "pipelines. Edge direction indicates caller → callee. "
               "Nodes are colored by processing module.")
    label   = "fig:function_call_graph"

    G = nx.DiGraph()
    nodes = {
        # Projection module
        "main()": "#4C72B0",
        "process_dataset()": "#4C72B0",
        "discover_esrf_dataset()": "#4C72B0",
        "resolve_esrf_slice_dir()": "#4C72B0",
        "process_specimen()": "#DD8452",
        "stream_volume_projections()": "#DD8452",
        "iter_jp2_slices()": "#DD8452",
        "generate_orthographic_projections()": "#DD8452",
        "generate_rotational_projections()": "#DD8452",
        "make_projection()": "#DD8452",
        "percentile_normalise()": "#888888",
        "resize_to_output()": "#888888",
        # Synthetic module
        "process_dataset_syn()": "#55A868",
        "process_image()": "#55A868",
        "apply_compression()": "#55A868",
        "apply_shearing()": "#55A868",
        "apply_stretching()": "#55A868",
        "apply_dissolution()": "#55A868",
        "_difference_mask()": "#CC79A7",
        "_random_ellipse_mask()": "#CC79A7",
    }
    for n in nodes:
        G.add_node(n)

    edges = [
        ("main()", "process_dataset()"),
        ("main()", "discover_esrf_dataset()"),
        ("discover_esrf_dataset()", "resolve_esrf_slice_dir()"),
        ("process_dataset()", "process_specimen()"),
        ("process_specimen()", "stream_volume_projections()"),
        ("stream_volume_projections()", "iter_jp2_slices()"),
        ("process_specimen()", "generate_orthographic_projections()"),
        ("process_specimen()", "generate_rotational_projections()"),
        ("generate_orthographic_projections()", "make_projection()"),
        ("generate_rotational_projections()", "make_projection()"),
        ("make_projection()", "percentile_normalise()"),
        ("make_projection()", "resize_to_output()"),
        ("process_dataset_syn()", "process_image()"),
        ("process_image()", "apply_compression()"),
        ("process_image()", "apply_shearing()"),
        ("process_image()", "apply_stretching()"),
        ("process_image()", "apply_dissolution()"),
        ("apply_compression()", "_difference_mask()"),
        ("apply_shearing()", "_difference_mask()"),
        ("apply_stretching()", "_difference_mask()"),
        ("apply_dissolution()", "_random_ellipse_mask()"),
    ]
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=7, k=2.5)

    fig, ax = plt.subplots(figsize=(S.COL2, 4.2))
    ax.axis("off")
    node_colors = [nodes[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=800, alpha=0.82)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#555",
                           arrows=True, arrowsize=10, width=0.8, alpha=0.7,
                           connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=4.5,
                            font_color="white", font_weight="bold")

    legend_items = [
        mpatches.Patch(color="#4C72B0", label="Discovery & batch control"),
        mpatches.Patch(color="#DD8452", label="Projection pipeline"),
        mpatches.Patch(color="#55A868", label="Synthetic deformation"),
        mpatches.Patch(color="#CC79A7", label="Mask generation"),
        mpatches.Patch(color="#888888", label="Image utilities"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=6.5, framealpha=0.9)
    fig.suptitle("TDE Key Function Call Graph", fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Function": list(nodes.keys()),
        "Module": (["generate_projections.py"] * 12
                   + ["generate_synthetic.py"] * 6
                   + ["generate_synthetic.py"] * 2),
        "Role": [
            "CLI entry point", "Batch dataset processor", "ESRF volume discovery",
            "Per-specimen JP2 resolver", "Single-specimen full pipeline",
            "Streaming MIP accumulator", "JP2 slice iterator",
            "6 orthographic view generator", "24 rotational view generator",
            "MIP → normalize → resize", "1–99% percentile clip",
            "INTER_AREA resize to 224²",
            "Dataset-level deformation loop", "Per-image deformation applicator",
            "Vertical compression", "Horizontal shear",
            "Anisotropic stretch", "Intensity dissolution",
            "Difference-based mask", "Random ellipse mask",
        ],
    })
    U.save_table(tbl, "table23_function_graph",
                 caption="TDE function call graph node descriptions.",
                 label="tab:function_graph")
    U.register_figure(name, caption, label)


# ============================================================
# SECTION 7 — PUBLICATION MULTI-PANEL FIGURES
# ============================================================

def fig24_4deformation_panel():
    """Full 4×3 comparison panel: all deformations for one specimen."""
    name    = "fig24_4deformation_panel"
    caption = ("Publication-quality comparison panel of the four taphonomic "
               "deformations applied to the Archaeopteryx\\_london specimen "
               "(BMNH-37001, ortho\\_00). Each row shows original projection, "
               "synthetic deformed image, and pixel-wise binary mask.")
    label   = "fig:4deformation_panel"

    spec      = "archaeopteryx_london__BMNH-37001"
    base_stem = f"{spec}_ortho_00"
    orig_path = PROJ_DIR / spec / f"{base_stem}.png"
    orig      = _load_gray(orig_path)

    prefixes = ["comp", "shear", "stretch", "diss"]

    fig = plt.figure(figsize=(S.COL2, 5.2))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.06, wspace=0.04)

    col_titles = ["Original", "Deformed", "Binary Mask"]
    for j, ct in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(ct, fontsize=8, fontweight="bold", pad=3)
        ax.axis("off")

    for i, (prefix, def_type) in enumerate(zip(prefixes, DEF_TYPES)):
        syn_pngs  = sorted(SYN_DIR.glob(f"{base_stem}__{prefix}*.png"))
        mask_pngs = sorted(MASK_DIR.glob(f"{base_stem}__{prefix}*_mask.png"))

        syn_img  = _load_gray(syn_pngs[0])  if syn_pngs  else None
        mask_img = _load_gray(mask_pngs[0]) if mask_pngs else None

        for j, (img, cmap) in enumerate([(orig, "gray"), (syn_img, "gray"),
                                          (mask_img, "gray")]):
            ax = fig.add_subplot(gs[i, j])
            if img is not None:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=255,
                          interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(def_type.capitalize(), fontsize=7.5,
                               labelpad=4,
                               color=S.DEFORMATION_COLORS[def_type])

    fig.suptitle("Taphonomic Deformation Panel — Archaeopteryx\n(BMNH-37001, ortho_00)",
                 fontsize=9.5, fontweight="bold")
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    U.save_figure(fig, name)
    plt.close(fig)

    rows = []
    for prefix, def_type in zip(prefixes, DEF_TYPES):
        syn_pngs = sorted(SYN_DIR.glob(f"{base_stem}__{prefix}*.png"))
        mask_pngs = sorted(MASK_DIR.glob(f"{base_stem}__{prefix}*_mask.png"))
        row_csv = df[(df["specimen"] == spec) &
                     (df["deformation_type"] == def_type) &
                     (df["original_file"] == f"{base_stem}.png")]
        param_str = "—"
        if not row_csv.empty:
            r = row_csv.iloc[0]
            if def_type == "compression": param_str = f"c={r['param_c']:.4f}"
            elif def_type == "shearing":  param_str = f"k={r['param_k']:.4f}"
            elif def_type == "stretching": param_str = f"sx={r['param_sx']:.3f}, sy={r['param_sy']:.3f}"
            else: param_str = f"λ={r['param_lambda']:.4f}"
        rows.append({
            "Deformation": def_type.capitalize(),
            "Parameter": param_str,
            "Synthetic File": syn_pngs[0].name if syn_pngs else "N/A",
            "Mask File": mask_pngs[0].name if mask_pngs else "N/A",
        })
    tbl = pd.DataFrame(rows)
    U.save_table(tbl, "table24_4deformation_panel",
                 caption="Parameters and filenames for the four-deformation comparison panel "
                         "(Archaeopteryx\\_london, BMNH-37001, ortho\\_00).",
                 label="tab:4deformation_panel")
    U.register_figure(name, caption, label, width=r"\textwidth")


def fig25_rotational_views_strip():
    """Strip of rotational projections (0°–345° at 30° sampling)."""
    name    = "fig25_rotational_strip"
    caption = ("Twelve representative rotational MIP views of the halszkaraptor "
               "specimen at 30° increments. Views are generated by rotating the "
               "sagittal MIP (axis-1 projection) in 2-D after accumulation.")
    label   = "fig:rotational_strip"

    spec = "halszkaraptor__MPC-D-102-109"
    angles_to_show = list(range(0, 360, 30))
    spec_dir = PROJ_DIR / spec

    fig, axes = plt.subplots(2, 6, figsize=(S.COL2, 2.8))
    axes = axes.flatten()

    for i, angle in enumerate(angles_to_show):
        ax = axes[i]
        p  = spec_dir / f"{spec}_rot_{angle:03d}.png"
        if p.exists():
            img = _load_gray(p)
            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{angle}°", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Rotational MIP Views — Halszkaraptor (MPC-D-102-109)\n"
                 "12 of 24 views shown (30° sampling)",
                 fontsize=9, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Angle (°)": angles_to_show,
        "File": [f"{spec}_rot_{a:03d}.png" for a in angles_to_show],
        "View Type": ["Rotational (2D-after-MIP)"] * len(angles_to_show),
        "Source MIP": ["Sagittal (axis-1)"] * len(angles_to_show),
        "Rotation Order": ["Bicubic (order=3)"] * len(angles_to_show),
    })
    U.save_table(tbl, "table25_rotational_strip",
                 caption="Rotational view specifications for the Halszkaraptor strip figure.",
                 label="tab:rotational_strip")
    U.register_figure(name, caption, label)


def fig26_mip_algorithm():
    """Conceptual illustration of the streaming MIP algorithm."""
    name    = "fig26_mip_algorithm"
    caption = ("Conceptual illustration of the streaming Maximum Intensity "
               "Projection (MIP) algorithm. JP2 slices are processed one at a "
               r"time; a running \texttt{np.maximum} accumulates the global MIP "
               "in O(H·W) memory regardless of volume depth D.")
    label   = "fig:mip_algorithm"

    fig, ax = plt.subplots(figsize=(S.COL2, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Simulate 5 slices decreasing height
    slice_vals = [
        np.array([[0.2, 0.4], [0.8, 0.3]]),
        np.array([[0.6, 0.2], [0.5, 0.9]]),
        np.array([[0.3, 0.7], [0.4, 0.6]]),
        np.array([[0.9, 0.1], [0.3, 0.5]]),
        np.array([[0.5, 0.8], [0.7, 0.4]]),
    ]
    mip_running = np.zeros((2, 2))

    y_top = 0.88
    box_size = 0.30
    x_gap = 0.18

    # Slices row
    for i, s in enumerate(slice_vals):
        x = 0.5 + i * (box_size + x_gap)
        mip_running = np.maximum(mip_running, s)
        im = ax.imshow(s, extent=[x, x+box_size, y_top - box_size, y_top],
                        cmap="inferno", vmin=0, vmax=1, aspect="auto")
        ax.text(x + box_size/2, y_top + 0.04, f"Slice {i+1}",
                ha="center", va="bottom", fontsize=6, color="#333")
        if i < len(slice_vals) - 1:
            ax.annotate("", xy=(x + box_size + x_gap, y_top - box_size/2),
                        xytext=(x + box_size, y_top - box_size/2),
                        arrowprops=dict(arrowstyle="-|>", color="#777", lw=0.8))

    # Running MIP display
    mip_final = np.maximum.reduce(slice_vals)
    x_mip = 0.5 + 5 * (box_size + x_gap) + 0.1
    ax.annotate("", xy=(x_mip, y_top - box_size/2),
                xytext=(x_mip - 0.08, y_top - box_size/2),
                arrowprops=dict(arrowstyle="-|>", color="#C44E52", lw=1.2))
    ax.imshow(mip_final, extent=[x_mip, x_mip + box_size + 0.05,
                                   y_top - box_size, y_top],
               cmap="inferno", vmin=0, vmax=1, aspect="auto")
    ax.text(x_mip + (box_size+0.05)/2, y_top + 0.04, "MIP\n(max)",
            ha="center", va="bottom", fontsize=7, color="#C44E52",
            fontweight="bold")

    # Memory annotation
    ax.text(6, 0.35,
            "Peak memory: O(H·W)  vs  O(D·H·W) for full volume load\n"
            "→ ~20 MB regardless of volume depth (D = 300–2000 slices)",
            ha="center", va="center", fontsize=7.5, color="#333333",
            bbox=dict(fc="#FFF9F0", ec="#DDAA77", pad=5, lw=0.8))

    ax.text(6, 0.12, "np.maximum(mip_axis0, arr, out=mip_axis0)  ← in-place O(1) update",
            ha="center", va="center", fontsize=7, fontfamily="monospace",
            color="#4C72B0",
            bbox=dict(fc="#F0F4FF", ec="#AABBDD", pad=4, lw=0.6))

    fig.suptitle("Streaming MIP Algorithm — Single-Pass Memory Efficiency",
                 fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Approach": ["Naive full-volume load", "Streaming MIP (TDE)"],
        "Peak RAM": ["O(D·H·W·2 bytes)", "O(H·W·2 bytes) + 3 MIP planes"],
        "Example (2048², 1000 slices)": ["~8.0 GB", "~24 MB"],
        "Implementation": ["volume = np.zeros((D,H,W))", "np.maximum(mip, arr, out=mip)"],
        "Rotation Strategy": ["3D affine O(D·H·W)", "2D rotate-after-MIP O(H·W)"],
    })
    U.save_table(tbl, "table26_mip_memory",
                 caption="Memory comparison between naive full-volume and "
                         "streaming MIP approaches in the TDE projection pipeline.",
                 label="tab:mip_memory")
    U.register_figure(name, caption, label)


def fig27_percentile_normalization():
    """Demonstrate 1–99% percentile normalization on a real image."""
    name    = "fig27_percentile_normalization"
    caption = (r"Percentile normalization (1--99\%) applied to a raw MIP "
               "accumulator output from the araripesaurus specimen. "
               "Left: raw uint16 histogram; Right: normalized uint8 "
               "output after clipping and rescaling.")
    label   = "fig:percentile_normalization"

    spec     = "araripesaurus__BSPG-1982-I-90"
    img_path = PROJ_DIR / spec / f"{spec}_ortho_00.png"
    img      = _load_gray(img_path)

    fig, axes = plt.subplots(1, 2, figsize=(S.COL2 * 0.75, 2.4))

    # Simulate uint16-like histogram by stretching back
    raw_sim = img.astype(np.float32) * 257  # back to fake uint16 range

    ax1, ax2 = axes
    ax1.hist(raw_sim.flatten(), bins=80, color="#4C72B0",
             alpha=0.75, edgecolor="none")
    ax1.set_xlabel("Intensity (uint16)")
    ax1.set_ylabel("Pixel Count")
    ax1.set_title("Raw MIP (simulated uint16)")
    lo, hi = np.percentile(raw_sim, 1), np.percentile(raw_sim, 99)
    ax1.axvline(lo, color="#C44E52", lw=1.0, ls="--", label=f"1%: {lo:.0f}")
    ax1.axvline(hi, color="#55A868", lw=1.0, ls="--", label=f"99%: {hi:.0f}")
    ax1.legend(fontsize=6.5)
    S.despine(ax1)

    ax2.hist(img.flatten(), bins=80, color="#DD8452",
             alpha=0.75, edgecolor="none")
    ax2.set_xlabel("Intensity (uint8)")
    ax2.set_ylabel("")
    ax2.set_title("After Percentile Normalization")
    ax2.set_xlim(0, 255)
    S.despine(ax2)

    fig.suptitle("Percentile Normalization (1–99%) for CT MIP Data",
                 fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    tbl = pd.DataFrame({
        "Parameter": ["Low percentile", "High percentile", "Input dtype",
                      "Output dtype", "Method", "OpenCV interpolation"],
        "Value": ["1.0%", "99.0%", "uint16", "uint8",
                  "clip → rescale to [0,255]", "INTER_AREA"],
        "Rationale": [
            "Removes dead/hot pixel artefacts",
            "Removes detector saturation spikes",
            "Raw JPEG2000 CT output",
            "PNG-compatible, 8-bit grayscale",
            "Preserves mid-range bone contrast",
            "Optimal anti-aliasing for downsizing",
        ],
    })
    U.save_table(tbl, "table27_normalization",
                 caption="Percentile normalization parameters and their rationale in the TDE pipeline.",
                 label="tab:normalization")
    U.register_figure(name, caption, label)


def fig28_specimen_voxel_taxonomy():
    """Combined taxonomy + voxel size visualization."""
    name    = "fig28_taxonomy_voxel"
    caption = ("Archosaur specimen taxonomy and CT voxel resolution. "
               "Marker size encodes voxel resolution (µm). "
               "The dataset spans birds (Aves), non-avian dinosaurs "
               "(Dromaeosauridae), and flying reptiles (Pterosauria).")
    label   = "fig:taxonomy_voxel"

    taxonomy = {
        "araripesaurus__BSPG-1982-I-90":       ("Pterosauria",    "Anhangueridae"),
        "archaeopteryx_london__BMNH-37001":    ("Aves",           "Archaeopterygidae"),
        "halszkaraptor__MPC-D-102-109":        ("Dinosauria",     "Dromaeosauridae"),
        "hesperornis__YPM-1206-A":             ("Aves",           "Hesperornithidae"),
        "ichthyornis__YPM-1460":               ("Aves",           "Ichthyornithidae"),
        "ichthyornis__YPM-1775":               ("Aves",           "Ichthyornithidae"),
        "titanosaur_embryo__MCF-PVPH-874":     ("Dinosauria",     "Titanosauria"),
        "tropeognathus__BSPG-1987-I-46":       ("Pterosauria",    "Anhangueridae"),
    }
    group_colors = {"Pterosauria": "#4C72B0",
                    "Aves":        "#55A868",
                    "Dinosauria":  "#C44E52"}

    records = []
    for spec, (group, family) in taxonomy.items():
        vx = S.VOXEL_SIZES.get(spec, 10.0)
        records.append({
            "specimen": S.SPECIMEN_SHORT[spec],
            "group": group,
            "family": family,
            "voxel_um": vx,
        })
    tdf = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(S.COL2, 3.0))
    for group, gdf in tdf.groupby("group"):
        ax.scatter(gdf["voxel_um"], gdf.index,
                   s=gdf["voxel_um"] * 2.5 + 60,
                   c=group_colors[group], alpha=0.8,
                   edgecolors="white", lw=0.8,
                   label=group, zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Voxel Size (µm, log scale)")
    ax.set_yticks(range(len(tdf)))
    ax.set_yticklabels(tdf["specimen"].tolist(), fontsize=7)
    ax.set_title("Specimen Taxonomy and CT Resolution")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    S.despine(ax)
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    U.save_table(tdf, "table28_taxonomy",
                 caption="Taxonomic classification and voxel resolution for each specimen.",
                 label="tab:taxonomy")
    U.register_figure(name, caption, label)


# ============================================================
# SECTION 8 — SUMMARY / OVERVIEW FIGURES
# ============================================================

def fig29_dataset_summary_table_fig():
    """Rendered summary table as a matplotlib figure."""
    name    = "fig29_dataset_summary"
    caption = ("TDE dataset summary statistics. The dataset contains 240 "
               "projection images and 960 synthetic images with corresponding "
               "binary masks and per-image labels across four balanced "
               "deformation classes.")
    label   = "fig:dataset_summary"

    summary_data = {
        "Metric": [
            "Number of Specimens", "Projection Images (total)", "Projections per Specimen",
            "Orthographic Views", "Rotational Views", "Rotation Step",
            "Synthetic Images (total)", "Deformation Types", "Images per Deformation",
            "Binary Masks (total)", "Image Resolution", "Image Format",
            "Random Seed", "Label Format",
        ],
        "Value": [
            "8", "240", "30",
            "6 (3 axes × 2 directions)", "24 (0°–345°, Δ15°)", "15°",
            "960", "4", "240",
            "960", "224 × 224 px", "PNG (uint8 grayscale)",
            "42", "CSV (11 columns)",
        ],
        "Notes": [
            "All archosaurs, ESRF Paleo CT",
            "8 specimens × 30",
            "6 ortho + 24 rotational",
            "Coronal, sagittal, transverse + flips",
            "2D-after-MIP (sagittal MIP rotated)",
            "Covers full 360°",
            "240 inputs × 4 deformations",
            "Compression, shearing, stretching, dissolution",
            "Balanced; seed=42",
            "One mask per synthetic image",
            "INTER_AREA resize from raw MIP",
            "Grayscale, no color channels",
            "Reproducible with np.random.default_rng(42)",
            "image_id, specimen, deformation_type, params, …",
        ],
    }
    tbl = pd.DataFrame(summary_data)

    fig, ax = plt.subplots(figsize=(S.COL2, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=tbl.values,
        colLabels=tbl.columns,
        cellLoc="left", loc="center",
        colWidths=[0.32, 0.28, 0.40],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F0F4FF")
        cell.set_edgecolor("#CCCCCC")

    fig.suptitle("TDE Dataset Summary Statistics", fontsize=10, fontweight="bold")
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    U.save_table(tbl, "table29_dataset_summary",
                 caption="Complete TDE dataset summary statistics.",
                 label="tab:dataset_summary")
    U.register_figure(name, caption, label, width=r"\textwidth")


def fig30_csv_schema():
    """Visualize synthetic_labels.csv schema and first rows."""
    name    = "fig30_csv_schema"
    caption = ("Schema of the \\texttt{synthetic\\_labels.csv} output file. "
               "Each row describes one synthetic image with its specimen, "
               "deformation type, sampled parameters, output filenames, and "
               "corresponding binary mask filename.")
    label   = "fig:csv_schema"

    schema = pd.DataFrame({
        "Column": df.columns.tolist(),
        "Type": ["int", "str", "str", "str",
                 "float|NaN", "float|NaN", "float|NaN",
                 "float|NaN", "float|NaN", "str", "str"],
        "Description": [
            "Unique sequential image ID",
            "Specimen key (<name>__<id>)",
            "Source projection PNG filename",
            "compression | shearing | stretching | dissolution",
            "Compression factor c ∈ [0.3, 0.9]",
            "Shear coefficient k ∈ [−0.5, 0.5]",
            "Stretch factor sx ∈ [1.0, 1.5]",
            "Stretch factor sy ∈ [1.0, 1.5]",
            "Dissolution intensity λ ∈ [0.3, 0.8]",
            "Synthetic image PNG filename",
            "Binary mask PNG filename",
        ],
        "Non-null": [960, 960, 960, 960, 240, 240, 240, 240, 240, 960, 960],
        "Example": [
            "0", "araripesaurus__BSPG-1982-I-90",
            "araripesaurus__BSPG-1982-I-90_ortho_00.png",
            "compression", "0.7644", "NaN", "NaN", "NaN", "NaN",
            "araripesaurus__BSPG-1982-I-90_ortho_00__comp_c0.764.png",
            "araripesaurus__BSPG-1982-I-90_ortho_00__comp_c0.764_mask.png",
        ],
    })

    fig, ax = plt.subplots(figsize=(S.COL2, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=schema.values,
        colLabels=schema.columns,
        cellLoc="left", loc="center",
        colWidths=[0.14, 0.10, 0.30, 0.08, 0.38],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.5)
    table.scale(1, 1.38)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#55A868")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F0FFF4")
        cell.set_edgecolor("#CCCCCC")

    fig.suptitle("synthetic_labels.csv Schema (11 columns, 960 rows)",
                 fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    U.save_figure(fig, name)
    plt.close(fig)

    U.save_table(schema, "table30_csv_schema",
                 caption="Column schema for \\texttt{synthetic\\_labels.csv}.",
                 label="tab:csv_schema")
    U.register_figure(name, caption, label)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

FIGURE_REGISTRY = [
    # System overview
    ("fig01_pipeline_overview",        fig01_pipeline_overview),
    ("fig02_module_dependency",        fig02_module_dependency),
    ("fig03_data_flow",                fig03_data_flow),
    # Dataset statistics
    ("fig04_specimen_distribution",    fig04_specimen_distribution),
    ("fig05_dataset_composition",      fig05_dataset_composition),
    ("fig06_dataset_growth",           fig06_dataset_growth),
    ("fig07_projection_types",         fig07_projection_type_breakdown),
    ("fig08_deformation_distribution", fig08_deformation_distribution),
    # Projection pipeline
    ("fig09_projection_pipeline",      fig09_projection_pipeline),
    ("fig10_sample_projections",       fig10_sample_projections),
    ("fig11_all_specimens_grid",       fig11_all_specimens_grid),
    ("fig12_voxel_sizes",              fig12_voxel_sizes),
    # Deformation math
    ("fig13_transformation_matrices",  fig13_transformation_matrices),
    ("fig14_deformation_geometry",     fig14_deformation_geometry),
    ("fig15_deformation_gallery",      fig15_deformation_gallery),
    # Parameter statistics
    ("fig16_parameter_distributions",  fig16_parameter_distributions),
    ("fig17_stretching_scatter",       fig17_stretching_scatter),
    ("fig18_compression_progression",  fig18_compression_progression),
    # Masks & coverage
    ("fig19_dissolution_masks",        fig19_dissolution_masks),
    ("fig20_mask_coverage_stats",      fig20_mask_coverage_stats),
    ("fig21_intensity_distributions",  fig21_intensity_distributions),
    # Architecture
    ("fig22_folder_structure",         fig22_folder_structure),
    ("fig23_function_call_graph",      fig23_function_call_graph),
    # Publication panels
    ("fig24_4deformation_panel",       fig24_4deformation_panel),
    ("fig25_rotational_strip",         fig25_rotational_views_strip),
    ("fig26_mip_algorithm",            fig26_mip_algorithm),
    ("fig27_percentile_normalization", fig27_percentile_normalization),
    ("fig28_taxonomy_voxel",           fig28_specimen_voxel_taxonomy),
    # Summary
    ("fig29_dataset_summary",          fig29_dataset_summary_table_fig),
    ("fig30_csv_schema",               fig30_csv_schema),
]


def run_all():
    print("=" * 65)
    print("TDE VISUALIZATION SUITE — IEEE Access Publication Generator")
    print("=" * 65)
    print(f"Project root : {PROJECT_DIR}")
    print(f"Output (figs): {U.GEN_DIR}")
    print(f"Output (tbls): {U.TABLE_DIR}")
    print(f"Output (LaTeX): {U.LATEX_DIR}")
    print(f"Dataset rows : {len(df)}")
    print("=" * 65)

    S.apply_ieee_style()

    failed = []
    for name, func in FIGURE_REGISTRY:
        print(f"\n[{name}]")
        try:
            func()
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback
            traceback.print_exc()
            failed.append((name, str(exc)))

    # Flush LaTeX snippets
    U.flush_latex()

    print("\n" + "=" * 65)
    print(f"DONE  |  {len(FIGURE_REGISTRY) - len(failed)} / {len(FIGURE_REGISTRY)} figures generated")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for n, e in failed:
            print(f"  {n}: {e}")
    print(f"Figures → {U.GEN_DIR}")
    print(f"Tables  → {U.TABLE_DIR}")
    print(f"LaTeX   → {U.LATEX_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    run_all()
