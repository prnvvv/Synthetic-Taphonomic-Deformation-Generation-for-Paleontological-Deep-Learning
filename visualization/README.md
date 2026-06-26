# TDE Visualization Suite — IEEE Access Publication Generator

Complete publication-quality figure and table generator for the
**Synthetic Taphonomic Deformation Generation for Paleontological Deep Learning** paper.

---

## Quick start

```bash
cd visualization/
python generate_all_visualizations.py
```

Or open the notebook and click **Kernel → Restart & Run All**:

```bash
jupyter notebook generate_all_visualizations.ipynb
```

---

## Output structure

```
visualization/
├── generate_all_visualizations.py   ← main script (run this)
├── generate_all_visualizations.ipynb ← documented notebook version
├── style.py                          ← IEEE-compatible Matplotlib style
├── utils.py                          ← export utilities (PNG/PDF/SVG/CSV/XLSX/MD/LaTeX)
│
├── generated/                        ← 90 figure files (30 × PNG + PDF + SVG)
│   ├── fig01_pipeline_overview.png
│   ├── fig01_pipeline_overview.pdf
│   ├── fig01_pipeline_overview.svg
│   └── ... (fig02 through fig30)
│
├── tables/                           ← 120 table files (30 × CSV + XLSX + MD + LaTeX)
│   ├── table01_pipeline_stages.csv
│   ├── table01_pipeline_stages.xlsx
│   ├── table01_pipeline_stages.md
│   ├── table01_pipeline_stages.tex
│   └── ... (table02 through table30)
│
└── latex/
    ├── figures.tex    ← \begin{figure}…\end{figure} for all 30 figures
    ├── tables.tex     ← \begin{table}…\end{table} for all 30 tables
    ├── captions.tex   ← caption reference list
    └── labels.tex     ← \label cross-reference commands
```

---

## Figure inventory

| # | File | Contents | Paper Section |
|---|------|----------|---------------|
| 01 | `fig01_pipeline_overview` | End-to-end 5-stage pipeline flowchart | §II / §III |
| 02 | `fig02_module_dependency` | Software module dependency graph | §III |
| 03 | `fig03_data_flow` | Data flow: JP2→MIP→PNG→deformation | §III-A |
| 04 | `fig04_specimen_distribution` | Projections per specimen (bar) | §IV |
| 05 | `fig05_dataset_composition` | Dataset composition by specimen (pie) | §IV |
| 06 | `fig06_dataset_growth` | Dataset growth ×30 ×4 through pipeline | §IV |
| 07 | `fig07_projection_types` | View type split + polar angle coverage | §III-B |
| 08 | `fig08_deformation_distribution` | Balanced deformation class bar chart | §III-C |
| 09 | `fig09_projection_pipeline` | Streaming MIP block diagram | §III-B |
| 10 | `fig10_sample_projections` | Real ortho + rotational images, 4 specimens | §IV |
| 11 | `fig11_all_specimens_grid` | All 8 specimens ortho_00 grid | §IV |
| 12 | `fig12_voxel_sizes` | CT voxel resolution log-scale bar | §IV |
| 13 | `fig13_transformation_matrices` | Affine matrices + dissolution formula | §III-C |
| 14 | `fig14_deformation_geometry` | Geometric effect on unit square | §III-C |
| 15 | `fig15_deformation_gallery` | Real Before/After/Mask gallery | §III-C / §III-D |
| 16 | `fig16_parameter_distributions` | 4-panel parameter histograms (CSV data) | §V |
| 17 | `fig17_stretching_scatter` | sx vs sy scatter + Pearson r | §V |
| 18 | `fig18_compression_progression` | c=0.3→0.9 progression on real image | §III-C |
| 19 | `fig19_dissolution_masks` | Real dissolution mask gallery | §III-D |
| 20 | `fig20_mask_coverage_stats` | Mask coverage box plots by type | §V |
| 21 | `fig21_intensity_distributions` | Pixel intensity violins (orig vs deformed) | §V |
| 22 | `fig22_folder_structure` | ESRF dataset folder hierarchy tree | §III-A |
| 23 | `fig23_function_call_graph` | Key function call graph (20 nodes) | §III |
| 24 | `fig24_4deformation_panel` | **Main figure**: 4×3 deformation panel | §III-C |
| 25 | `fig25_rotational_strip` | 12 rotational views at 30° sampling | §III-B |
| 26 | `fig26_mip_algorithm` | Streaming MIP memory illustration | §III-B |
| 27 | `fig27_percentile_normalization` | 1–99% normalization histogram | §III-B |
| 28 | `fig28_taxonomy_voxel` | Taxonomy + voxel size scatter | §IV |
| 29 | `fig29_dataset_summary` | Summary statistics table figure | §IV |
| 30 | `fig30_csv_schema` | synthetic_labels.csv schema | §III-C |

---

## LaTeX integration

Copy the `generated/*.pdf` files to your paper's `figures/` folder, then in your LaTeX document:

```latex
\input{figures.tex}   % includes all 30 \begin{figure} environments
\input{tables.tex}    % includes all 30 \begin{table} environments
```

Or use individual snippets from `latex/figures.tex` and `latex/tables.tex`.

---

## Requirements

```
matplotlib
networkx
numpy
opencv-python
pandas
scipy
Pillow
openpyxl
tabulate
```

All are standard scientific Python packages.
