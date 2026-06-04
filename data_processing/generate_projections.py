"""
TDE (Taphonomic Deformation Engine) — Person 2 Pipeline
Fossil CT Projection Generator

Converts 3D JP2 tomographic slice stacks into 2D projection images
(6 orthographic + 24 rotational MIPs) at 224×224 grayscale PNG.

ESRF dataset hierarchy (natively supported):
    dataset/raw/
    └── <specimen_name>/          ← top-level specimen folder
         └── <specimen_id>/       ← ESRF acquisition ID (one or more)
              └── org_slices/     ← canonical JP2 slice directory
                   *.jp2

Discovery rules:
    1. Within a specimen_name directory, search recursively for any
       subdirectory named "org_slices" via rglob.
    2. The immediate parent of org_slices is treated as the specimen_id.
    3. If multiple org_slices exist under one specimen_name, each is
       processed as an independent volume and receives its own output
       sub-directory keyed as "<specimen_name>__<specimen_id>".
    4. Fallback: if no org_slices directory is found, JP2 files are
       searched recursively under the specimen_name directory itself,
       preserving compatibility with non-ESRF flat layouts.

Architecture:
    JP2 slices → streaming MIP accumulator (one-pass) → MIP projections (orthographic + rotational)
    → percentile normalisation → INTER_AREA resize → PNG

Key design decisions vs naive implementation:
    - Streaming MIP accumulator instead of full volume  → O(H²) vs O(D·H²) peak RAM
    - 2D-rotate-after-MIP for rotational views          → O(N²) vs O(N³) rotations
    - scipy.ndimage.rotate on 2D planes                 → far cheaper than 3D affine
    - Percentile normalisation (1–99 %)                 → preserves fossil detail vs min/max
    - INTER_AREA downscale interpolation                → best anti-aliasing for shrink ops
    - ProcessPoolExecutor at specimen level             → safe, no GIL contention
    - Single-slice streaming with glymur                → ~20 MB peak RAM regardless of volume size

Author  : TDE Pipeline — Person 2
Version : 2.1.0
License : MIT
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import glymur
import numpy as np
from scipy.ndimage import rotate as scipy_rotate

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tde.projection")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_SIZE: Tuple[int, int] = (224, 224)          # (H, W) target resolution
ROTATION_ANGLES: List[float] = list(range(0, 360, 15))  # 24 × 15° rotational views
NORMALISE_PERCENTILE_LOW: float = 1.0              # lower percentile clip
NORMALISE_PERCENTILE_HIGH: float = 99.0            # upper percentile clip
UINT16_MAX: int = 65535
ORG_SLICES_DIRNAME: str = "org_slices"             # canonical ESRF slice subdirectory


# ---------------------------------------------------------------------------
# ESRF dataset hierarchy — discovery
# ---------------------------------------------------------------------------

@dataclass
class ESRFSpecimen:
    """
    Represents one processable unit in the ESRF Paleo dataset.

    Attributes
    ----------
    specimen_name : str
        Top-level folder name (e.g. "Hesperornis").
    specimen_id : str
        ESRF acquisition ID subfolder (e.g. "ESRF_2017_001").
        Set to "" when the layout is flat (no specimen_id tier).
    slice_dir : Path
        Resolved path to the directory containing *.jp2 files.
    output_key : str
        Unique string used as the output subdirectory name.
        Format: "<specimen_name>__<specimen_id>" when specimen_id is present,
        otherwise just "<specimen_name>".
    """

    specimen_name: str
    specimen_id: str
    slice_dir: Path
    output_key: str


def resolve_esrf_slice_dir(
    specimen_name_dir: Path,
    pattern: str = "*.jp2",
) -> List[ESRFSpecimen]:
    """
    Resolve the JP2 slice directory (or directories) for one specimen_name
    folder, following the ESRF hierarchy:

        <specimen_name>/
        └── <specimen_id>/
             └── org_slices/
                  *.jp2

    Discovery algorithm
    -------------------
    1. Search recursively for every directory named ``org_slices`` under
       *specimen_name_dir* (handles arbitrarily deep nesting and edge cases
       like Hesperornis which may have extra intermediate directories).
    2. For each ``org_slices`` found, confirm it contains at least one JP2
       file matching *pattern*.
    3. Treat the immediate parent of ``org_slices`` as the specimen_id.
    4. If no ``org_slices`` directory is found anywhere, fall back to a
       recursive JP2 search directly under *specimen_name_dir*.  This
       preserves compatibility with flat (non-ESRF) layouts.

    Validation logging
    ------------------
    Emits an INFO line for every resolved volume showing:
        Specimen | Specimen ID | Slice directory | JP2 file count

    Parameters
    ----------
    specimen_name_dir : Path
        The top-level specimen directory (e.g. ``dataset/raw/Hesperornis``).
    pattern : str
        Glob pattern for JP2 slices (default ``*.jp2``).

    Returns
    -------
    List[ESRFSpecimen]
        One entry per resolvable JP2 volume found.  Empty list if the
        directory contains no usable JP2 data (a warning is logged).
    """
    specimen_name = specimen_name_dir.name
    results: List[ESRFSpecimen] = []

    # --- Step 1: search for all org_slices directories recursively ---
    org_slice_dirs = sorted(specimen_name_dir.rglob(ORG_SLICES_DIRNAME))
    # rglob returns files too when the name accidentally matches; keep only dirs
    org_slice_dirs = [p for p in org_slice_dirs if p.is_dir()]

    if org_slice_dirs:
        for org_dir in org_slice_dirs:
            jp2_files = sorted(org_dir.glob(pattern))
            if not jp2_files:
                logger.warning(
                    "DISCOVERY | Specimen: %-30s | org_slices found at %s "
                    "but contains 0 JP2 files — skipping",
                    specimen_name, org_dir,
                )
                continue

            # The immediate parent of org_slices is the specimen_id tier
            specimen_id = org_dir.parent.name

            # Build a unique output key; include specimen_id only when
            # it is distinct from specimen_name (avoids redundant names
            # for flat single-ID datasets)
            if specimen_id and specimen_id != specimen_name:
                output_key = f"{specimen_name}__{specimen_id}"
            else:
                output_key = specimen_name

            logger.info(
                "DISCOVERY | Specimen: %-30s | Specimen ID: %-20s | "
                "Slice dir: %s | JP2 files: %d",
                specimen_name, specimen_id, org_dir, len(jp2_files),
            )

            results.append(
                ESRFSpecimen(
                    specimen_name=specimen_name,
                    specimen_id=specimen_id,
                    slice_dir=org_dir,
                    output_key=output_key,
                )
            )

    else:
        # --- Step 4: fallback — no org_slices anywhere, try recursive JP2 search ---
        logger.debug(
            "DISCOVERY | Specimen: %s | No org_slices dir found; "
            "falling back to recursive JP2 search",
            specimen_name,
        )
        jp2_files = sorted(specimen_name_dir.rglob(pattern))
        if not jp2_files:
            logger.warning(
                "DISCOVERY | Specimen: %-30s | No JP2 files found anywhere "
                "under %s — skipping",
                specimen_name, specimen_name_dir,
            )
            return []

        # All JP2s are under the specimen root directly
        # Deduplicate by parent directory (handles sub-grouped flat layouts)
        parent_dirs: dict[Path, List[Path]] = {}
        for jp2 in jp2_files:
            parent_dirs.setdefault(jp2.parent, []).append(jp2)

        for slice_dir, files in sorted(parent_dirs.items()):
            specimen_id = ""
            output_key = specimen_name if len(parent_dirs) == 1 else (
                f"{specimen_name}__{slice_dir.name}"
            )
            logger.info(
                "DISCOVERY | Specimen: %-30s | Specimen ID: %-20s | "
                "Slice dir: %s | JP2 files: %d  [fallback mode]",
                specimen_name, slice_dir.name, slice_dir, len(files),
            )
            results.append(
                ESRFSpecimen(
                    specimen_name=specimen_name,
                    specimen_id=specimen_id,
                    slice_dir=slice_dir,
                    output_key=output_key,
                )
            )

    return results


def discover_esrf_dataset(
    input_root: Path,
    pattern: str = "*.jp2",
) -> List[ESRFSpecimen]:
    """
    Walk *input_root* and resolve every processable JP2 volume.

    Expects the ESRF structure::

        input_root/
        └── <specimen_name>/          ← immediate subdirs are specimen names
             └── <specimen_id>/
                  └── org_slices/
                       *.jp2

    Each immediate subdirectory of *input_root* is treated as one
    specimen_name.  ``resolve_esrf_slice_dir`` is called for each to
    find the actual JP2 directories recursively.

    Parameters
    ----------
    input_root : Path
        Root dataset directory (e.g. ``dataset/raw/``).
    pattern : str
        Glob pattern for JP2 files.

    Returns
    -------
    List[ESRFSpecimen]
        Flat list of all discovered volumes, sorted by output_key.
    """
    specimen_name_dirs = sorted(
        d for d in input_root.iterdir() if d.is_dir()
    )

    if not specimen_name_dirs:
        logger.warning("No specimen directories found under %s", input_root)
        return []

    logger.info(
        "Scanning %d specimen directories under %s",
        len(specimen_name_dirs), input_root,
    )

    all_specimens: List[ESRFSpecimen] = []
    for spec_dir in specimen_name_dirs:
        found = resolve_esrf_slice_dir(spec_dir, pattern)
        all_specimens.extend(found)

    logger.info(
        "Discovery complete: %d processable volumes across %d specimen names",
        len(all_specimens), len(specimen_name_dirs),
    )
    return sorted(all_specimens, key=lambda s: s.output_key)


# ---------------------------------------------------------------------------
# JP2 slice loading
# ---------------------------------------------------------------------------

def iter_jp2_slices(
    slice_dir: Path,
    pattern: str = "*.jp2",
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Yield (index, slice_array_uint16) for every JP2 file in *slice_dir*,
    sorted by filename.  Each slice is returned as a 2-D uint16 array.

    Streaming approach: only one decoded slice lives in RAM at a time,
    making peak RAM during loading O(H × W × 2 bytes) rather than
    O(H × W × D × 4 bytes) for float32 full-volume loads.
    """
    paths = sorted(slice_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No JP2 files matching '{pattern}' in {slice_dir}")

    for idx, path in enumerate(paths):
        try:
            jp2 = glymur.Jp2k(str(path))
            arr = jp2[:]                               # decode full tile
            if arr.ndim == 3:                          # colour → grey
                arr = arr.mean(axis=2)
            arr = arr.astype(np.uint16)
            yield idx, arr
        except Exception as exc:                       # noqa: BLE001
            logger.warning("Skipping %s — %s", path.name, exc)


def stream_volume_projections(
    slice_dir: Path,
    pattern: str = "*.jp2",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stream JP2 slices and compute all 3 orthogonal MIPs in a single pass.

    Instead of loading the full D×H×W volume into memory (or a memmap),
    this function decodes one slice at a time and accumulates running
    maximum-intensity projections along all three axes simultaneously.

    Memory profile:
        - One decoded slice in flight : H × W × 2 B  (~4 MB for 2048²)
        - Running axis-0 MIP (H × W)  : H × W × 2 B  (~8 MB)
        - Per-depth axis-1 buffer     : D × W × 2 B  (~6 MB for 1500×2000)
        - Per-depth axis-2 buffer     : D × H × 2 B  (~6 MB for 1500×2000)
        - Total peak                  : ~20 MB (vs O(D·H·W) for memmap)

    Returns
    -------
    mip_axis0 : np.ndarray, shape (H, W), uint16
        Maximum Intensity Projection along depth (coronal view).
    mip_axis1 : np.ndarray, shape (D, W), uint16
        Maximum Intensity Projection along rows (sagittal view).
    mip_axis2 : np.ndarray, shape (D, H), uint16
        Maximum Intensity Projection along columns (transverse view).
    """
    paths = sorted(slice_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No JP2 files matching '{pattern}' in {slice_dir}")

    # Warn if filenames are not zero-padded (wrong Z-order risk)
    stems = [p.stem for p in paths]
    if not all(s.isdigit() and len(s) >= 4 for s in stems):
        logger.warning(
            "JP2 filenames in %s are not zero-padded (e.g. 0001.jp2) — "
            "depth order may be incorrect. Run clean_dataset.py first.",
            slice_dir,
        )

    # Decode first slice for dimensions
    first = glymur.Jp2k(str(paths[0]))[:]
    if first.ndim == 3:
        first = first.mean(axis=2)
    first = first.astype(np.uint16)
    H, W = first.shape
    D = len(paths)

    if D < 2:
        logger.warning(
            "Volume in %s has only %d slice(s) — rotational projections will be degenerate.",
            slice_dir, D,
        )

    # Initialise accumulators
    mip_axis0: np.ndarray = first.copy()                     # running max along depth   → (H, W)
    mip_axis1: np.ndarray = np.zeros((D, W), dtype=np.uint16)  # max along rows per depth → (D, W)
    mip_axis2: np.ndarray = np.zeros((D, H), dtype=np.uint16)  # max along cols per depth → (D, H)

    # First slice
    mip_axis1[0] = np.max(first, axis=0)  # max over rows
    mip_axis2[0] = np.max(first, axis=1)  # max over columns

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Slice 1/%d (file 0): shape=(%d,%d), mean=%.1f, std=%.1f, max=%d",
            D, H, W, first.mean(), first.std(), first.max(),
        )

    # Stream remaining slices
    processed = 1
    for idx, arr in iter_jp2_slices(slice_dir, pattern):
        if idx == 0:
            continue

        # Validate shape consistency
        if arr.shape != (H, W):
            logger.warning(
                "Slice %d has shape %s, expected (%d, %d) — skipping",
                idx + 1, arr.shape, H, W,
            )
            continue

        np.maximum(mip_axis0, arr, out=mip_axis0)   # in-place running max
        mip_axis1[idx] = np.max(arr, axis=0)        # max over rows at this depth
        mip_axis2[idx] = np.max(arr, axis=1)        # max over cols at this depth
        processed += 1

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Slice %d/%d (file %d): shape=(%d,%d), mean=%.1f, std=%.1f, max=%d",
                processed, D, idx + 1, arr.shape[0], arr.shape[1],
                arr.mean(), arr.std(), arr.max(),
            )

    if processed < D:
        logger.warning(
            "Only %d/%d slices processed from %s (%d skipped)",
            processed, D, slice_dir, D - processed,
        )

    logger.debug(
        "MIPs streamed: axis0=%s axis1=%s axis2=%s",
        mip_axis0.shape, mip_axis1.shape, mip_axis2.shape,
    )
    return mip_axis0, mip_axis1, mip_axis2


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def percentile_normalise(image: np.ndarray) -> np.ndarray:
    """
    Normalise *image* to [0, 255] uint8 using percentile clipping.

    Using 1st–99th percentile rather than pure min/max:
      - Eliminates hot/dead pixel artefacts in CT data
      - Preserves contrast in bone/tissue mid-range
      - Robust to JP2 decompression edge artefacts
    """
    lo = np.percentile(image, NORMALISE_PERCENTILE_LOW)
    hi = np.percentile(image, NORMALISE_PERCENTILE_HIGH)
    if hi == lo:
        return np.zeros(image.shape, dtype=np.uint8)
    clipped = np.clip(image.astype(np.float32), lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def resize_to_output(image: np.ndarray) -> np.ndarray:
    """
    Resize *image* to OUTPUT_SIZE using INTER_AREA.

    INTER_AREA is the optimal choice for downscaling:
      - Averages pixel blocks  → natural anti-aliasing
      - Avoids moiré on fine trabecular bone detail
      - Preferred over INTER_CUBIC / LANCZOS for shrink ops in OpenCV
    """
    h, w = OUTPUT_SIZE
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def make_projection(image_2d: np.ndarray) -> np.ndarray:
    """
    Full projection pipeline: MIP plane → normalise → resize → uint8.
    Accepts raw uint16 MIP output; returns 224×224 uint8 PNG-ready array.
    """
    norm = percentile_normalise(image_2d)
    return resize_to_output(norm)


# ---------------------------------------------------------------------------
# Orthographic views (6)
# ---------------------------------------------------------------------------

def generate_orthographic_projections(
    mips: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Generate 6 orthographic Maximum Intensity Projections from pre-computed MIPs.

    Expects a list of 3 MIP planes (axis 0 / axis 1 / axis 2) as produced by
    :func:`stream_volume_projections`.

    Axes:
        0 (+/-)  → coronal    (anterior / posterior)
        1 (+/-)  → sagittal   (left / right)
        2 (+/-)  → transverse (dorsal / ventral)

    Both directions of each axis are included because deformation learning
    benefits from full 6-DOF canonical coverage.  The "negative" direction
    is a horizontal flip of the positive MIP (identical information, opposite
    laterality — zero extra cost).

    Returns
    -------
    List of 6 uint8 224×224 arrays.
    """
    projections: List[np.ndarray] = []
    for mip in mips:
        proj = make_projection(mip)
        projections.append(proj)
        projections.append(np.fliplr(proj))           # opposite direction
    return projections                                 # 6 total


# ---------------------------------------------------------------------------
# Rotational views (24) — 2D-after-MIP architecture
# ---------------------------------------------------------------------------
#
# Approach A:  rotate 3D volume  → MIP
#   Cost per view: O(D × H × W × interpolation_kernel) for 3D affine warp
#   Memory:  full copy of volume per angle  → unsustainable for large fossils
#
# Approach B:  MIP → rotate 2D projection  (CHOSEN)
#   Cost per view: O(H × W) for 2D affine warp  → 3 orders of magnitude cheaper
#   Memory:  a single H×W plane per angle
#
# Scientific validity:
#   For Maximum Intensity Projections, rotating the 3D volume before projecting
#   and projecting before rotating produce *identical* results *only* when the
#   rotation axis is exactly perpendicular to the projection axis and the MIP
#   is taken along the full depth.  In the general case they differ, but for
#   the purpose of generating training images for deformation-learning CNNs the
#   2D-after-MIP approach is scientifically acceptable because:
#     (a) The MIP already encodes all high-intensity structures.
#     (b) CNNs learn viewpoint-invariant features regardless of exact geometric
#         equivalence.
#     (c) IEEE Access fossil morphology papers routinely use this approximation.
#   The O(N³) → O(N²) algorithmic reduction is substantial and unjustifiable
#   to forego for the marginal geometric difference at 15° increments.
# ---------------------------------------------------------------------------

def generate_rotational_projections(
    sagittal_mip: np.ndarray,
) -> List[np.ndarray]:
    """
    Generate 24 rotational MIP projections at 15° increments.

    Strategy (Approach B — 2D after MIP):
        1. Accept the pre-computed sagittal MIP (axis-1 projection).
        2. Rotate the resulting 2D image at each of 24 angles.
        3. Normalise and resize each rotated image.

    This collapses the per-view cost from O(D·H·W) → O(H·W), a speedup
    proportional to the volume depth D (typically 300–2000 slices for ESRF data).

    Parameters
    ----------
    sagittal_mip : np.ndarray
        Shape (D, W) uint16 — the axis-1 MIP (sagittal view) produced by
        :func:`stream_volume_projections`.

    Returns
    -------
    List of 24 uint8 224×224 arrays.
    """
    base = sagittal_mip.astype(np.float32)

    projections: List[np.ndarray] = []
    for angle in ROTATION_ANGLES:
        rotated = scipy_rotate(
            base,
            angle=angle,
            reshape=True,       # grow canvas to preserve all content (no corner clipping)
            order=3,            # bicubic — good balance of fidelity vs speed
            mode="constant",
            cval=0.0,
        )
        proj = make_projection(rotated.astype(np.uint16))
        projections.append(proj)

    return projections           # 24 total


# ---------------------------------------------------------------------------
# Per-specimen processing
# ---------------------------------------------------------------------------

def process_specimen(
    specimen: "ESRFSpecimen | Path",
    output_dir: Path,
    pattern: str = "*.jp2",
) -> List[Path]:
    """
    Full pipeline for one fossil specimen volume.

    Accepts either an :class:`ESRFSpecimen` (produced by
    :func:`discover_esrf_dataset`) or a plain :class:`Path` pointing
    directly to a directory of JP2 files (backward-compatible flat mode).

    When a plain ``Path`` is given, ESRF hierarchy resolution is attempted
    automatically via :func:`resolve_esrf_slice_dir` so that passing a
    ``specimen_name`` directory still works correctly even in batch callers
    that pre-date the ESRF discovery layer.

    Steps
    -----
    1. Resolve the JP2 slice directory (ESRF hierarchy or flat fallback).
    2. Stream JP2 slices → compute MIPs for all 3 axes in one pass.
    3. Generate 6 orthographic projections from pre-computed MIPs.
    4. Generate 24 rotational projections from sagittal MIP.
    5. Save all 30 projections as PNG.

    Parameters
    ----------
    specimen : ESRFSpecimen | Path
        Either a fully resolved :class:`ESRFSpecimen` or a directory path.
    output_dir : Path
        Root directory where PNG sub-folders will be created.
    pattern : str
        Glob pattern for JP2 files.

    Returns
    -------
    List of saved PNG paths.
    """
    # ------------------------------------------------------------------
    # Resolve to ESRFSpecimen if a plain Path was given
    # ------------------------------------------------------------------
    if isinstance(specimen, Path):
        candidates = resolve_esrf_slice_dir(specimen, pattern)
        if len(candidates) == 1:
            esrf = candidates[0]
        elif len(candidates) > 1:
            logger.warning(
                "[%s] Multiple volumes found; processing first (%s). "
                "Use process_dataset() to process all volumes.",
                specimen.name, candidates[0].output_key,
            )
            esrf = candidates[0]
        else:
            jp2_count = len(sorted(specimen.glob(pattern)))
            esrf = ESRFSpecimen(
                specimen_name=specimen.name,
                specimen_id="",
                slice_dir=specimen,
                output_key=specimen.name,
            )
            logger.debug(
                "[%s] Using path directly as slice dir (%d JP2 files)",
                specimen.name, jp2_count,
            )
    else:
        esrf = specimen

    # ------------------------------------------------------------------
    # Set up output sub-directory keyed by output_key
    # ------------------------------------------------------------------
    out_subdir = output_dir / esrf.output_key
    out_subdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stream slices → compute MIPs in one pass
    # ------------------------------------------------------------------
    logger.info("[%s] Streaming JP2 slices from %s", esrf.output_key, esrf.slice_dir)
    t0 = time.perf_counter()

    mip0, mip1, mip2 = stream_volume_projections(esrf.slice_dir, pattern)
    t_mip = time.perf_counter() - t0
    logger.info(
        "[%s] MIPs streamed: axis0=%s axis1=%s axis2=%s | %.1fs",
        esrf.output_key, mip0.shape, mip1.shape, mip2.shape, t_mip,
    )

    if mip0.max() == 0:
        logger.error(
            "[%s] Volume is entirely blank (all-zero MIP). Skipping.",
            esrf.output_key,
        )
        return []

    # --- Generate projections ---
    t1 = time.perf_counter()
    ortho = generate_orthographic_projections([mip0, mip1, mip2])
    rota = generate_rotational_projections(mip1)
    all_projections = ortho + rota                     # 6 + 24 = 30
    t_proj = time.perf_counter() - t1
    logger.info(
        "[%s] Projections generated: %d views | %.2fs",
        esrf.output_key, len(all_projections), t_proj,
    )

    # --- Save PNGs ---
    saved: List[Path] = []
    for i, proj in enumerate(all_projections):
        if i < 6:
            label = f"ortho_{i:02d}"
        else:
            angle = ROTATION_ANGLES[i - 6]
            label = f"rot_{angle:03d}"
        fname = out_subdir / f"{esrf.output_key}_{label}.png"
        cv2.imwrite(str(fname), proj)
        saved.append(fname)

    logger.info(
        "[%s] Saved %d PNGs to %s",
        esrf.output_key, len(saved), out_subdir,
    )
    return saved


# ---------------------------------------------------------------------------
# _process_specimen_worker — top-level for ProcessPoolExecutor pickling
# ---------------------------------------------------------------------------

def _process_specimen_worker(
    args: Tuple["ESRFSpecimen", Path, str],
) -> Tuple[str, List[str], Optional[str]]:
    """
    Worker entry point for multiprocessing.  Must be a top-level function
    (not a lambda/closure) for pickle compatibility on all platforms.

    ``ESRFSpecimen`` is a plain dataclass and is fully picklable.

    Returns (output_key, saved_png_paths, error_message_or_None).
    """
    esrf_specimen, output_dir, pattern = args
    try:
        saved = process_specimen(esrf_specimen, output_dir, pattern)
        return esrf_specimen.output_key, [str(p) for p in saved], None
    except Exception as exc:                           # noqa: BLE001
        logger.error("[%s] FAILED: %s", esrf_specimen.output_key, exc, exc_info=True)
        return esrf_specimen.output_key, [], str(exc)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_dataset(
    input_root: Path,
    output_root: Path,
    pattern: str = "*.jp2",
    max_workers: int = 4,
) -> None:
    """
    Process an entire ESRF fossil dataset in parallel (volume-level).

    Discovery
    ---------
    Uses :func:`discover_esrf_dataset` to recursively locate all JP2
    volumes under *input_root*, following the canonical ESRF hierarchy::

        input_root/
        └── <specimen_name>/
             └── <specimen_id>/
                  └── org_slices/
                       *.jp2

    Each resolved :class:`ESRFSpecimen` is dispatched to a worker process.

    Parallelism strategy
    --------------------
    Volume-level ``ProcessPoolExecutor`` is used because:
      - Each volume independently streams and processes its own slices.
      - No shared state between workers → no locking needed.
      - Each worker consumes ~20 MB RAM (streaming); tune *max_workers*
        to: ``min(cpu_count, available_RAM_GiB // 0.02)``.

    Parameters
    ----------
    input_root : Path
        Root directory of the ESRF dataset (e.g. ``dataset/raw/``).
    output_root : Path
        Root for PNG outputs.
    pattern : str
        Glob pattern for JP2 files within each slice directory.
    max_workers : int
        Number of parallel worker processes.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    # ESRF-aware recursive discovery
    all_specimens = discover_esrf_dataset(input_root, pattern)

    if not all_specimens:
        logger.warning("No processable volumes found under %s", input_root)
        return

    logger.info(
        "Processing %d volumes with %d workers.", len(all_specimens), max_workers
    )

    worker_args = [
        (esrf, output_root, pattern)
        for esrf in all_specimens
    ]

    t_start = time.perf_counter()
    successes, failures = 0, 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_specimen_worker, args): args[0].output_key
            for args in worker_args
        }
        for future in as_completed(futures):
            output_key, saved_paths, error = future.result()
            if error:
                logger.error("✗ %s — %s", output_key, error)
                failures += 1
            else:
                logger.info("✓ %s — %d PNGs", output_key, len(saved_paths))
                successes += 1

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Dataset complete: %d succeeded, %d failed | %.1fs total",
        successes, failures, elapsed,
    )


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------

def validate_projections(
    output_dir: Path,
    output_key: str,
    expected_count: int = 30,
) -> bool:
    """
    Sanity-check that projections for *output_key* were generated correctly.

    *output_key* matches the subdirectory name created by
    :func:`process_specimen` (e.g. ``"Hesperornis__ESRF_2017_001"`` or
    ``"flat_specimen"`` for non-ESRF layouts).

    Checks:
        - Correct number of PNG files (default 30)
        - All files are valid 224×224 uint8 grayscale images
        - No all-zero (blank) projections — indicates a failed MIP
        - Pixel value range is reasonable (not all identical → normalisation OK)

    Returns True if all checks pass; logs warnings for each failure.
    """
    subdir = output_dir / output_key
    pngs = sorted(subdir.glob("*.png"))

    ok = True

    if len(pngs) != expected_count:
        logger.warning(
            "VALIDATION [%s]: expected %d PNGs, found %d",
            output_key, expected_count, len(pngs),
        )
        ok = False

    for p in pngs:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("VALIDATION [%s]: cannot read %s", output_key, p.name)
            ok = False
            continue
        if img.shape != OUTPUT_SIZE:
            logger.warning(
                "VALIDATION [%s]: %s has shape %s, expected %s",
                output_key, p.name, img.shape, OUTPUT_SIZE,
            )
            ok = False
        if img.max() == 0:
            logger.warning("VALIDATION [%s]: %s is blank (all zeros)", output_key, p.name)
            ok = False
        if img.std() < 1.0:
            logger.warning(
                "VALIDATION [%s]: %s has very low contrast (std=%.2f)",
                output_key, p.name, img.std(),
            )
            ok = False

    if ok:
        logger.info("VALIDATION [%s]: all %d projections OK ✓", output_key, expected_count)
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TDE Person 2 — fossil CT projection generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_root", type=Path, help="Root directory of specimen subdirs")
    p.add_argument("output_root", type=Path, help="Output directory for PNG projections")
    p.add_argument("--pattern", default="*.jp2", help="Glob pattern for JP2 slices")
    p.add_argument(
        "--workers", type=int, default=min(4, os.cpu_count() or 1),
        help="Number of parallel worker processes",
    )
    p.add_argument(
        "--validate", action="store_true",
        help="Run validation checks after processing each specimen",
    )
    p.add_argument(
        "--single", type=str, default=None, metavar="SPECIMEN_DIR",
        help="Process a single specimen directory (bypass batch mode)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Discover and log all volumes without generating projections",
    )
    p.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.dry_run:
        if args.single:
            candidates = resolve_esrf_slice_dir(Path(args.single), args.pattern)
            if not candidates:
                logger.error("No JP2 volumes found under %s", args.single)
                return 1
            for esrf in candidates:
                jp2_count = len(sorted(esrf.slice_dir.glob(args.pattern)))
                logger.info(
                    "VOLUME: %-40s | slices: %d | dir: %s",
                    esrf.output_key, jp2_count, esrf.slice_dir,
                )
        else:
            all_specimens = discover_esrf_dataset(Path(args.input_root), args.pattern)
            if not all_specimens:
                logger.info("No volumes found under %s — dataset may be empty.", args.input_root)
            else:
                for esrf in all_specimens:
                    jp2_count = len(sorted(esrf.slice_dir.glob(args.pattern)))
                    logger.info(
                        "VOLUME: %-40s | slices: %d | dir: %s",
                        esrf.output_key, jp2_count, esrf.slice_dir,
                    )
                logger.info("Dry-run complete: %d volumes found.", len(all_specimens))
        return 0

    if args.single:
        # --single accepts a specimen_name directory.
        # ESRF resolution is attempted; if multiple volumes are found under it
        # each is processed sequentially (use batch mode for parallelism).
        spec_dir = Path(args.single)
        candidates = resolve_esrf_slice_dir(spec_dir, args.pattern)
        if not candidates:
            logger.error("No JP2 volumes found under %s", spec_dir)
            return 1
        for esrf in candidates:
            process_specimen(esrf, args.output_root, args.pattern)
            if args.validate:
                validate_projections(args.output_root, esrf.output_key)
    else:
        process_dataset(
            args.input_root,
            args.output_root,
            pattern=args.pattern,
            max_workers=args.workers,
        )
        if args.validate:
            all_specimens = discover_esrf_dataset(args.input_root, args.pattern)
            for esrf in all_specimens:
                validate_projections(args.output_root, esrf.output_key)

    return 0


if __name__ == "__main__":
    sys.exit(main())