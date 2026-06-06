"""
TDE (Taphonomic Deformation Engine) — Person 3 Pipeline
Synthetic Taphonomic Deformation Generator

Applies 4 synthetic deformation types to 2D projection images produced
by Person 2, generating a labelled dataset for deformation classification
and Grad-CAM explainability validation.

Deformations (Section 2.3):
    Compression:         [x', y'] = [[1,0],[0,c]]   · [x,y],  c      ∈ [0.3, 0.9]
    Shearing:            [x', y'] = [[1,0],[k,1]]   · [x,y],  k      ∈ [-0.5, 0.5]
    Stretching:          [x', y'] = [[sx,0],[0,sy]] · [x,y],  sx,sy  ∈ [1.0, 1.5]
    Partial Dissolution: I'(x,y)  = I(x,y) − λ·M(x,y),       λ      ∈ [0.3, 0.8]

Masks (Section 2.4):
    Pixel-wise binary masks mark the deformed region per image for Grad-CAM validation.

Outputs (Section 2.5):
    dataset/synthetic/
    ├── synthetic_images/       ← all deformed PNGs (240 inputs × 4 = 960 images)
    ├── deformation_masks/      ← aligned binary masks (960 masks)
    ├── synthetic_labels.csv    ← per-image metadata + deformation parameters
    ├── synthetic_images.zip    ← zipped synthetic_images/
    └── deformation_masks.zip   ← zipped deformation_masks/

Usage (from project root):
    python data_processing/generate_synthetic.py dataset/projections dataset/synthetic

    Optional flags:
    --seed INT      Random seed for reproducibility (default: 42)
    --debug         Enable DEBUG-level logging

Author  : TDE Pipeline — Person 3
Version : 1.0.0
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tde.synthetic")

# ---------------------------------------------------------------------------
# Parameter ranges  (Section 2.3)
# ---------------------------------------------------------------------------
COMPRESSION_C_RANGE:   Tuple[float, float] = (0.3, 0.9)
SHEARING_K_RANGE:      Tuple[float, float] = (-0.5, 0.5)
STRETCHING_S_RANGE:    Tuple[float, float] = (1.0, 1.5)
DISSOLUTION_LAM_RANGE: Tuple[float, float] = (0.3, 0.8)

DEFORMATION_TYPES: Tuple[str, ...] = (
    "compression", "shearing", "stretching", "dissolution"
)

CSV_FIELDNAMES: List[str] = [
    "image_id",
    "specimen",
    "original_file",
    "deformation_type",
    "param_c",
    "param_k",
    "param_sx",
    "param_sy",
    "param_lambda",
    "synthetic_file",
    "mask_file",
]


# ---------------------------------------------------------------------------
# Deformation implementations  (Section 2.3)
# ---------------------------------------------------------------------------

def apply_compression(
    image: np.ndarray,
    c: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vertical compression: [x', y'] = [[1,0],[0,c]] · [x,y]

    Scales the y-axis by c ∈ [0.3, 0.9] centred on the image midpoint so
    fossil content stays centred rather than drifting to one edge.

    Returns (deformed_image, binary_mask).
    Mask = pixels that differ from the original (difference > 5 DN).
    """
    H, W = image.shape
    cy = H / 2.0
    M = np.float32([
        [1, 0, 0],
        [0, c, cy * (1.0 - c)],
    ])
    deformed = cv2.warpAffine(
        image, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return deformed, _difference_mask(image, deformed)


def apply_shearing(
    image: np.ndarray,
    k: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Horizontal shear: [x', y'] = [[1,0],[k,1]] · [x,y]

    Shears the image along the y-axis by k ∈ [-0.5, 0.5] centred on the
    image midpoint so the fossil stays approximately centred.

    Returns (deformed_image, binary_mask).
    """
    H, W = image.shape
    cx = W / 2.0
    M = np.float32([
        [1, 0,  0],
        [k, 1, -k * cx],
    ])
    deformed = cv2.warpAffine(
        image, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return deformed, _difference_mask(image, deformed)


def apply_stretching(
    image: np.ndarray,
    sx: float,
    sy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Anisotropic stretch: [x', y'] = [[sx,0],[0,sy]] · [x,y]

    Independently scales x by sx and y by sy (both ∈ [1.0, 1.5]) from the
    image centre so content expands outward symmetrically.

    Returns (deformed_image, binary_mask).
    """
    H, W = image.shape
    cx, cy = W / 2.0, H / 2.0
    M = np.float32([
        [sx, 0,  cx * (1.0 - sx)],
        [0,  sy, cy * (1.0 - sy)],
    ])
    deformed = cv2.warpAffine(
        image, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return deformed, _difference_mask(image, deformed)


def apply_dissolution(
    image: np.ndarray,
    lam: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partial dissolution: I'(x,y) = I(x,y) − λ · M(x,y)

    M(x,y) is a binary elliptical region randomly placed inside the image.
    λ ∈ [0.3, 0.8] controls how much intensity is removed in that region.
    The subtraction is proportional to local intensity (relative dissolution)
    and is clipped to [0, 255].

    Returns (deformed_image, binary_mask).
    Mask = M (the dissolution region) scaled to {0, 255}.
    """
    H, W = image.shape
    ellipse_mask = _random_ellipse_mask(H, W, rng)           # float32, values 0/1

    img_f = image.astype(np.float32)
    dissolved = img_f - lam * ellipse_mask * img_f
    deformed = np.clip(dissolved, 0, 255).astype(np.uint8)

    binary_mask = (ellipse_mask > 0).astype(np.uint8) * 255  # 0 or 255
    return deformed, binary_mask


# ---------------------------------------------------------------------------
# Mask helpers  (Section 2.4)
# ---------------------------------------------------------------------------

def _difference_mask(
    original: np.ndarray,
    deformed: np.ndarray,
    threshold: int = 5,
) -> np.ndarray:
    """
    Binary mask (0/255) of pixels that changed between original and deformed.
    Threshold of 5 DN ignores sub-pixel interpolation noise.
    Used for compression, shearing, and stretching.
    """
    diff = cv2.absdiff(original, deformed)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask


def _random_ellipse_mask(
    H: int,
    W: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Random filled ellipse mask within the image canvas.
    Centre is constrained to the middle half of the image so the ellipse
    always overlaps the fossil region rather than landing at a corner.
    Axes span 15–50 % of image dimensions.
    """
    mask = np.zeros((H, W), dtype=np.float32)
    cx    = int(rng.integers(W // 4, 3 * W // 4))
    cy    = int(rng.integers(H // 4, 3 * H // 4))
    rx    = int(rng.integers(W // 8, W // 2))
    ry    = int(rng.integers(H // 8, H // 2))
    angle = int(rng.integers(0, 180))
    cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1.0, thickness=-1)
    return mask


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------

def process_image(
    img_path: Path,
    specimen: str,
    out_images_dir: Path,
    out_masks_dir: Path,
    rng: np.random.Generator,
    image_id_start: int,
) -> List[Dict]:
    """
    Apply all 4 deformations to one input PNG.

    Saves 4 synthetic images and 4 masks to disk.
    Returns a list of 4 CSV row dicts (one per deformation).
    """
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning("Cannot read %s — skipping", img_path.name)
        return []

    stem = img_path.stem
    rows: List[Dict] = []

    for offset, def_type in enumerate(DEFORMATION_TYPES):
        image_id = image_id_start + offset

        if def_type == "compression":
            c        = float(rng.uniform(*COMPRESSION_C_RANGE))
            deformed, mask = apply_compression(image, c)
            params   = dict(param_c=f"{c:.4f}", param_k="", param_sx="", param_sy="", param_lambda="")
            suffix   = f"comp_c{c:.3f}"

        elif def_type == "shearing":
            k        = float(rng.uniform(*SHEARING_K_RANGE))
            deformed, mask = apply_shearing(image, k)
            params   = dict(param_c="", param_k=f"{k:.4f}", param_sx="", param_sy="", param_lambda="")
            suffix   = f"shear_k{k:.3f}"

        elif def_type == "stretching":
            sx       = float(rng.uniform(*STRETCHING_S_RANGE))
            sy       = float(rng.uniform(*STRETCHING_S_RANGE))
            deformed, mask = apply_stretching(image, sx, sy)
            params   = dict(param_c="", param_k="", param_sx=f"{sx:.4f}", param_sy=f"{sy:.4f}", param_lambda="")
            suffix   = f"stretch_sx{sx:.3f}_sy{sy:.3f}"

        else:  # dissolution
            lam      = float(rng.uniform(*DISSOLUTION_LAM_RANGE))
            deformed, mask = apply_dissolution(image, lam, rng)
            params   = dict(param_c="", param_k="", param_sx="", param_sy="", param_lambda=f"{lam:.4f}")
            suffix   = f"diss_lam{lam:.3f}"

        syn_name  = f"{stem}__{suffix}.png"
        mask_name = f"{stem}__{suffix}_mask.png"

        cv2.imwrite(str(out_images_dir / syn_name), deformed)
        cv2.imwrite(str(out_masks_dir  / mask_name), mask)

        rows.append({
            "image_id":        image_id,
            "specimen":        specimen,
            "original_file":   img_path.name,
            "deformation_type": def_type,
            **params,
            "synthetic_file":  syn_name,
            "mask_file":       mask_name,
        })

    return rows


# ---------------------------------------------------------------------------
# Dataset-level processing
# ---------------------------------------------------------------------------

def process_dataset(
    input_root: Path,
    output_root: Path,
    seed: int,
) -> None:
    """
    Walk input_root recursively for PNG files and apply all 4 deformations
    to every image.  Writes synthetic images, masks, CSV, and zip archives
    into output_root.
    """
    rng = np.random.default_rng(seed)

    out_images_dir = output_root / "synthetic_images"
    out_masks_dir  = output_root / "deformation_masks"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_root.rglob("*.png"))
    if not png_files:
        logger.error("No PNG files found under %s", input_root)
        sys.exit(1)

    logger.info("Found %d input images across %d specimen folders",
                len(png_files),
                len({p.parent for p in png_files}))

    all_rows: List[Dict] = []
    image_id = 0

    for img_path in png_files:
        specimen = img_path.parent.name
        logger.info("[%s] %s", specimen, img_path.name)

        rows = process_image(
            img_path, specimen,
            out_images_dir, out_masks_dir,
            rng, image_id,
        )
        all_rows.extend(rows)
        image_id += len(rows)

    # --- synthetic_labels.csv ---
    csv_path = output_root / "synthetic_labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info("CSV written: %s  (%d rows)", csv_path, len(all_rows))

    # --- synthetic_images.zip ---
    _zip_dir(out_images_dir, output_root / "synthetic_images.zip")
    logger.info("Zipped: synthetic_images.zip")

    # --- deformation_masks.zip ---
    _zip_dir(out_masks_dir, output_root / "deformation_masks.zip")
    logger.info("Zipped: deformation_masks.zip")

    logger.info(
        "Complete — %d synthetic images | %d masks | seed=%d",
        len(all_rows), len(all_rows), seed,
    )


def _zip_dir(src_dir: Path, dest_zip: Path) -> None:
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(src_dir.iterdir()):
            if f.is_file():
                zf.write(f, f.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TDE Person 3 — synthetic taphonomic deformation generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "input_root", type=Path,
        help="Root directory containing projection PNGs (dataset/projections)",
    )
    p.add_argument(
        "output_root", type=Path,
        help="Output directory for synthetic dataset (dataset/synthetic)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Input  : %s", args.input_root)
    logger.info("Output : %s", args.output_root)
    logger.info("Seed   : %d", args.seed)

    process_dataset(args.input_root, args.output_root, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
