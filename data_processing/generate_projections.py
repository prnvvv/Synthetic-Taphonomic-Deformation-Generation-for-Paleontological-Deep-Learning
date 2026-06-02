#!/usr/bin/env python3
"""
Person 2 - TDE Project: 3D Tomographic Slice Stack -> 2D Projection Images
Generates 30 projection views per specimen for Person 3 (synthetic deformation generation)
"""

import os
import csv
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(r"D:\proj\fossils\Synthetic-Taphonomic-Deformation-Generation-for-Paleontological-Deep-Learning")
DATASET_ROOT = PROJECT_ROOT / "dataset"
RAW = DATASET_ROOT / "raw"
PROJECTIONS_OUT = DATASET_ROOT / "projections"

OUTPUT_SIZE = (224, 224)

SPECIMENS = [
    ("araripesaurus",        "BSPG-1982-I-90"),
    ("archaeopteryx_london", "BMNH-37001"),
    ("halszkaraptor",        "MPC-D-102-109"),
    ("hesperornis",          "YPM-1206-A"),
    ("ichthyornis",          "YPM-1460"),
    ("ichthyornis",          "YPM-1775"),
    ("titanosaur_embryo",    "MCF-PVPH-874"),
    ("tropeognathus",        "BSPG-1987-I-46"),
]

ROTATION_ANGLES = list(range(0, 360, 15))  # 0, 15, 30 ... 345

# =========================================================
# UTILITY: Normalize to 8-bit
# =========================================================

def to_uint8(arr):
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn) * 255.0
    return arr.astype(np.uint8)

# =========================================================
# UTILITY: Save projection as 224x224 PNG
# =========================================================

def save_projection(arr2d, out_path):
    img = to_uint8(arr2d)
    img_resized = cv2.resize(img, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), img_resized)

# =========================================================
# UTILITY: Locate org_slices folder (handles hesperornis subfolder)
# =========================================================

def find_org_slices(id_dir):
    # Primary location
    primary = id_dir / "org_slices"
    if primary.exists():
        jp2_files = sorted(primary.glob("*.jp2"))
        if jp2_files:
            return primary, jp2_files

    # Search one level deeper (hesperornis special case)
    for sub in id_dir.iterdir():
        if sub.is_dir():
            candidate = sub / "org_slices"
            if candidate.exists():
                jp2_files = sorted(candidate.glob("*.jp2"))
                if jp2_files:
                    return candidate, jp2_files
            # Also check if sub itself contains jp2 files directly
            jp2_files = sorted(sub.glob("*.jp2"))
            if jp2_files:
                return sub, jp2_files

    # Last resort: search recursively
    all_jp2 = sorted(id_dir.rglob("*.jp2"))
    if all_jp2:
        return all_jp2[0].parent, all_jp2

    return None, []

# =========================================================
# STEP 1: Load slice stack into 3D numpy array
# =========================================================

def load_stack(jp2_files, max_slices=None):
    """Load .jp2 slices into a 3D array [Z, H, W]."""
    files = jp2_files[:max_slices] if max_slices else jp2_files

    # Read first slice to get dimensions
    first = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise ValueError(f"Cannot read first slice: {files[0]}")

    H, W = first.shape
    Z = len(files)

    stack = np.zeros((Z, H, W), dtype=np.float32)
    stack[0] = first.astype(np.float32)

    print(f"         Loading {Z} slices [{H}x{W}]...", end="", flush=True)
    for i, f in enumerate(files[1:], 1):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            stack[i] = img.astype(np.float32)
        if (i + 1) % 500 == 0:
            print(f" {i+1}", end="", flush=True)
    print(" done.")
    return stack  # shape: [Z, H, W]

# =========================================================
# STEP 2: Max Intensity Projections - 6 orthographic views
# =========================================================

def generate_orthographic(stack):
    """
    stack shape: [Z, H, W]
      Z-axis -> dorsal/ventral  (project along axis=0)
      Y-axis -> anterior/posterior (project along axis=1, i.e. H)
      X-axis -> lateral          (project along axis=2, i.e. W)
    """
    views = {}

    # Dorsal: top-down, MIP along Z
    views["dorsal"]         = np.max(stack, axis=0)          # [H, W]
    # Ventral: bottom-up = flip Z then MIP (equivalent to flipping result)
    views["ventral"]        = np.max(stack[::-1], axis=0)    # same MIP but Z flipped first
    # Anterior: front view, MIP along H (axis=1)
    views["anterior"]       = np.max(stack, axis=1)          # [Z, W]
    # Posterior: back view, flip H
    views["posterior"]      = np.max(stack[:, ::-1, :], axis=1)
    # Lateral left: MIP along W (axis=2)
    views["lateral_left"]   = np.max(stack, axis=2)          # [Z, H]
    # Lateral right: flip W
    views["lateral_right"]  = np.max(stack[:, :, ::-1], axis=2)

    return views

# =========================================================
# STEP 3: Rotational views (Y-axis rotation, 15-degree steps)
# =========================================================

def rotate_volume_y(stack, angle_deg):
    """
    Rotate the volume around the Y-axis (vertical axis) by angle_deg.
    stack: [Z, H, W]  -> treated as [Z, H, W] with Y=H fixed.
    We rotate in the Z-W plane (equivalent to yaw).
    """
    Z, H, W = stack.shape
    cx, cz = W / 2.0, Z / 2.0
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Build output array [Z_new, H, W_new] - keep same dims
    out = np.zeros_like(stack)

    # For each output voxel, compute source via inverse rotation
    z_idx, w_idx = np.meshgrid(np.arange(Z), np.arange(W), indexing='ij')
    dz = z_idx - cz
    dw = w_idx - cx

    src_dz = cos_t * dz + sin_t * dw
    src_dw = -sin_t * dz + cos_t * dw

    src_z = np.round(src_dz + cz).astype(int)
    src_w = np.round(src_dw + cx).astype(int)

    valid = (src_z >= 0) & (src_z < Z) & (src_w >= 0) & (src_w < W)

    for h in range(H):
        slice_out = np.zeros((Z, W), dtype=np.float32)
        vz, vw = z_idx[valid], w_idx[valid]
        sz, sw = src_z[valid], src_w[valid]
        slice_out[vz, vw] = stack[sz, h, sw]
        out[:, h, :] = slice_out

    return out

def generate_rotational(stack):
    """Generate 24 rotational MIP views around Y-axis."""
    views = {}
    for angle in ROTATION_ANGLES:
        print(f"            rot_{angle:03d}", end=" ", flush=True)
        if angle == 0:
            rot = stack
        else:
            rot = rotate_volume_y(stack, angle)
        # MIP along X (axis=2) after rotation -> front-facing projection
        views[f"rot_{angle:03d}"] = np.max(rot, axis=2)
    print()
    return views

# =========================================================
# STEP 4: Save all 30 views
# =========================================================

def save_all_views(ortho_views, rot_views, out_dir):
    saved = 0
    for name, arr in ortho_views.items():
        save_projection(arr, out_dir / f"{name}.png")
        saved += 1
    for name, arr in rot_views.items():
        save_projection(arr, out_dir / f"{name}.png")
        saved += 1
    return saved

# =========================================================
# MAIN PROCESSING LOOP
# =========================================================

def main():
    print("=" * 70)
    print("PERSON 2 - TDE PROJECT: 3D -> 2D PROJECTION GENERATOR")
    print("=" * 70)
    print(f"Output directory: {PROJECTIONS_OUT}")
    print("=" * 70)

    os.makedirs(PROJECTIONS_OUT, exist_ok=True)

    log_rows = []
    verified = []
    failed = []

    for specimen_name, spec_id in SPECIMENS:
        label = f"{specimen_name}/{spec_id}"
        print(f"\n[SPECIMEN] {label}")

        id_dir = RAW / specimen_name / spec_id
        if not id_dir.exists():
            print(f"   [SKIP] Directory not found: {id_dir}")
            log_rows.append([specimen_name, spec_id, 0, 0, "MISSING_DIR"])
            failed.append(label)
            continue

        slices_dir, jp2_files = find_org_slices(id_dir)
        if not jp2_files:
            print(f"   [SKIP] No .jp2 files found under {id_dir}")
            log_rows.append([specimen_name, spec_id, 0, 0, "NO_SLICES"])
            failed.append(label)
            continue

        print(f"   [FOUND] {len(jp2_files)} slices in: {slices_dir}")

        out_dir = PROJECTIONS_OUT / specimen_name
        os.makedirs(out_dir, exist_ok=True)

        try:
            print("   [LOAD] Loading stack...")
            stack = load_stack(jp2_files)

            print("   [PROJ] Generating 6 orthographic views...")
            ortho = generate_orthographic(stack)

            print("   [PROJ] Generating 24 rotational views (Y-axis, 15-deg steps)...")
            rots = generate_rotational(stack)

            print("   [SAVE] Writing PNGs...")
            saved = save_all_views(ortho, rots, out_dir)

            print(f"   [OK] Saved {saved} projection PNGs to {out_dir}")
            log_rows.append([specimen_name, spec_id, len(jp2_files), saved, "OK"])
            verified.append(label)

        except Exception as e:
            print(f"   [ERROR] {e}")
            log_rows.append([specimen_name, spec_id, len(jp2_files), 0, f"ERROR: {e}"])
            failed.append(label)

    # =========================================================
    # Write projection_log.csv
    # =========================================================
    log_path = PROJECTIONS_OUT / "projection_log.csv"
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["specimen", "id", "slice_count", "png_count", "status"])
        writer.writerows(log_rows)
    print(f"\n[LOG] projection_log.csv written: {log_path}")

    # =========================================================
    # Write PERSON2_README.txt
    # =========================================================
    readme_path = PROJECTIONS_OUT / "PERSON2_README.txt"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("TDE PROJECT - PERSON 2 OUTPUT: 2D PROJECTION IMAGES\n")
        f.write("=" * 60 + "\n\n")
        f.write("GENERATED BY: Person 2 (3D->2D Projection Generator)\n")
        f.write(f"GENERATED ON: {datetime.now().isoformat()}\n\n")
        f.write("OUTPUT FORMAT\n")
        f.write("-" * 40 + "\n")
        f.write("Each specimen folder contains 30 PNG files:\n\n")
        f.write("ORTHOGRAPHIC VIEWS (6 files):\n")
        f.write("  dorsal.png          - Top-down MIP (Z-axis)\n")
        f.write("  ventral.png         - Bottom-up MIP (Z-axis flipped)\n")
        f.write("  lateral_left.png    - Left side MIP (X-axis)\n")
        f.write("  lateral_right.png   - Right side MIP (X-axis flipped)\n")
        f.write("  anterior.png        - Front view MIP (Y-axis)\n")
        f.write("  posterior.png       - Back view MIP (Y-axis flipped)\n\n")
        f.write("ROTATIONAL VIEWS (24 files, 15-degree increments):\n")
        f.write("  rot_000.png ... rot_345.png\n")
        f.write("  Rotation around Y-axis (vertical), MIP along X-axis\n\n")
        f.write("IMAGE SPECIFICATIONS\n")
        f.write("-" * 40 + "\n")
        f.write("  Format:     PNG (lossless)\n")
        f.write("  Size:       224 x 224 pixels\n")
        f.write("  Depth:      8-bit grayscale\n")
        f.write("  Projection: Max Intensity Projection (MIP)\n\n")
        f.write("FOLDER STRUCTURE\n")
        f.write("-" * 40 + "\n")
        f.write("  dataset/projections/\n")
        f.write("    [specimen_name]/\n")
        f.write("      dorsal.png\n")
        f.write("      ventral.png\n")
        f.write("      lateral_left.png\n")
        f.write("      lateral_right.png\n")
        f.write("      anterior.png\n")
        f.write("      posterior.png\n")
        f.write("      rot_000.png\n")
        f.write("      ... (rot_015 to rot_345)\n\n")
        f.write("INSTRUCTIONS FOR PERSON 3\n")
        f.write("-" * 40 + "\n")
        f.write("1. Use these 224x224 PNG images as input for synthetic\n")
        f.write("   taphonomic deformation generation.\n")
        f.write("2. All images are normalized to 8-bit (0-255 range).\n")
        f.write("3. Rotational views provide full 360-degree coverage\n")
        f.write("   in 15-degree increments for orientation diversity.\n")
        f.write("4. Check projection_log.csv for slice counts and status.\n")
        f.write("5. If a specimen shows status != OK, re-run after fixing\n")
        f.write("   the raw data source.\n")
        f.write("=" * 60 + "\n")
    print(f"[README] PERSON2_README.txt written.")

    # =========================================================
    # Write VERIFICATION.txt
    # =========================================================
    verify_path = PROJECTIONS_OUT / "VERIFICATION.txt"
    with open(verify_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("TDE PROJECT - PERSON 2 VERIFICATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Completed: {datetime.now().isoformat()}\n\n")
        f.write(f"SUCCESSFULLY PROCESSED ({len(verified)}):\n")
        for v in verified:
            f.write(f"  [OK] {v}\n")
        f.write(f"\nFAILED / SKIPPED ({len(failed)}):\n")
        for fail in failed:
            f.write(f"  [FAIL] {fail}\n")
        f.write("\nVIEWS PER SPECIMEN: 30\n")
        f.write("  - 6 orthographic (dorsal, ventral, lateral x2, anterior, posterior)\n")
        f.write("  - 24 rotational (0 to 345 degrees, 15-degree steps)\n")
        f.write("\nOUTPUT SPECS: 224x224 px, 8-bit grayscale PNG, MIP\n")
        f.write("=" * 60 + "\n")
        f.write("STATUS: READY FOR PERSON 3 (SYNTHETIC DEFORMATION)\n")
        f.write("=" * 60 + "\n")
    print(f"[VERIFY] VERIFICATION.txt written.")

    # =========================================================
    # Final summary
    # =========================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Specimens processed successfully : {len(verified)}")
    print(f"Specimens failed / skipped       : {len(failed)}")
    print(f"Projections per specimen         : 30 (6 ortho + 24 rotational)")
    print(f"Output folder                    : {PROJECTIONS_OUT}")
    print("=" * 70)
    print("[DONE] Person 2 projection generation complete.")
    print("[INFO] Person 3 can now start synthetic deformation generation.")
    print("=" * 70)

if __name__ == "__main__":
    main()