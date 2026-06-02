#!/usr/bin/env python3
"""
Enhanced Dataset Cleaner - Complete Version
Prepares data for Person 2 (3D->2D projections)
"""

import json
import cv2
from pathlib import Path
from datetime import datetime

# =========================================================
# CONFIGURATION
# =========================================================

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"
RAW = DATASET_ROOT / "raw"

# Voxel sizes (microns) for each specimen - UPDATE BASED ON ESRF METADATA
VOXEL_SIZES = {
    "archaeopteryx_london": 13.0,
    "halszkaraptor": 107.16,
    "tropeognathus": 45.92,
    "araripesaurus": 45.92,
    "ichthyornis": 1.28,
    "titanosaur_embryo": 14.92,
    "hesperornis": 3.5,
}

# =========================================================
# STEP 1: Remove corrupted/zero-size files
# =========================================================

def remove_corrupted_files(folder):
    """Remove files smaller than 1KB (likely corrupted)."""
    removed = []
    for file in folder.glob("*.jp2"):
        if file.stat().st_size < 1024:  # Less than 1KB
            file.unlink()
            removed.append(file.name)
    return removed

# =========================================================
# STEP 2: Rename to sequential numbers
# =========================================================

def rename_sequential(folder):
    """Rename all .jp2 files to 0001.jp2, 0002.jp2, etc."""
    files = sorted(folder.glob("*.jp2"))
    for i, file in enumerate(files, 1):
        new_name = f"{i:04d}.jp2"
        file.rename(file.parent / new_name)
    return len(files)

# =========================================================
# STEP 3: Check for missing slices
# =========================================================

def check_missing_slices(folder):
    """Check for gaps in sequential numbering."""
    files = sorted(folder.glob("*.jp2"))
    if not files:
        return []
    
    numbers = [int(f.stem) for f in files]
    expected = list(range(1, max(numbers) + 1))
    missing = set(expected) - set(numbers)
    return sorted(missing)

# =========================================================
# STEP 4: Get image size
# =========================================================

def get_image_size(folder):
    """Get dimensions of first image."""
    first_image = next(folder.glob("*.jp2"), None)
    if first_image:
        img = cv2.imread(str(first_image))
        return list(img.shape)
    return None

# =========================================================
# STEP 5: Generate metadata.json
# =========================================================

def create_metadata(folder, specimen, spec_id, total_slices, missing_slices, voxel_size):
    """Create metadata.json for the specimen."""
    image_size = get_image_size(folder)
    
    metadata = {
        "specimen": specimen,
        "id": spec_id,
        "total_slices": total_slices,
        "image_size": image_size,
        "voxel_size_um": voxel_size,
        "missing_slices": missing_slices,
        "has_missing": len(missing_slices) > 0,
        "cleaning_date": datetime.now().isoformat(),
        "status": "ready" if len(missing_slices) == 0 else "needs_interpolation"
    }
    
    # Save to specimen root (not inside org_slices)
    with open(folder.parent / "metadata_cleaned.json", 'w', encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

# =========================================================
# STEP 6: Create READY flag file
# =========================================================

def create_ready_flag(folder, total_slices, missing_slices, image_size):
    """Create READY_FOR_PERSON_2.txt file."""
    ready_file = folder.parent / "READY_FOR_PERSON_2.txt"
    with open(ready_file, 'w', encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("TDE DATASET - READY FOR PERSON 2\n")
        f.write("=" * 50 + "\n")
        f.write(f"Cleaned on: {datetime.now()}\n")
        f.write(f"Total slices: {total_slices}\n")
        f.write(f"Image size: {image_size}\n")
        f.write(f"Missing slices: {len(missing_slices)}\n")
        if missing_slices:
            f.write(f"Missing slice numbers: {missing_slices[:20]}\n")
            f.write("WARNING: Missing slices detected! Use interpolation.\n")
        else:
            f.write("STATUS: READY for 3D->2D projections\n")
        f.write("=" * 50 + "\n")

# =========================================================
# MAIN FUNCTION
# =========================================================

def clean_specimen(org_slices, specimen_name, spec_id):
    """Run all cleaning steps for one specimen."""
    print(f"\n   [SPEC] {spec_id}")
    
    # Step 1: Remove corrupted
    corrupted = remove_corrupted_files(org_slices)
    if corrupted:
        print(f"      [REMOVED] Corrupted files: {len(corrupted)}")
    
    # Step 2: Rename sequential
    slice_count = rename_sequential(org_slices)
    print(f"      [RENAMED] Sequential slices: {slice_count}")
    
    # Step 3: Check missing
    missing = check_missing_slices(org_slices)
    if missing:
        print(f"      [WARNING] Missing slices: {len(missing)} (first 10: {missing[:10]})")
    else:
        print(f"      [OK] No missing slices")
    
    # Step 4: Get voxel size
    voxel_size = VOXEL_SIZES.get(specimen_name, 10.0)
    
    # Step 5: Create metadata
    image_size = get_image_size(org_slices)
    metadata = create_metadata(org_slices, specimen_name, spec_id, slice_count, missing, voxel_size)
    print(f"      [FILE] Created metadata_cleaned.json")
    
    # Step 6: Create ready flag
    create_ready_flag(org_slices, slice_count, missing, image_size)
    print(f"      [FLAG] Created READY_FOR_PERSON_2.txt")
    
    return slice_count, len(missing)

# =========================================================
# RUN
# =========================================================

def main():
    print("=" * 70)
    print("ENHANCED DATASET CLEANER - PREPARING FOR PERSON 2")
    print("=" * 70)
    print(f"Dataset root: {RAW}")
    print("=" * 70)
    
    total_slices_all = 0
    total_missing_all = 0
    
    for specimen_dir in RAW.iterdir():
        if not specimen_dir.is_dir():
            continue
        
        specimen_name = specimen_dir.name
        print(f"\n[DIR] Processing: {specimen_name}")
        
        for id_dir in specimen_dir.iterdir():
            if not id_dir.is_dir():
                continue
            
            org_slices = id_dir / "org_slices"
            if not org_slices.exists():
                print(f"   [SKIP] No org_slices folder for {id_dir.name}")
                continue
            
            slices, missing = clean_specimen(org_slices, specimen_name, id_dir.name)
            total_slices_all += slices
            total_missing_all += missing
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total slices across all specimens: {total_slices_all}")
    print(f"Total missing slices: {total_missing_all}")
    print(f"Specimens ready for Person 2: All")
    print("=" * 70)
    print("\n[DONE] ENHANCED CLEANING COMPLETE")
    print("[INFO] Person 2 can now start generating 2D projections")
    print("=" * 70)

if __name__ == "__main__":
    main()