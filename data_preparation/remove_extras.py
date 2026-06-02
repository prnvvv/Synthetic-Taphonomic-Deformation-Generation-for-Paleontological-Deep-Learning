#!/usr/bin/env python3
"""
TDE Dataset Cleaner - Handles nested folders (like Hesperornis)
"""

import os
import shutil
from pathlib import Path

# =========================================================
# PATHS
# =========================================================

ROOT = (Path(__file__).resolve().parent / "..").resolve()
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", ROOT / "dataset"))
RAW = DATASET_ROOT / "raw"
EXTRAS = DATASET_ROOT / "raw_extras"

# =========================================================
# IMAGE FILE EXTENSIONS TO KEEP
# =========================================================

IMAGE_EXTENSIONS = {'.jp2', '.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}

# =========================================================
# FUNCTION: Find all image files recursively
# =========================================================

def find_all_images(folder):
    """Recursively find all image files in a folder."""
    images = []
    for file in folder.rglob("*"):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file)
    return images

def find_all_non_images(folder):
    """Recursively find all non-image files."""
    non_images = []
    for file in folder.rglob("*"):
        if file.is_file() and file.suffix.lower() not in IMAGE_EXTENSIONS:
            non_images.append(file)
    return non_images

# =========================================================
# MAIN CLEANING FUNCTION
# =========================================================

def clean_specimen(specimen_path, extras_path):
    """Recursively separate images from other files."""
    
    org_slices = specimen_path / "org_slices"
    
    if not org_slices.exists():
        print(f"   ⚠️ No org_slices folder")
        return 0, 0
    
    # Create extras folder for this specimen (preserve structure)
    extras_specimen = extras_path / specimen_path.parent.name / specimen_path.name
    extras_specimen.mkdir(parents=True, exist_ok=True)
    
    # Find ALL images recursively
    all_images = find_all_images(org_slices)
    all_non_images = find_all_non_images(org_slices)
    
    image_count = len(all_images)
    extra_count = 0
    
    # Move non-image files to extras (preserve relative path)
    for file in all_non_images:
        # Get relative path from org_slices
        rel_path = file.relative_to(org_slices)
        dest = extras_specimen / "org_slices" / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file), str(dest))
        extra_count += 1
        print(f"      📄 Moved: {rel_path}")
    
    # For images: optionally flatten them to org_slices root (simpler)
    # Or keep them in their subfolders - your choice
    for img in all_images:
        # Keep in place or flatten - keeping in subfolders preserves structure
        pass  # Images stay where they are
    
    return image_count, extra_count

def create_empty_mesh_folders():
    """Ensure mesh folders exist."""
    print("\n📁 Creating/verifying mesh folders...")
    
    for specimen_dir in RAW.iterdir():
        if specimen_dir.is_dir():
            for id_dir in specimen_dir.iterdir():
                if id_dir.is_dir():
                    mesh_folder = id_dir / "mesh"
                    mesh_folder.mkdir(exist_ok=True)
                    print(f"   ✅ {specimen_dir.name}/{id_dir.name}/mesh/")

def verify_structure():
    """Print summary of what's in org_slices."""
    print("\n🔍 VERIFYING STRUCTURE")
    print("=" * 70)
    
    for specimen_dir in RAW.iterdir():
        if not specimen_dir.is_dir():
            continue
        for id_dir in specimen_dir.iterdir():
            if not id_dir.is_dir():
                continue
            org_slices = id_dir / "org_slices"
            if not org_slices.exists():
                print(f"⚠️ {specimen_dir.name}/{id_dir.name}: no org_slices")
                continue
            
            # Count images recursively
            images = find_all_images(org_slices)
            non_images = find_all_non_images(org_slices)
            
            print(f"\n📁 {specimen_dir.name}/{id_dir.name}")
            print(f"   🖼️ Images: {len(images)}")
            print(f"   📄 Other files: {len(non_images)}")
            
            if non_images:
                print(f"   ⚠️ Non-image files remain (run cleaner again)")

# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 70)
    print("TDE DATASET CLEANER (WITH NESTED FOLDER SUPPORT)")
    print("=" * 70)
    print(f"Raw data: {RAW}")
    print(f"Extras will be moved to: {EXTRAS}")
    print("=" * 70)
    
    EXTRAS.mkdir(parents=True, exist_ok=True)
    
    specimen_stats = {}
    
    for specimen_dir in RAW.iterdir():
        if not specimen_dir.is_dir():
            continue
        
        print(f"\n📁 Processing: {specimen_dir.name}")
        
        for id_dir in specimen_dir.iterdir():
            if not id_dir.is_dir():
                continue
            
            print(f"   🔸 {id_dir.name}")
            
            images, extras = clean_specimen(id_dir, EXTRAS)
            specimen_stats[f"{specimen_dir.name}/{id_dir.name}"] = (images, extras)
            
            print(f"      ✅ Images found: {images}")
            print(f"      📦 Extras moved: {extras}")
    
    create_empty_mesh_folders()
    verify_structure()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_images = sum(v[0] for v in specimen_stats.values())
    total_extras = sum(v[1] for v in specimen_stats.values())
    
    for name, (img, ext) in specimen_stats.items():
        print(f"{name:40} | Images: {img:6} | Extras: {ext:4}")
    
    print("-" * 70)
    print(f"{'TOTAL':40} | Images: {total_images:6} | Extras: {total_extras:4}")
    print("=" * 70)
    
    print(f"\n✅ DONE! Images ready for Project 1 (Synthetic Deformation)")

if __name__ == "__main__":
    main()