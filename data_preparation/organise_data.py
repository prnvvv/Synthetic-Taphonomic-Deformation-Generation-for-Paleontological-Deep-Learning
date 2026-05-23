#!/usr/bin/env python3
"""
TDE Dataset Organizer - For already extracted folders
"""

import shutil
from pathlib import Path

# =========================================================
# PATHS
# =========================================================

SOURCE = Path(r"C:\Users\Asus\Documents\fossils")
DEST = Path(r"C:\Users\Asus\Documents\GitHub\Synthetic-Taphonomic-Deformation-Generation-for-Paleontological-Deep-Learning\dataset\raw")

# =========================================================
# MAPPING: Your folder names -> (specimen_name, id)
# =========================================================

MAPPING = {
    "Archaeopteryx": ("archaeopteryx_london", "BMNH-37001"),
    "Halszkaraptor": ("halszkaraptor", "MPC-D-102-109"),
    "Tropeognathus": ("tropeognathus", "BSPG-1987-I-46"),
    "Araripesaurus": ("araripesaurus", "BSPG-1982-I-90"),
    "Ichthyornis (YPM-1460)": ("ichthyornis", "YPM-1460"),
    "Ichthyornis (YPM-1775)": ("ichthyornis", "YPM-1775"),
    "Titanosaur embryo": ("titanosaur_embryo", "MCF-PVPH-874"),
}

# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 60)
    print("TDE DATASET ORGANIZER")
    print("=" * 60)
    print(f"Source: {SOURCE}")
    print(f"Destination: {DEST}")
    print("=" * 60)
    
    total_files = 0
    
    for folder_name, (specimen, spec_id) in MAPPING.items():
        src_path = SOURCE / folder_name
        
        if not src_path.exists():
            print(f"\n❌ Missing: {folder_name}")
            continue
        
        # Create destination folders
        dest_folder = DEST / specimen / spec_id / "org_slices"
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        mesh_folder = DEST / specimen / spec_id / "mesh"
        mesh_folder.mkdir(parents=True, exist_ok=True)
        
        # Move all files
        files = list(src_path.glob("*"))
        file_count = 0
        
        for file in files:
            if file.is_file():
                dest_file = dest_folder / file.name
                if dest_file.exists():
                    name, ext = file.stem, file.suffix
                    dest_file = dest_folder / f"{name}_dup{ext}"
                shutil.move(str(file), str(dest_file))
                file_count += 1
        
        total_files += file_count
        
        print(f"\n✅ {folder_name}")
        print(f"   → {specimen}/{spec_id}/org_slices/")
        print(f"   📁 Moved {file_count} files")
    
    print("\n" + "=" * 60)
    print(f"✅ COMPLETE! Total files moved: {total_files}")
    print(f"📍 Location: {DEST}")
    print("=" * 60)

if __name__ == "__main__":
    main()