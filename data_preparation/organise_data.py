#!/usr/bin/env python3
"""
TDE Dataset Organizer - For already extracted folders
"""

import argparse
import shutil
from pathlib import Path

# =========================================================
# PATHS
# =========================================================

ROOT = (Path(__file__).resolve().parent / "..").resolve()
DEFAULT_SOURCE = ROOT.parent
DEFAULT_DEST = ROOT / "dataset" / "raw"

# Runtime source/destination (can be overridden by args in future)
SOURCE = DEFAULT_SOURCE
DEST = DEFAULT_DEST

# =========================================================
# MAPPING: Your folder names -> (specimen_name, id)
# =========================================================

MAPPING = {
    "Archaeopteryx_lithographica_BMNH_37001_13um_": ("archaeopteryx_london", "BMNH-37001"),
    "107.16_mu_Halszkaraptor_jp2_": ("halszkaraptor", "MPC-D-102-109"),
    "Tropeognathus_mesembrinus_BSPG-1987-I-46_45.92um_": ("tropeognathus", "BSPG-1987-I-46"),
    "Araripesaurus_santanae_BSPG-1982-I-90_91.24um_": ("araripesaurus", "BSPG-1982-I-90"),
    "HA-S_1.28_YPM1460_-8.45_8.14_CC_RC_crop_bin2jp2_": ("ichthyornis", "YPM-1460"),
    "HA-S_1.28_YPM-1775_-9.09_6.09_CC_RC_crop_bin2jp2_": ("ichthyornis", "YPM-1775"),
    "14.92um_Titanosaurian_embryo_skull_MCF-PVPH-874_pag_-1.12_1.49_jp2_": ("titanosaur_embryo", "MCF-PVPH-874"),
    "HA_3.5_YPM1206-A_-13.00_14.33_CC_RC_crop_bin2jp2_": ("hesperornis", "YPM-1206-A"),
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
        files = [f for f in src_path.rglob("*") if f.is_file()]
        file_count = 0
        
        for file in files:
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