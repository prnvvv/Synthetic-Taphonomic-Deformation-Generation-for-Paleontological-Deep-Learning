#!/usr/bin/env python3
"""
Person 1: Create folder structure for ESRF fossil data
Creates the complete dataset/raw hierarchy with org_slices and mesh folders
"""

from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_RAW = PROJECT_ROOT / "dataset" / "raw"

# =========================================================
# SPECIMEN LIST: (specimen_name, id)
# =========================================================

SPECIMENS = [
    ("archaeopteryx", "BMNH-37001"),
    ("halszkaraptor", "MPC-D-102-109"),
    ("tropeognathus", "BSPG-1987-I-46"),
    ("araripesaurus", "BSPG-1982-I-90"),
    ("ichthyornis", "YPM-1460"),
    ("ichthyornis", "YPM-1775"),
    ("titanosaur_embryo", "MCF-PVPH-874"),
    ("hesperornis", "YPM-1206A"),
]

# =========================================================
# CREATE FOLDER STRUCTURE
# =========================================================

def create_structure():
    print("=" * 70)
    print("PERSON 1: CREATING DATASET FOLDER STRUCTURE")
    print("=" * 70)
    print(f"Destination: {DATASET_RAW}")
    print("=" * 70)

    for specimen, spec_id in SPECIMENS:
        # Create paths
        spec_path = DATASET_RAW / specimen / spec_id
        org_slices_path = spec_path / "org_slices"
        mesh_path = spec_path / "mesh"

        # Create folders
        org_slices_path.mkdir(parents=True, exist_ok=True)
        mesh_path.mkdir(parents=True, exist_ok=True)

        print(f"✅ Created: {specimen}/{spec_id}/")
        print(f"   ├── org_slices/")
        print(f"   └── mesh/")

    # Create download log file
    log_path = DATASET_RAW / "download_log.csv"
    if not log_path.exists():
        with open(log_path, 'w') as f:
            f.write("specimen,id,download_date,status,notes\n")
            for specimen, spec_id in SPECIMENS:
                f.write(f"{specimen},{spec_id},pending,not_downloaded,\n")
        print(f"\n✅ Created: download_log.csv")

    print("\n" + "=" * 70)
    print("✅ PERSON 1 FOLDER STRUCTURE CREATED")
    print(f"📍 Location: {DATASET_RAW}")
    print("=" * 70)
    print("\nNext steps for Person 1:")
    print("1. Download ZIP files from ESRF Paleo database")
    print("2. Extract ZIPs to C:\\Users\\Asus\\Documents\\fossils\\")
    print("3. Run arrange_fossils.py to move files into this structure")
    print("4. Run clean_dataset.py to rename slices sequentially")
    print("=" * 70)

if __name__ == "__main__":
    create_structure()