#!/usr/bin/env python3

import shutil
import zipfile
from pathlib import Path

# =========================================================
# DATASET ROOT
# =========================================================

DATASET_ROOT = Path(
    r"C:\Users\Asus\Documents\GitHub\Synthetic-Taphonomic-Deformation-Generation-for-Paleontological-Deep-Learning\dataset\raw"
)

# =========================================================
# MANUAL LOCAL PATHS
# =========================================================
# Attach your ZIP/folder paths manually here

DATASETS = {

    # =====================================================
    # ARARIPESAURUS
    # =====================================================

    "araripesaurus": {
        "BSPG-1982-I-90": {

            "mesh": Path(
                r"C:\YOUR_PATH\araripesaurus_mesh.zip"
            ),

            "scans": Path(
                r"C:\YOUR_PATH\araripesaurus_scans.zip"
            )
        }
    },

    # =====================================================
    # ARCHAEOPTERYX
    # =====================================================

    "archaeopteryx_london": {
        "BMNH-37001": {

            "mesh": Path(
                r"C:\YOUR_PATH\archaeopteryx_mesh.zip"
            ),

            "scans": Path(
                r"C:\YOUR_PATH\archaeopteryx_scans.zip"
            )
        }
    },

    # =====================================================
    # HALSZKARAPTOR
    # =====================================================

    "halszkaraptor": {
        "MPC-D-102-109": {

            "mesh": Path(
                r"C:\YOUR_PATH\halszkaraptor_mesh.zip"
            ),

            "scans": Path(
                r"C:\YOUR_PATH\halszkaraptor_scans.zip"
            )
        }
    },

    # =====================================================
    # HESPERORNIS
    # =====================================================

    "hesperornis": {
        "YPM-1206A": {

            "mesh": Path(
                r"C:\YOUR_PATH\hesperornis_mesh.zip"
            ),

            "scans": Path(
                r"C:\YOUR_PATH\hesperornis_scans.zip"
            )
        }
    },

    # =====================================================
    # ICHTHYORNIS (SPECIAL CASE)
    # =====================================================

    "ichthyornis": {

        # -------------------------------
        # YPM-1460
        # -------------------------------

        "YPM-1460": {

            "mesh": Path(
                r"C:\YOUR_PATH\ichthyornis_1460_mesh.zip"
            ),

            "scans": Path(
                r"C:\YOUR_PATH\ichthyornis_1460_scans.zip"
            )
        },

        # -------------------------------
        # YPM-1775
        # -------------------------------

        "YPM-1775": {

            "mesh": Path(
                r"C:\YOUR_PATH\ichthyornis_1775_mesh.zip"
            ),

            "scans": Path(
                r"C:\YOUR_PATH\ichthyornis_1775_scans.zip"
            )
        }
    },

    # =====================================================
    # TITANOSAUR
    # =====================================================

    "titanosaur_embryo": {
        "MCF-PVPH-874": {

            "mesh": Path(
                r"C:\YOUR_PATH\titanosaur_mesh.zip"
            ),

            "scans": Path(
                r"C:\YOUR_PATH\titanosaur_scans.zip"
            )
        }
    },

    # =====================================================
    # TROPEOGNATHUS
    # =====================================================

    "tropeognathus": {
        "BSPG-1987-I-46": {

            "mesh": Path(
                r"C:\YOUR_PATH\tropeognathus_mesh.zip"
            ),

            "scans": Path(
                r"C:\YOUR_PATH\tropeognathus_scans.zip"
            )
        }
    }
}

# =========================================================
# HELPERS
# =========================================================

def extract_zip(zip_path, destination):

    print(f"   📦 Extracting: {zip_path.name}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination)

    print("   ✅ Extraction complete")


def copy_folder(src, dst):

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)


def process_input(source_path, destination):

    if not source_path.exists():

        print(f"   ❌ Missing: {source_path}")
        return

    # ZIP
    if source_path.suffix.lower() == ".zip":

        extract_zip(source_path, destination)

    # FOLDER
    elif source_path.is_dir():

        copy_folder(source_path, destination)

        print(f"   ✅ Copied folder")

    else:

        print(f"   ❌ Unsupported file type")


# =========================================================
# MAIN ORGANIZER
# =========================================================

def organize():

    print("=" * 60)
    print("TDE DATASET ORGANIZER")
    print("=" * 60)

    for specimen, ids in DATASETS.items():

        print(f"\n🦴 {specimen}")

        for specimen_id, files in ids.items():

            print(f"\n   📁 {specimen_id}")

            target_base = (
                DATASET_ROOT /
                specimen /
                specimen_id
            )

            mesh_dest = target_base / "mesh"
            scans_dest = target_base / "org_slices"

            mesh_dest.mkdir(parents=True, exist_ok=True)
            scans_dest.mkdir(parents=True, exist_ok=True)

            # ---------------------------------------------
            # MESH
            # ---------------------------------------------

            print("   🔹 mesh")

            process_input(
                files["mesh"],
                mesh_dest
            )

            # ---------------------------------------------
            # SCANS
            # ---------------------------------------------

            print("   🔹 scans -> org_slices")

            process_input(
                files["scans"],
                scans_dest
            )

    print("\n" + "=" * 60)
    print("✅ ORGANIZATION COMPLETE")
    print("=" * 60)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    organize()