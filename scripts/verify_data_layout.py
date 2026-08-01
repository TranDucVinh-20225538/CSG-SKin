# scripts/verify_data_layout.py
# Quick check: expected files under data/ and whether master_metadata paths exist.

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.paths import DATA_ROOT, PROJECT_ROOT


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_ROOT:   ", DATA_ROOT)
    print()

    expected = [
        DATA_ROOT / "ISIC_2019_Training_Metadata.csv",
        DATA_ROOT / "ISIC_2019_Training_Input",
        DATA_ROOT / "pad_ufes20" / "metadata.csv",
        DATA_ROOT / "pad_ufes20" / "images",
        DATA_ROOT / "master_metadata.csv",
    ]
    print("=== Expected locations (CSG-Skin layout) ===")
    for p in expected:
        ok = p.is_file() if p.suffix else p.is_dir()
        sym = ""
        if p.is_symlink():
            sym = " -> {}".format(os.readlink(p))
        print("  [{}] {}{}".format("OK" if ok else "MISSING", p, sym))

    gt = DATA_ROOT / "ISIC_2019_Training_GroundTruth.csv"
    if gt.is_file():
        print("  [OK] {} (lesion one-hot labels)".format(gt))
    else:
        print(
            "  [OPTIONAL] {} — needed if training metadata has no MEL/NV/... columns".format(gt)
        )

    master = DATA_ROOT / "master_metadata.csv"
    if not master.is_file():
        print("\nNo master_metadata.csv — run: python -m src.datasets.preprocess_metadata")
        return

    df = pd.read_csv(master)
    n = len(df)
    bad = 0
    for i, row in df.iterrows():
        p = str(row["path"])
        if not os.path.isfile(p):
            bad += 1
            if bad <= 5:
                print("Missing file: {}".format(p))

    print("\n=== master_metadata.csv ===")
    print("Rows: {}".format(n))
    print("Rows with missing files on disk: {}".format(bad))
    if bad == 0:
        print("All paths resolve to existing files.")

    root_s = str(PROJECT_ROOT.resolve())
    outside = 0
    for _, row in df.iterrows():
        p = str(row["path"])
        if not p.startswith(root_s):
            outside += 1
    if outside:
        print(
            "\nNote: {} rows use paths outside PROJECT_ROOT (e.g. symlinks). "
            "Re-run preprocess after placing images under data/ to self-contain paths.".format(
                outside
            )
        )


if __name__ == "__main__":
    main()
