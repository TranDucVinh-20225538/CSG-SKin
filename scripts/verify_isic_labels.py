#!/usr/bin/env python3
"""
Check that ISIC rows in master_metadata.csv match official one-hot labels
(ISIC 2019 Training GroundTruth). ~12.5% classification accuracy with 8 classes
usually means wrong image–label pairs if this script reports low agreement.

Usage:
  python scripts/verify_isic_labels.py
  python scripts/verify_isic_labels.py --metadata data/master_metadata.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.preprocess_metadata import (
    ISIC_ONE_HOT_COLS,
    _find_image_column,
    _iter_groundtruth_csv_paths,
    _normalize_gt_dataframe,
    _one_hot_to_label,
)
from src.utils.paths import DATA_ROOT


def _norm_isic_id(raw):
    s = str(raw).strip()
    if s.lower().endswith(".jpg"):
        s = s[:-4]
    return s


def _stem_from_master_path(path_str):
    """Basename without extension and optional _downsampled suffix."""
    name = Path(str(path_str)).name
    if name.lower().endswith(".jpg"):
        name = name[:-4]
    if name.endswith("_downsampled"):
        name = name[: -len("_downsampled")]
    return name


def _load_ground_truth_mapping():
    """image_id (normalized) -> lesion label string (MEL, NV, ...)."""
    for gt_path in _iter_groundtruth_csv_paths():
        if gt_path.stat().st_size == 0:
            continue
        gt = _normalize_gt_dataframe(pd.read_csv(gt_path))
        if not all(c in gt.columns for c in ISIC_ONE_HOT_COLS):
            continue
        img_col = _find_image_column(gt)
        if img_col is None:
            continue
        m = {}
        for _, row in gt.iterrows():
            lab = _one_hot_to_label(row, ISIC_ONE_HOT_COLS)
            if isinstance(lab, float) and pd.isna(lab):
                continue
            iid = _norm_isic_id(row[img_col])
            m[iid] = str(lab)
        return m, gt_path
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, default=DATA_ROOT / "master_metadata.csv")
    p.add_argument("--max_mismatch_print", type=int, default=15)
    args = p.parse_args()

    if not args.metadata.is_file():
        print("File not found: {}".format(args.metadata))
        return 1

    gt_map, gt_path = _load_ground_truth_mapping()
    if gt_map is None:
        print(
            "No valid ISIC GroundTruth CSV found under {}.\n"
            "  Expected e.g. {} with columns {}.\n"
            "  Download ISIC 2019 Task 3 training ground truth from the challenge site,\n"
            "  then rebuild master with: python -m src.datasets.preprocess_metadata"
            "".format(DATA_ROOT, "ISIC_2019_Training_GroundTruth.csv", ISIC_ONE_HOT_COLS)
        )
        return 2

    print("Using ground truth file: {}".format(gt_path))

    master = pd.read_csv(args.metadata)
    isic = master[master["domain"] == "isic"].copy()
    if isic.empty:
        print("No ISIC rows in master.")
        return 1

    n = 0
    n_ok = 0
    n_no_gt = 0
    n_mismatch = 0
    mismatches = []

    for _, row in isic.iterrows():
        stem = _stem_from_master_path(row["path"])
        if stem not in gt_map:
            n_no_gt += 1
            continue
        n += 1
        exp = gt_map[stem]
        got = str(row["label"]).strip()
        if got == exp:
            n_ok += 1
        else:
            n_mismatch += 1
            if len(mismatches) < args.max_mismatch_print:
                mismatches.append((row["path"], got, exp))

    print("\n=== ISIC label check (master vs GroundTruth) ===")
    print("ISIC rows in master: {}".format(len(isic)))
    print("Rows with matching image_id in GT: {}".format(n))
    print("Rows with no GT entry for that image_id: {}".format(n_no_gt))
    if n > 0:
        print("Agreement on overlapping rows: {:.2%} ({}/{})".format(n_ok / n, n_ok, n))
    if n_mismatch:
        print("\nSample path | label in master | label in GroundTruth")
        for path, got, exp in mismatches:
            print("  {} | {} | {}".format(Path(path).name, got, exp))
        print(
            "\nIf agreement is not ~100%, regenerate master_metadata.csv with preprocess_metadata.py\n"
            "after placing the official ISIC 2019 Training GroundTruth CSV under data/."
        )
    else:
        if n == 0:
            print("Could not compare any row (image IDs not found in GT).")
        else:
            print("All comparable ISIC rows match GroundTruth.")

    if n_mismatch:
        return 3
    if n == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
