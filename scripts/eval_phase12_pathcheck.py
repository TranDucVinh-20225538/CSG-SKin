#!/usr/bin/env python3
"""Amendment F: verify every image path in Camelyon17 and iWildCam metadata.

Read-only against the DST-Skin WILDS tree. Writes only under results/paperB/phase12/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

DATA_ROOT = Path("/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds")
CAM_DIR = DATA_ROOT / "camelyon17_v1.0"
IW_DIR = DATA_ROOT / "iwildcam_v2.0"
OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase12")


def sha256_text(path: Path) -> dict:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "text": raw.decode("utf-8", errors="replace"),
    }


def check_camelyon(limit=None):
    meta = pd.read_csv(CAM_DIR / "metadata.csv", index_col=0, dtype={"patient": "str"})
    missing = []
    n = len(meta) if limit is None else min(limit, len(meta))
    for i, row in enumerate(meta.itertuples(index=False)):
        if limit is not None and i >= limit:
            break
        rel = (
            "patches/patient_{patient}_node_{node}/"
            "patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png".format(
                patient=row.patient, node=row.node, x=row.x_coord, y=row.y_coord
            )
        )
        p = CAM_DIR / rel
        if not os.path.exists(p):
            missing.append(rel)
            if len(missing) >= 50:
                break
    return {
        "n_metadata_rows": int(len(meta)),
        "n_checked": int(n),
        "n_missing": int(len(missing)),
        "missing_sample": missing[:20],
        "complete": len(missing) == 0 and (limit is None or n == len(meta)),
    }


def check_iwildcam(limit=None):
    meta = pd.read_csv(IW_DIR / "metadata.csv")
    missing = []
    n = len(meta) if limit is None else min(limit, len(meta))
    for i, fn in enumerate(meta["filename"].astype(str)):
        if limit is not None and i >= limit:
            break
        p = IW_DIR / "train" / fn
        if not os.path.exists(p):
            missing.append(fn)
            if len(missing) >= 50:
                break
    return {
        "n_metadata_rows": int(len(meta)),
        "n_checked": int(n),
        "n_missing": int(len(missing)),
        "missing_sample": missing[:20],
        "complete": len(missing) == 0 and (limit is None or n == len(meta)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="check only first N rows (dry sanity)")
    return p.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "data_root": str(DATA_ROOT.resolve()),
        "read_only": True,
        "camelyon17_release": sha256_text(CAM_DIR / "RELEASE_v1.0.txt"),
        "iwildcam_release": sha256_text(IW_DIR / "RELEASE_v2.0.txt"),
    }
    if args.dry_run:
        report["camelyon17"] = check_camelyon(limit=args.limit or 32)
        report["iwildcam"] = check_iwildcam(limit=args.limit or 32)
        report["dry_run"] = True
        print(json.dumps(report, indent=2)[:2000])
        return
    print("checking camelyon17 ({} rows)...".format(len(pd.read_csv(CAM_DIR / "metadata.csv", index_col=0))), flush=True)
    report["camelyon17"] = check_camelyon(limit=None)
    print("camelyon missing", report["camelyon17"]["n_missing"], flush=True)
    print("checking iwildcam...", flush=True)
    report["iwildcam"] = check_iwildcam(limit=None)
    print("iwildcam missing", report["iwildcam"]["n_missing"], flush=True)
    dest = OUT / "pathcheck.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", dest)


if __name__ == "__main__":
    main()
