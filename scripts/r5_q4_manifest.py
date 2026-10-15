#!/usr/bin/env python3
"""Paper B R5 Q4 / Q5 / R6 R4: manifest of run feature directories (CPU, file listing only).

Writes results/paperB/r5/q4/manifest.csv, one row per run, with the contract every consumer (r5_q4_collapse.py,
r6_r4_warning_signals.py, r5_q5_frozen_runs.py) reads:
  run_id, task, protocol, family, lambda, seed, feat_dir, rep, id_train, id_test, oods (';'-separated split names),
  primary_ood, n_classes, majority (class index; ISIC: NV = 1), status (ready | pending), source
A run is `ready` when feat_dir holds <id_train>.npz, <id_test>.npz and every OOD split, each with keys rep, logits, labels.

Built-in families (exist now): ISIC lesion-level (35), ISIC image-level Phase 13 (31), single-encoder Phase 15.1
(after P5 extraction), P4 adversary-LR runs (after the GPU part). Later runs (R5/R6 GPU items, Q1 CXR, Q2/Q3) are added by
listing them in one or more extra CSVs with the same columns (missing columns take the ISIC defaults):
  r5_q4_manifest.py --extra results/paperB/r5/q4/extra_r5.csv results/paperB/r6/q4_extra.csv
Rows whose features are absent are kept with status `pending`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402

OUT = R.R5 / "q4"
COLS = ("run_id", "task", "protocol", "family", "lambda", "seed", "feat_dir", "rep", "id_train", "id_test", "oods",
        "primary_ood", "n_classes", "majority", "status", "source")
ISIC = {"task": "isic_skin", "rep": "z_lesion_norm", "id_train": "isic_train", "id_test": "isic_test",
        "oods": "pad_heldout;fitzpatrick17k", "primary_ood": "pad_heldout", "n_classes": 8, "majority": R.NV}


def scan(root, family, protocol, pattern, rep="z_lesion_norm", source=""):
    rows = []
    root = Path(root)
    if not root.exists():
        return rows
    for d in sorted(root.iterdir()):
        m = re.fullmatch(pattern, d.name)
        if m:
            rows.append({**ISIC, "run_id": "{}/{}".format(family, d.name), "protocol": protocol, "family": family,
                         "lambda": float(m.group(1).replace("p", ".")), "seed": int(m.group(2)), "feat_dir": str(d), "rep": rep,
                         "source": source})
    return rows


def status(r):
    d = Path(r["feat_dir"])
    need = [r["id_train"], r["id_test"]] + str(r["oods"]).split(";")
    if not all((d / "{}.npz".format(s)).exists() for s in need):
        return "pending"
    with np.load(d / "{}.npz".format(r["id_test"])) as f:
        return "ready" if all(k in f.files for k in (r["rep"], "logits", "labels")) else "pending"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    rows = []
    rows += scan(R.FEAT_L, "csg_dual_lesion", "isic_lesion_split", r"runB_orth1_ladv([0-9p]+)_s(\d+)", source="r2/item4")
    rows += scan(R.FEAT_I, "csg_dual_image", "isic_image_split", r"runB_orth1_ladv([0-9p]+)_s(\d+)", source="phase13")
    rows += scan(R.R5 / "p5" / "features", "single_dann_image", "isic_image_split", r"single_dann_ladv([0-9p]+)_s(\d+)", rep="z",
                 source="phase15/single_dann (P5 extraction)")
    rows += scan(R.R5 / "p4" / "advlr" / "features", "csg_dual_lesion_advlr", "isic_lesion_split",
                 r"runB_orth1_ladv([0-9p]+)_s(\d+)_advlr\d+", source="r5/p4 GPU part")
    for p in a.extra:
        ex = pd.read_csv(p)
        for r in ex.to_dict("records"):
            rows.append({**ISIC, "source": p, **{k: v for k, v in r.items() if pd.notna(v)}})
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df:
            df[c] = None
    df["status"] = [status(r) for r in df.to_dict("records")]
    OUT.mkdir(parents=True, exist_ok=True)
    df[list(COLS)].to_csv(OUT / "manifest.csv", index=False)
    print(df.groupby(["family", "status"]).size().to_string())


if __name__ == "__main__":
    main()
