#!/usr/bin/env python3
"""R2 Item 4: Phase 3 training with the ISIC split grouped by lesion_id. Everything else fixed.

Null lesion_id policy: each such image is its own group (kept). Phase 2 checkpoint reuse at λ=2 is
disabled because those checkpoints were trained on the image-level split.
After training, Phase 13-style features are cached for the Table 1 columns that need logits (OOD ECE).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import train_phase3_sweep as p3  # noqa: E402  (installs the transformers stub)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import eval_final_gpu as fin  # noqa: E402
import extract_phase13_features as x13  # noqa: E402
import r2_common as C  # noqa: E402
import train_phase2_pad_holdout as p2  # noqa: E402
from eval_fitz_domain_axis import FolderJpegDataset  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust  # noqa: E402

OUT = C.R2 / "item4"
RUNS = OUT / "runs"
CKPT = C.PAPERB / "checkpoints" / "r2_item4"
FEAT = OUT / "features"


def lesion_groups(frame):
    meta = pd.read_csv(C.ISIC_META, usecols=["image", "lesion_id"])
    stem = frame.path.str.extract(r"([^/]+)\.\w+$")[0]
    lid = stem.map(dict(zip(meta.image, meta.lesion_id)))
    return lid.fillna(stem).to_numpy(), lid.isna().to_numpy()


def isic_splits_by_lesion(isic):
    isic = isic.reset_index(drop=True)
    groups, _ = lesion_groups(isic)
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tv_i, te_i = next(sgk.split(np.zeros(len(isic)), isic["label_idx"], groups))
    tv, test = isic.iloc[tv_i].reset_index(drop=True), isic.iloc[te_i].reset_index(drop=True)
    g_tv = groups[tv_i]
    sgk2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr_i, va_i = next(sgk2.split(np.zeros(len(tv)), tv["label_idx"], g_tv))
    return tv.iloc[tr_i].reset_index(drop=True), tv.iloc[va_i].reset_index(drop=True), test


def split_record():
    df = p2.load_master()
    isic = df[df.domain == "isic"].copy().reset_index(drop=True)
    train, val, test = isic_splits_by_lesion(isic)
    rec = {"policy_null_lesion_id": "each image with null lesion_id is its own group (kept, not dropped)"}
    gs = {}
    for name, f in (("train", train), ("val", val), ("test", test)):
        g, null = lesion_groups(f)
        gs[name] = set(g)
        rec[name] = {"n": int(len(f)), "null_lesion_frac": float(null.mean()),
                     "label_counts": f.label.value_counts().to_dict()}
    _, null_all = lesion_groups(isic)
    rec["null_lesion_frac_all_isic"] = float(null_all.mean())
    rec["n_null_lesion_all_isic"] = int(null_all.sum())
    rec["lesion_overlap_train_test"] = len(gs["train"] & gs["test"])
    rec["lesion_overlap_val_test"] = len(gs["val"] & gs["test"])
    rec["lesion_overlap_train_val"] = len(gs["train"] & gs["val"])
    assert rec["lesion_overlap_train_test"] == rec["lesion_overlap_val_test"] == rec["lesion_overlap_train_val"] == 0
    return rec, (train, val, test)


def extract_features(tag, splits):
    summary = json.loads((RUNS / tag / "summary.json").read_text())
    net = fin.load_csg(Path(summary["best_checkpoint"]), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(net.parameters()).device
    spec = json.loads(p3.P2_SPLIT.read_text())
    df = p2.load_master()
    pad = df[df.domain == "pad_ufes"].copy()
    hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    tf = build_val_transform_robust()
    train, _val, test = splits
    out = FEAT / tag
    out.mkdir(parents=True, exist_ok=True)
    frames = {
        "isic_train": SkinDataset(train, transform=tf),
        "isic_test": SkinDataset(test, transform=tf),
        "pad_full": SkinDataset(pad, transform=tf),
        "pad_heldout": SkinDataset(hold, transform=tf),
        "fitzpatrick17k": FolderJpegDataset(sorted(p3.FITZ_DIR.glob("*.jpg")), tf),
    }
    for name, ds in frames.items():
        pack = x13.collect_depths(net, DataLoader(ds, batch_size=64, shuffle=False, num_workers=4), device)
        np.savez_compressed(out / "{}.npz".format(name), **{k: pack[k] for k in x13.REQUIRED})
        print("features", tag, name, pack["labels"].shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_adv", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split_only", action="store_true")
    a, rest = ap.parse_known_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rec, splits = split_record()
    (OUT / "split_record.json").write_text(json.dumps(rec, indent=2) + "\n")
    print("split", {k: rec[k]["n"] for k in ("train", "val", "test")}, "null frac", rec["null_lesion_frac_all_isic"], flush=True)
    if a.split_only:
        return

    p2.isic_splits = isic_splits_by_lesion
    p3.find_best_ckpt = lambda _d: None
    sys.argv = [sys.argv[0], "--lambda_adv", str(a.lambda_adv), "--seed", str(a.seed),
                "--output_root", str(RUNS), "--ckpt_root", str(CKPT)] + rest
    p3.main()
    tag = "runB_orth1_ladv{}_s{}".format(p3.lam_tag(a.lambda_adv), a.seed)
    if not (FEAT / tag / "fitzpatrick17k.npz").exists():
        extract_features(tag, splits)


if __name__ == "__main__":
    main()
