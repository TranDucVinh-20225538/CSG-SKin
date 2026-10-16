#!/usr/bin/env python3
"""Extract lesion-level item4 features for D1 follow-up dose cells (same path as train_r2_item4_lesion_split).

Only reads checkpoints under checkpoints/r2_item4/ (lesion-level Table 1). Never phase3 / phase13.
Usage:
  r5_d1_followup_extract.py --task N          # N indexes missing (lam, seed) pairs
  r5_d1_followup_extract.py --list
  r5_d1_followup_extract.py --lam 0.25 --seed 52
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import train_phase3_sweep as p3  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import eval_final_gpu as fin  # noqa: E402
import extract_phase13_features as x13  # noqa: E402
import r2_common as C  # noqa: E402
import train_phase2_pad_holdout as p2  # noqa: E402
import train_r2_item4_lesion_split as T  # noqa: E402
from eval_fitz_domain_axis import FolderJpegDataset  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust  # noqa: E402

ITEM4 = C.R2 / "item4"
CKPT = C.PAPERB / "checkpoints" / "r2_item4"
FEAT = ITEM4 / "features"
RUNS = ITEM4 / "runs"
P7A = C.PAPERB / "results" / "paperB" / "r5" / "p7a" / "sets"
CORE = ("isic_train", "isic_test", "pad_full", "pad_heldout", "fitzpatrick17k")
P7A_NAMES = ("mednode", "sd198", "scin")
ALL = CORE + P7A_NAMES
# Dose lambdas that may still need extract when lesion-level ckpts appear.
LAMS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 0.0)
SEEDS = (42, 52, 62, 72, 82)


def tag(lam, seed):
    return "runB_orth1_ladv{}_s{}".format(C.lam_tag(lam), seed)


def missing_pairs():
    """(lam, seed) with r2_item4 checkpoint but incomplete item4 features."""
    out = []
    for lam in LAMS:
        for seed in SEEDS:
            t = tag(lam, seed)
            ck = CKPT / t
            if not ck.exists() and not (RUNS / t / "summary.json").exists():
                continue
            feat = FEAT / t
            miss = [n for n in ALL if not (feat / "{}.npz".format(n)).exists()]
            if miss:
                out.append((lam, seed, miss))
    return out


def extract_one(lam, seed):
    t = tag(lam, seed)
    summary_path = RUNS / t / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("no item4 run summary (lesion-level) for {}".format(t))
    summary = json.loads(summary_path.read_text())
    ckpt = Path(summary["best_checkpoint"])
    if "r2_item4" not in str(ckpt) and "checkpoints/r2_item4" not in str(ckpt.resolve()):
        raise RuntimeError("refusing non-item4 checkpoint for {}: {}".format(t, ckpt))
    _rec, (train, _val, test) = T.split_record()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = fin.load_csg(ckpt, device)
    spec = json.loads(p3.P2_SPLIT.read_text())
    df = p2.load_master()
    pad = df[df.domain == "pad_ufes"].copy()
    hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    tf = build_val_transform_robust()
    frames = {
        "isic_train": SkinDataset(train, transform=tf),
        "isic_test": SkinDataset(test, transform=tf),
        "pad_full": SkinDataset(pad, transform=tf),
        "pad_heldout": SkinDataset(hold, transform=tf),
        "fitzpatrick17k": FolderJpegDataset(sorted(p3.FITZ_DIR.glob("*.jpg")), tf),
    }
    for name in P7A_NAMES:
        paths = pd.read_csv(P7A / "{}.csv".format(name)).path.tolist()
        frames[name] = FolderJpegDataset(paths, tf)
    out = FEAT / t
    out.mkdir(parents=True, exist_ok=True)
    todo = [n for n in ALL if not (out / "{}.npz".format(n)).exists()]
    if not todo:
        print("skip_done", t, flush=True)
        return
    for name in todo:
        ds = frames[name]
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
        with torch.no_grad():
            pack = x13.collect_depths(net, loader, device)
        np.savez_compressed(out / "{}.npz".format(name), **{k: pack[k] for k in x13.REQUIRED})
        print("features", t, name, pack["labels"].shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    pairs = missing_pairs()
    if a.list:
        print("missing_count", len(pairs))
        for i, (lam, seed, miss) in enumerate(pairs):
            print(i, "λ={:g}".format(lam), "seed", seed, "miss", miss)
        # also report seeds with no lesion-level ckpt at mid-λ
        for lam in (0.25, 0.5, 1.0, 4.0, 8.0):
            absent = [s for s in SEEDS if not (CKPT / tag(lam, s)).exists()
                      and not (RUNS / tag(lam, s) / "summary.json").exists()]
            if absent:
                print("no_lesion_ckpt λ={:g} seeds {}".format(lam, absent))
        return
    if a.lam is not None and a.seed is not None:
        extract_one(a.lam, a.seed)
        return
    if a.task is None:
        ap.error("need --list, --task, or --lam/--seed")
    if not pairs:
        print("nothing_to_extract")
        return
    lam, seed, _miss = pairs[a.task]
    extract_one(lam, seed)


if __name__ == "__main__":
    main()
