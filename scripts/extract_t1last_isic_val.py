#!/usr/bin/env python3
"""Extract isic_val.npz for Table-1 last.ckpt into r5/t1last/features_last/<run>/."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import eval_final_gpu as fin  # noqa: E402
import extract_phase13_features as x13  # noqa: E402
import r5_ckpt_audit as A  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust  # noqa: E402

OUT = R.RES / "r5" / "t1last" / "features_last"


def tags_l0_l2():
    return [t for t in A.table1_runs() if A.lam_of(t) in (0.0, 2.0)]


def extract_one(tag, device, num_workers=4):
    out = OUT / tag / "isic_val.npz"
    if out.exists():
        print("skip", tag, flush=True)
        return
    data = R.load_data()
    ds = SkinDataset(data["val"], transform=build_val_transform_robust())
    net = fin.load_csg(A.ITEM4_CKPT / tag / "last.ckpt", device)
    pack = x13.collect_depths(
        net, DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers), device)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: pack[k] for k in R.FEAT_KEYS})
    print("wrote", tag, pack["labels"].shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", type=int, default=None, help="index into λ∈{0,2} tag list")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=4)
    a = ap.parse_args()
    tags = tags_l0_l2()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    print("device", device, "n_tags", len(tags), flush=True)
    if a.task_id is None:
        for tag in tags:
            extract_one(tag, device, a.num_workers)
    else:
        extract_one(tags[a.task_id], device, a.num_workers)


if __name__ == "__main__":
    main()
