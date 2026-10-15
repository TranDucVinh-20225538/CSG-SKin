#!/usr/bin/env python3
"""Paper B R5 P4 (GPU part): lesion-level retrain at λ = 2 with adversary LR ×10 and ×100 instead of ×30
(decisions/paperB_r5_p4.md). Same recipe as train_r2_item4_lesion_split.py; only --adv_lr_multiplier changes.

Why a wrapper: train_r2_item4_lesion_split.main builds the feature tag without the `_advlr<m>` suffix that
train_phase3_sweep appends, so it would skip extraction for these runs (the λ = 2 ×30 features already exist under the
unsuffixed tag). This wrapper calls the same pieces and writes features under the suffixed tag in its own output tree.
Usage: r5_p4_advlr_train.py --seed 42 --adv_lr_multiplier 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import train_r2_item4_lesion_split as T  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

OUT = T.C.PAPERB / "results" / "paperB" / "r5" / "p4" / "advlr"
RUNS, FEAT, CKPT = OUT / "runs", OUT / "features", T.C.PAPERB / "checkpoints" / "r5_p4_advlr"
LAM = 2.0


def extract(tag, splits):
    summary = json.loads((RUNS / tag / "summary.json").read_text())
    assert not summary.get("reused_phase2_ckpt"), "Phase 2 checkpoint reuse must be off"
    net = T.fin.load_csg(Path(summary["best_checkpoint"]), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(net.parameters()).device
    spec = json.loads(T.p3.P2_SPLIT.read_text())
    df = T.p2.load_master()
    pad = df[df.domain == "pad_ufes"].copy()
    hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    tf = T.build_val_transform_robust()
    train, _val, test = splits
    out = FEAT / tag
    out.mkdir(parents=True, exist_ok=True)
    frames = {"isic_train": T.SkinDataset(train, transform=tf), "isic_test": T.SkinDataset(test, transform=tf),
              "pad_full": T.SkinDataset(pad, transform=tf), "pad_heldout": T.SkinDataset(hold, transform=tf),
              "fitzpatrick17k": T.FolderJpegDataset(sorted(T.p3.FITZ_DIR.glob("*.jpg")), tf)}
    for name, ds in frames.items():
        pack = T.x13.collect_depths(net, DataLoader(ds, batch_size=64, shuffle=False, num_workers=4), device)
        np.savez_compressed(out / "{}.npz".format(name), **{k: pack[k] for k in T.x13.REQUIRED})
        print("features", tag, name, pack["labels"].shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--adv_lr_multiplier", type=float, required=True)
    a, rest = ap.parse_known_args()
    assert a.adv_lr_multiplier in (10.0, 100.0)
    _rec, splits = T.split_record()
    T.p2.isic_splits = T.isic_splits_by_lesion
    T.p3.find_best_ckpt = lambda _d: None
    sys.argv = [sys.argv[0], "--lambda_adv", str(LAM), "--seed", str(a.seed), "--adv_lr_multiplier", str(a.adv_lr_multiplier),
                "--output_root", str(RUNS), "--ckpt_root", str(CKPT)] + rest
    T.p3.main()
    tag = "runB_orth1_ladv2_s{}_advlr{}".format(a.seed, T.p3.lam_tag(a.adv_lr_multiplier))
    if not (FEAT / tag / "fitzpatrick17k.npz").exists():
        extract(tag, splits)


if __name__ == "__main__":
    main()
