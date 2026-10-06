#!/usr/bin/env python3
"""Phase 13 — cache lesion-branch depths + logits from Phase 3 checkpoints.

No new training. CPU by default (does not compete with Phase 12 GPUs).
Outputs: results/paperB/phase13/features/<run>/<split>.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

if "transformers" not in sys.modules:
    _tf = types.ModuleType("transformers")
    _tf.AutoModel = object
    _tf.AutoTokenizer = object
    _cfg = types.ModuleType("transformers.configuration_utils")

    class _PC:
        def __eq__(self, other):
            return False

    _cfg.PretrainedConfig = _PC
    _tf.configuration_utils = _cfg
    sys.modules["transformers"] = _tf
    sys.modules["transformers.configuration_utils"] = _cfg

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import eval_final_gpu as fin
import train_phase2_pad_holdout as p2
import train_phase3_sweep as p3
from eval_fitz_domain_axis import FolderJpegDataset
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust
from src.utils.seed import seed_everything

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
OUT = PAPERB / "results" / "paperB" / "phase13" / "features"
P2_SPLIT = p3.P2_SPLIT
FITZ_DIR = p3.FITZ_DIR
REQUIRED = (
    "backbone_raw_lesion",
    "z_lesion",
    "z_lesion_norm",
    "z_context",
    "logits",
    "labels",
)
SPLITS = ("isic_train", "isic_test", "pad_full", "pad_heldout", "fitzpatrick17k")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true", default=True)
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


@torch.no_grad()
def collect_depths(net, loader, device):
    net.eval()
    bb, zpre, znorm, zc, lg, y = [], [], [], [], [], []
    for batch in loader:
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        x_les = net._rgb_to_gray3(images)
        feat_les = net.extract_lesion_features(x_les)
        z = net.lesion_projector(feat_les)
        zn = net.lesion_bn(z)
        logits = net.lesion_classifier(zn)
        feat_ctx = net.extract_context_features(images)
        zctx = net.context_projector(feat_ctx)
        bb.append(feat_les.cpu().numpy())
        zpre.append(z.cpu().numpy())
        znorm.append(zn.cpu().numpy())
        zc.append(zctx.cpu().numpy())
        lg.append(logits.cpu().numpy())
        y.append(labels.numpy() if torch.is_tensor(labels) else np.asarray(labels))
    return {
        "backbone_raw_lesion": np.concatenate(bb).astype(np.float32),
        "z_lesion": np.concatenate(zpre).astype(np.float32),
        "z_lesion_norm": np.concatenate(znorm).astype(np.float32),
        "z_context": np.concatenate(zc).astype(np.float32),
        "logits": np.concatenate(lg).astype(np.float32),
        "labels": np.concatenate(y).astype(np.int64),
        "note": (
            "z_lesion = projector output (16-d, pre-BN). "
            "z_lesion_norm = lesion_bn(z_lesion); this is what Phase 3 stored as 'z_lesion'. "
            "backbone_raw_lesion = EffNet-B3 lesion encoder (~1536-d), grayscale input."
        ),
    }


def split_done(run_dir):
    for name in SPLITS:
        p = run_dir / "{}.npz".format(name)
        if not p.exists():
            return False
        with np.load(p) as z:
            if any(k not in z.files for k in REQUIRED):
                return False
    return True


def main():
    args = parse_args()
    if args.task_id < 0 or args.task_id >= len(p3.JOBS):
        raise SystemExit("task_id out of range 0-{}".format(len(p3.JOBS) - 1))
    lam, seed = p3.JOBS[args.task_id]
    seed_everything(int(seed))
    tag = "runB_orth1_ladv{}_s{}".format(p3.lam_tag(lam), seed)
    run_dir = OUT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_done and split_done(run_dir):
        print("skip_done", run_dir, flush=True)
        return
    summary_path = PAPERB / "results" / "paperB" / "phase3_sweep" / tag / "summary.json"
    summary = json.loads(summary_path.read_text())
    ckpt = Path(summary["best_checkpoint"])
    if not ckpt.exists():
        raise RuntimeError("missing ckpt {}".format(ckpt))
    cfg = {
        "run_name": tag,
        "lambda_adv": float(lam),
        "seed": int(seed),
        "checkpoint": str(ckpt),
        "phase3_z_lesion_is": "z_lesion_norm",
        "cpu": True,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if args.dry_run:
        print("dry_run", cfg)
        return

    device = torch.device("cpu")
    print("device", device, "run", tag, flush=True)
    net = fin.load_csg(ckpt, device)

    spec = json.loads(P2_SPLIT.read_text())
    df = p2.load_master()
    isic = df[df.domain == "isic"].copy()
    pad = df[df.domain == "pad_ufes"].copy()
    train, _val, test = p2.isic_splits(isic)
    pad_hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    eval_tf = build_val_transform_robust()

    def loader(frame):
        return p2.make_loader(SkinDataset(frame, transform=eval_tf), args.batch_size, args.num_workers, False)

    frames = {
        "isic_train": loader(train),
        "isic_test": loader(test),
        "pad_full": loader(pad),
        "pad_heldout": loader(pad_hold),
    }
    fitz_paths = sorted(FITZ_DIR.glob("*.jpg"))
    frames["fitzpatrick17k"] = DataLoader(
        FolderJpegDataset(fitz_paths, eval_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    for name, ld in frames.items():
        print("extract", name, flush=True)
        pack = collect_depths(net, ld, device)
        pack["split"] = name
        pack["run_name"] = tag
        np.savez_compressed(run_dir / "{}.npz".format(name), **{k: pack[k] for k in REQUIRED})
        print("  saved", name, pack["labels"].shape, pack["backbone_raw_lesion"].shape, flush=True)

    print("done", tag, flush=True)


if __name__ == "__main__":
    main()
