#!/usr/bin/env python3
"""Phase 2: retrain runB_orth1 with PAD patient-level holdout (pad_adv 70% / pad_heldout 30%)."""

from __future__ import annotations

import argparse
import json
import os
import re
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
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader

REPO = Path("/data2/hpcshared/Vinh/CSG-Skin")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.constants import LABEL_TO_INDEX
from src.datasets.skin_dataset import (
    CombinedTrainDataset,
    SkinDataset,
    build_lesion_branch_transform_gray,
    build_train_transform_robust,
    build_val_transform_robust,
    csg_lite_paired_collate,
)
from src.models.csg_lightning import CSGLiteLightning
from src.utils import ood_metrics
from src.utils.seed import seed_everything

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
META = REPO / "data" / "master_metadata_lesion_only_soft.csv"
PAT = re.compile(r"PAT_(\d+)_(\d+)_(\d+)")


def remap(p: str) -> str:
    s = str(p)
    if s.startswith("/mnt/data2/Vinh/"):
        return "/data2/hpcshared/Vinh/" + s[len("/mnt/data2/Vinh/") :]
    return s


def load_master():
    df = pd.read_csv(META)
    df = df[df["label"].isin(LABEL_TO_INDEX)].copy()
    df["path"] = df["path"].map(remap)
    df["label_idx"] = df["label"].map(LABEL_TO_INDEX).astype(int)
    df = df[df["path"].map(os.path.isfile)].reset_index(drop=True)
    return df


def pad_patient_split(pad_df, seed=42, heldout_frac=0.3):
    pad = pad_df.copy()
    pids, lids = [], []
    for p in pad["path"]:
        m = PAT.search(str(p))
        if not m:
            raise RuntimeError("Cannot parse patient_id from {}".format(p))
        pids.append(m.group(1))
        lids.append(m.group(2))
    pad["patient_id"] = pids
    pad["lesion_id"] = lids
    gss = GroupShuffleSplit(n_splits=1, test_size=heldout_frac, random_state=seed)
    idx = np.arange(len(pad))
    adv_i, hold_i = next(gss.split(idx, groups=pad["patient_id"]))
    adv = pad.iloc[adv_i].reset_index(drop=True)
    hold = pad.iloc[hold_i].reset_index(drop=True)
    # hygiene: no patient overlap
    if set(adv.patient_id) & set(hold.patient_id):
        raise RuntimeError("Patient leak in PAD split")
    return adv, hold


def isic_splits(isic):
    tv, test = train_test_split(isic, test_size=0.2, stratify=isic["label_idx"], random_state=42)
    train, val = train_test_split(tv, test_size=0.2, stratify=tv["label_idx"], random_state=42)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def make_loader(ds, batch_size, workers, shuffle, drop_last=False, collate=None):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=drop_last,
        collate_fn=collate,
    )


@torch.no_grad()
def collect_z(model, loader, device, max_batches=None):
    model.eval()
    zl, zc, y = [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, labels = batch[0], batch[1]
        images = images.to(device)
        out = model(images, x_lesion=None, return_latents=True)
        _logits, _dctx, _dadv, z_l, z_ctx = out
        zl.append(z_l.cpu().numpy())
        zc.append(z_ctx.cpu().numpy())
        y.append(labels.numpy())
    return np.concatenate(zl), np.concatenate(zc), np.concatenate(y)


def maha_auroc(ztr, ytr, zid, zood, n_classes=8):
    means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(ztr, ytr, num_classes=n_classes, reg_eps=1e-3)
    sid = ood_metrics.mahalanobis_min_squared_distances(zid, means, prec)
    sood = ood_metrics.mahalanobis_min_squared_distances(zood, means, prec)
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    s = np.concatenate([sid, sood])
    return float(roc_auc_score(y, s))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_adv", type=float, default=2.0)
    p.add_argument("--lambda_orth", type=float, default=1.0)
    p.add_argument("--output_root", type=Path, default=PAPERB / "results" / "paperB" / "phase2_pad_holdout")
    p.add_argument("--ckpt_root", type=Path, default=PAPERB / "checkpoints" / "phase2")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--split_seed", type=int, default=42, help="PAD patient split seed (fixed; independent of train seed).")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    split_meta = args.output_root / "pad_patient_split.json"
    split_paths = args.output_root / "pad_patient_split_paths.json"

    df = load_master()
    isic = df[df.domain == "isic"].copy()
    pad = df[df.domain == "pad_ufes"].copy()
    train, val, test = isic_splits(isic)
    # PAD split is FIXED (split_seed), independent of training seed.
    if split_paths.exists():
        spec = json.loads(split_paths.read_text())
        adv_paths = set(spec["pad_adv_paths"])
        hold_paths = set(spec["pad_heldout_paths"])
        pad_adv = pad[pad.path.isin(adv_paths)].reset_index(drop=True)
        pad_hold = pad[pad.path.isin(hold_paths)].reset_index(drop=True)
        if pad_adv.empty or pad_hold.empty:
            raise RuntimeError("Loaded PAD split is empty. Delete {} and rebuild.".format(split_paths))
    else:
        pad_adv, pad_hold = pad_patient_split(pad, seed=args.split_seed, heldout_frac=0.3)
        spec = {
            "heldout_frac": 0.3,
            "split_seed": int(args.split_seed),
            "n_pad": int(len(pad)),
            "n_adv": int(len(pad_adv)),
            "n_heldout": int(len(pad_hold)),
            "n_adv_patients": int(pad_adv.patient_id.nunique()),
            "n_heldout_patients": int(pad_hold.patient_id.nunique()),
            "n_adv_lesions": int(pad_adv.groupby(["patient_id", "lesion_id"]).ngroups),
            "n_heldout_lesions": int(pad_hold.groupby(["patient_id", "lesion_id"]).ngroups),
            "patient_overlap": sorted(set(pad_adv.patient_id) & set(pad_hold.patient_id)),
            "lesion_pair_overlap": sorted(
                set(zip(pad_adv.patient_id, pad_adv.lesion_id)) & set(zip(pad_hold.patient_id, pad_hold.lesion_id))
            ),
            "pad_adv_paths": pad_adv.path.tolist(),
            "pad_heldout_paths": pad_hold.path.tolist(),
        }
        if spec["patient_overlap"] or spec["lesion_pair_overlap"]:
            raise RuntimeError("PAD split leaked patients or lesions")
        split_meta.write_text(json.dumps({k: v for k, v in spec.items() if k not in ("pad_adv_paths", "pad_heldout_paths")}, indent=2) + "\n")
        split_paths.write_text(json.dumps(spec) + "\n")
        print("wrote PAD split adv={} hold={}".format(len(pad_adv), len(pad_hold)))

    run_name = "runB_orth1_padhold_s{}".format(args.seed)
    run_dir = args.output_root / run_name
    ckpt_dir = args.ckpt_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_done and (run_dir / "summary.json").exists() and not args.dry_run:
        print("skip_done", run_dir / "summary.json")
        return
    cfg = vars(args).copy()
    cfg["output_root"] = str(args.output_root)
    cfg["ckpt_root"] = str(args.ckpt_root)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    train_tf = build_train_transform_robust()
    les_tf = build_lesion_branch_transform_gray()
    eval_tf = build_val_transform_robust()
    train_ds = CombinedTrainDataset(train, pad_adv, transform=train_tf, lesion_transform=les_tf)
    val_ds = SkinDataset(val, transform=eval_tf)

    if args.dry_run:
        print("dry_run train_ds", len(train_ds), "val", len(val_ds), "pad_adv", len(pad_adv), "pad_hold", len(pad_hold))
        batch = csg_lite_paired_collate([train_ds[0], train_ds[1]])
        print("batch shapes", [tuple(x.shape) if torch.is_tensor(x) else x for x in batch])
        (run_dir / "dry_run_ok.json").write_text(json.dumps({"ok": True, "n_train_steps_len": len(train_ds), "n_adv": len(pad_adv), "n_hold": len(pad_hold)}) + "\n")
        return

    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, True, drop_last=True, collate=csg_lite_paired_collate)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, False)

    model = CSGLiteLightning(
        lesion_classes=8,
        domain_classes=2,
        lesion_latent_dim=16,
        context_latent_dim=64,
        learning_rate=args.lr,
        weight_decay=1e-4,
        lambda_ctx=1.0,
        lambda_adv=args.lambda_adv,
        lambda_orth=args.lambda_orth,
        lambda_supcon=0.0,
        backbone_variant="b3",
        pretrained=True,
    )
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:02d}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_cb],
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        default_root_dir=str(run_dir),
        log_every_n_steps=20,
    )
    ckpt_path = None
    last = ckpt_dir / "last.ckpt"
    if args.resume and last.exists():
        ckpt_path = str(last)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else last
    print("best", best)
    lit = CSGLiteLightning.load_from_checkpoint(str(best), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = lit.model.to(device).eval()

    id_loader = make_loader(SkinDataset(test, transform=eval_tf), 64, args.num_workers, False)
    train_eval_loader = make_loader(SkinDataset(train, transform=eval_tf), 64, args.num_workers, False)
    ztr_l, ztr_c, ytr = collect_z(net, train_eval_loader, device)
    zid_l, zid_c, yid = collect_z(net, id_loader, device)
    pred = None  # ID acc from a second pass
    # logits for ID bal acc
    logits = []
    ys = []
    with torch.no_grad():
        for images, labels in id_loader:
            images = images.to(device)
            out, _, _ = net(images)
            logits.append(out.argmax(1).cpu().numpy())
            ys.append(labels.numpy())
    pred = np.concatenate(logits)
    yid_lab = np.concatenate(ys)
    from sklearn.metrics import balanced_accuracy_score, accuracy_score

    id_acc = float(accuracy_score(yid_lab, pred))
    id_bal = float(balanced_accuracy_score(yid_lab, pred))

    ood_sets = {
        "pad_adv": pad_adv,
        "pad_heldout": pad_hold,
        "pad_full": pad,
    }
    scores = {}
    for oname, odf in ood_sets.items():
        oloader = make_loader(SkinDataset(odf, transform=eval_tf), 64, args.num_workers, False)
        zood_l, zood_c, _ = collect_z(net, oloader, device)
        scores[oname] = {
            "z_lesion_maha_auroc": maha_auroc(ztr_l, ytr, zid_l, zood_l),
            "z_context_maha_auroc": maha_auroc(ztr_c, ytr, zid_c, zood_c),
            "n_ood": int(len(odf)),
        }
        print(oname, scores[oname])

    summary = {
        "run_name": run_name,
        "seed": args.seed,
        "best_checkpoint": str(best),
        "id_acc": id_acc,
        "id_balanced_acc": id_bal,
        "ood": scores,
        "n_pad_adv": int(len(pad_adv)),
        "n_pad_heldout": int(len(pad_hold)),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("saved", run_dir / "summary.json")


if __name__ == "__main__":
    main()
