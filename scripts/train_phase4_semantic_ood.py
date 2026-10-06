#!/usr/bin/env python3
"""Phase 4: semantic OOD with domain held constant (ISIC only; PAD not used).

4a: hold {DF, VASC} → 6-class train
4b: hold {SCC} → 7-class train
Methods: baseline (ResNet-50, no-robust recipe), effb3_control, runB_orth1.
"""

from __future__ import annotations

import argparse
import json
import os
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

REPO = Path("/data2/hpcshared/Vinh/CSG-Skin")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.constants import LABEL_TO_INDEX
from src.datasets.skin_dataset import (
    SkinDataset,
    build_train_transform_robust,
    build_val_transform_robust,
)
from src.models.baseline import BaselineResNet50
from src.models.csg_lightning import CSGLiteLightning
from src.models.effb3_single import EffB3SingleLightning
from src.utils import ood_metrics
from src.utils.seed import seed_everything
from torchvision import transforms as T
from src.datasets.constants import IMAGENET_MEAN, IMAGENET_STD

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
META = REPO / "data" / "master_metadata_lesion_only_soft.csv"
CONFIGS = {
    "4a": {"hold": ["DF", "VASC"], "n_keep": 6},
    "4b": {"hold": ["SCC"], "n_keep": 7},
}


def remap(p: str) -> str:
    s = str(p)
    if s.startswith("/mnt/data2/Vinh/"):
        return "/data2/hpcshared/Vinh/" + s[len("/mnt/data2/Vinh/") :]
    return s


def load_isic():
    df = pd.read_csv(META)
    df = df[df["label"].isin(LABEL_TO_INDEX)].copy()
    df["path"] = df["path"].map(remap)
    df["label_idx"] = df["label"].map(LABEL_TO_INDEX).astype(int)
    df = df[df["domain"] == "isic"]
    df = df[df["path"].map(os.path.isfile)].reset_index(drop=True)
    return df


def isic_splits(isic):
    tv, test = train_test_split(isic, test_size=0.2, stratify=isic["label_idx"], random_state=42)
    train, val = train_test_split(tv, test_size=0.2, stratify=tv["label_idx"], random_state=42)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def remap_kept(df, keep_labels):
    mapping = {lab: i for i, lab in enumerate(keep_labels)}
    out = df[df["label"].isin(keep_labels)].copy()
    out["label_idx"] = out["label"].map(mapping).astype(int)
    return out.reset_index(drop=True), mapping


class DomainConstantDataset(Dataset):
    """(image, y, domain=0) for CSG ISIC-only training."""

    def __init__(self, df, transform):
        self.inner = SkinDataset(df, transform=transform)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        img, y = self.inner[idx]
        return img, y, torch.tensor(0, dtype=torch.long)


def baseline_train_tf():
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def baseline_eval_tf():
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_loader(ds, batch_size, workers, shuffle, drop_last=False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=drop_last,
    )


def ece(logits, labels, n_bins=15):
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    n = max(len(labels), 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        out += (float(m.sum()) / n) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(out)


@torch.no_grad()
def collect(method, model, loader, device, max_batches=None):
    model.eval()
    logits, feats, zl, zc, ys = [], [], [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, labels = batch[0], batch[1]
        images = images.to(device)
        if method == "runB_orth1":
            out = model(images, x_lesion=None, return_latents=True)
            yhat, _dctx, _dadv, z_l, z_ctx = out
            logits.append(yhat.cpu().numpy())
            zl.append(z_l.cpu().numpy())
            zc.append(z_ctx.cpu().numpy())
        elif method == "effb3":
            yhat, z = model(images, return_features=True)
            logits.append(yhat.cpu().numpy())
            feats.append(z.cpu().numpy())
        else:
            yhat, z = model(images, return_features=True)
            logits.append(yhat.cpu().numpy())
            feats.append(z.cpu().numpy())
        ys.append(labels.numpy())
    y = np.concatenate(ys)
    pack = {"labels": y, "logits": np.concatenate(logits)}
    if method == "runB_orth1":
        pack["z_lesion"] = np.concatenate(zl)
        pack["z_context"] = np.concatenate(zc)
    else:
        pack["backbone_raw"] = np.concatenate(feats)
    return pack


def maha_auroc(ztr, ytr, zid, zood, n_classes):
    means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(ztr, ytr, num_classes=n_classes, reg_eps=1e-3)
    sid = ood_metrics.mahalanobis_min_squared_distances(zid, means, prec)
    sood = ood_metrics.mahalanobis_min_squared_distances(zood, means, prec)
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    s = np.concatenate([sid, sood])
    return {
        "AUROC": float(roc_auc_score(y, s)),
        "n_id": int(len(sid)),
        "n_ood": int(len(sood)),
        "feat_dim": int(ztr.shape[1]),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=["4a", "4b"], required=True)
    p.add_argument("--method", choices=["baseline", "effb3", "runB_orth1"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_root", type=Path, default=PAPERB / "results" / "paperB" / "phase4_semantic_ood")
    p.add_argument("--ckpt_root", type=Path, default=PAPERB / "checkpoints" / "phase4")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    cfg = CONFIGS[args.config]
    hold = cfg["hold"]
    all_labels = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    keep = [x for x in all_labels if x not in hold]
    n_classes = len(keep)
    assert n_classes == cfg["n_keep"]

    isic = load_isic()
    train8, val8, test8 = isic_splits(isic)
    train, _ = remap_kept(train8, keep)
    val, _ = remap_kept(val8, keep)
    id_test, mapping = remap_kept(test8, keep)
    ood = isic[isic["label"].isin(hold)].copy().reset_index(drop=True)
    ood["label_idx"] = -1

    run_name = "{}_{}_s{}".format(args.config, args.method, args.seed)
    run_dir = args.output_root / run_name
    ckpt_dir = args.ckpt_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_done and (run_dir / "summary.json").exists() and not args.dry_run:
        print("skip_done", run_dir / "summary.json")
        return

    split_note = {
        "config": args.config,
        "hold_out": hold,
        "keep": keep,
        "label_mapping": mapping,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_id_test": int(len(id_test)),
        "n_ood_holdout_classes": int(len(ood)),
        "pad_involved": False,
        "split_random_state": 42,
        "ood_definition": "all ISIC images of held-out classes (never used in train/val)",
        "id_definition": "ISIC test split restricted to kept classes",
    }
    (run_dir / "split.json").write_text(json.dumps(split_note, indent=2) + "\n")

    if args.method == "baseline":
        train_tf, eval_tf = baseline_train_tf(), baseline_eval_tf()
        batch_size, lr = 96, 2e-4
        train_ds = SkinDataset(train, transform=train_tf)
    elif args.method == "effb3":
        train_tf, eval_tf = build_train_transform_robust(), build_val_transform_robust()
        batch_size, lr = 32, 1e-4
        train_ds = SkinDataset(train, transform=train_tf)
    else:
        train_tf, eval_tf = build_train_transform_robust(), build_val_transform_robust()
        batch_size, lr = 32, 1e-4
        train_ds = DomainConstantDataset(train, transform=train_tf)

    val_ds = SkinDataset(val, transform=eval_tf)
    id_ds = SkinDataset(id_test, transform=eval_tf)
    ood_ds = SkinDataset(ood, transform=eval_tf)
    train_eval_ds = SkinDataset(train, transform=eval_tf)

    if args.dry_run:
        print(
            "dry_run",
            args.method,
            args.config,
            "train",
            len(train_ds),
            "val",
            len(val_ds),
            "id",
            len(id_ds),
            "ood",
            len(ood_ds),
            "keep",
            keep,
        )
        sample = train_ds[0]
        print("sample types", [type(x).__name__ for x in (sample if isinstance(sample, tuple) else (sample,))])
        (run_dir / "dry_run_ok.json").write_text(
            json.dumps({"ok": True, **split_note, "method": args.method, "seed": args.seed}, indent=2) + "\n"
        )
        return

    train_loader = make_loader(train_ds, batch_size, args.num_workers, True, drop_last=True)
    val_loader = make_loader(val_ds, batch_size, args.num_workers, False)

    if args.method == "baseline":
        lit = BaselineResNet50(
            num_classes=n_classes,
            learning_rate=lr,
            weight_decay=1e-4,
            label_smoothing=0.03,
            warmup_epochs=5,
            max_epochs=args.max_epochs,
        )
    elif args.method == "effb3":
        lit = EffB3SingleLightning(num_classes=n_classes, latent_dim=16, learning_rate=lr, weight_decay=1e-4)
    else:
        lit = CSGLiteLightning(
            lesion_classes=n_classes,
            domain_classes=2,
            lesion_latent_dim=16,
            context_latent_dim=64,
            learning_rate=lr,
            weight_decay=1e-4,
            lambda_ctx=1.0,
            lambda_adv=2.0,
            lambda_orth=1.0,
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
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else last
    print("best", best)
    if args.method == "baseline":
        lit = BaselineResNet50.load_from_checkpoint(str(best), strict=False)
        net = lit.net
    elif args.method == "effb3":
        lit = EffB3SingleLightning.load_from_checkpoint(str(best), strict=False)
        net = lit.net
    else:
        lit = CSGLiteLightning.load_from_checkpoint(str(best), strict=False)
        net = lit.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device).eval()

    id_loader = make_loader(id_ds, 64, args.num_workers, False)
    ood_loader = make_loader(ood_ds, 64, args.num_workers, False)
    train_eval_loader = make_loader(train_eval_ds, 64, args.num_workers, False)
    ptr = collect(args.method, net, train_eval_loader, device)
    pid = collect(args.method, net, id_loader, device)
    pood = collect(args.method, net, ood_loader, device)

    pred = pid["logits"].argmax(1)
    yid = pid["labels"]
    id_acc = float(accuracy_score(yid, pred))
    id_bal = float(balanced_accuracy_score(yid, pred))
    id_ece = ece(pid["logits"], yid)

    ood_scores = {}
    if args.method == "runB_orth1":
        ood_scores["z_lesion"] = maha_auroc(ptr["z_lesion"], ptr["labels"], pid["z_lesion"], pood["z_lesion"], n_classes)
        ood_scores["z_context"] = maha_auroc(ptr["z_context"], ptr["labels"], pid["z_context"], pood["z_context"], n_classes)
        ood_scores["z_lesion"]["supervision"] = "unsupervised"
        ood_scores["z_context"]["supervision"] = "unsupervised_here_no_PAD_domain_labels"
    else:
        ood_scores["backbone_raw"] = maha_auroc(
            ptr["backbone_raw"], ptr["labels"], pid["backbone_raw"], pood["backbone_raw"], n_classes
        )
        ood_scores["backbone_raw"]["supervision"] = "unsupervised"

    summary = {
        "run_name": run_name,
        "config": args.config,
        "method": args.method,
        "seed": args.seed,
        "n_classes": n_classes,
        "keep": keep,
        "hold_out": hold,
        "best_checkpoint": str(best),
        "id_acc": id_acc,
        "id_balanced_acc": id_bal,
        "id_ece": id_ece,
        "ood": ood_scores,
        "n_train": int(len(train)),
        "n_id": int(len(id_test)),
        "n_ood": int(len(ood)),
        "pad_involved": False,
        "prediction_P4": "z_lesion >= baseline on semantic OOD; z_context ≈ 0.50",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("saved", run_dir / "summary.json")
    print(json.dumps(ood_scores, indent=2))


if __name__ == "__main__":
    main()
