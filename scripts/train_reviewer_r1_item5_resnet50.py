#!/usr/bin/env python3
"""Reviewer R1 item 5 — single-encoder DANN, ResNet-50, λ∈{0,0.5,2} × 3 seeds."""

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
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import train_phase2_pad_holdout as p2
from phase15_common import (
    LN2,
    GradientReversal,
    assert_split_hygiene,
    detector_block,
    derm_domain_head,
    ece_from_logits,
    grl_alpha,
    leakage_probe,
    six_class_xfer,
)
from src.datasets.skin_dataset import (
    CombinedTrainDataset,
    SkinDataset,
    build_lesion_branch_transform_gray,
    build_train_transform_robust,
    build_val_transform_robust,
    csg_lite_paired_collate,
)
from src.utils.seed import seed_everything
from train_phase15_1_single_dann import collect_all, gate_from_log

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
P2_SPLIT = PAPERB / "results" / "paperB" / "phase2_pad_holdout" / "pad_patient_split_paths.json"
OUT_ROOT = PAPERB / "results" / "paperB" / "reviewer_r1" / "resnet50_single_dann"
CKPT_ROOT = PAPERB / "checkpoints" / "reviewer_r1" / "resnet50_single_dann"


class ResNet50Dann(nn.Module):
    def __init__(self, n_classes=8, n_domains=2, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet50(weights=weights)
        self.feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.feat_bn = nn.BatchNorm1d(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, n_classes)
        self.grl = GradientReversal(1.0)
        self.domain_head = derm_domain_head(self.feat_dim, n_domains)

    def forward(self, x):
        z = self.backbone(x)
        z_norm = self.feat_bn(z)
        logits = self.classifier(z_norm)
        dlogits = self.domain_head(self.grl(z_norm))
        return logits, dlogits, z_norm

    def set_grl_lambda(self, lambd):
        self.grl.set_lambd(lambd)


class ResNet50DannLightning(pl.LightningModule):
    def __init__(
        self,
        n_classes=8,
        n_domains=2,
        learning_rate=1e-4,
        weight_decay=1e-4,
        lambda_adv=0.0,
        adv_lr_multiplier=30.0,
        cls_lr_multiplier=0.2,
        log_path=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["log_path"])
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_adv = float(lambda_adv)
        self.adv_lr_multiplier = float(adv_lr_multiplier)
        self.cls_lr_multiplier = float(cls_lr_multiplier)
        self.net = ResNet50Dann(n_classes=n_classes, n_domains=n_domains, pretrained=True)
        self._last_alpha = 0.0
        self._g_enc = []
        self._epoch_rows = []
        self.log_path = Path(log_path) if log_path else None
        self.automatic_optimization = True

    def forward(self, x):
        return self.net(x)

    def _set_alpha(self):
        total = 0
        try:
            total = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        except RuntimeError:
            total = 0
        alpha, progress = grl_alpha(self.global_step, total)
        self.net.set_grl_lambda(alpha)
        self._last_alpha = alpha
        return alpha, progress

    def training_step(self, batch, _batch_idx):
        alpha, progress = self._set_alpha()
        images_ctx, _les, y, _ys, domain = batch
        logits, dlogits, z_norm = self.net(images_ctx)
        loss_cls = F.cross_entropy(logits, y, ignore_index=-1)
        loss_adv = F.cross_entropy(dlogits, domain)
        loss = loss_cls + self.lambda_adv * loss_adv
        with torch.no_grad():
            adv_pred = dlogits.argmax(1)
            adv_acc = (adv_pred == domain).float().mean()
        self.log("train/loss", loss, prog_bar=True, batch_size=images_ctx.size(0))
        self.log("train/adv_acc", adv_acc, prog_bar=True, batch_size=images.size(0))
        if self.log_path and self.global_step % 20 == 0:
            row = {
                "step": int(self.global_step),
                "epoch": int(self.current_epoch),
                "alpha": float(alpha),
                "grl_progress": float(progress),
                "adv_acc": float(adv_acc),
                "adv_ce": float(loss_adv),
            }
            with self.log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return loss

    def validation_step(self, batch, _batch_idx):
        images, labels = batch
        logits, _, _ = self.net(images)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(1)
        acc = (preds == labels).float().mean()
        self.log("val/loss", loss, prog_bar=True, batch_size=images.size(0))
        self.log("val/acc", acc, prog_bar=True, batch_size=images.size(0))

    def configure_optimizers(self):
        adv = list(self.net.domain_head.parameters()) + list(self.net.grl.parameters())
        cls = list(self.net.classifier.parameters())
        adv_ids = {id(p) for p in adv}
        cls_ids = {id(p) for p in cls}
        main = [p for p in self.net.parameters() if id(p) not in adv_ids and id(p) not in cls_ids]
        return torch.optim.AdamW(
            [
                {"params": main, "lr": self.learning_rate, "weight_decay": self.weight_decay},
                {"params": adv, "lr": self.learning_rate * self.adv_lr_multiplier, "weight_decay": self.weight_decay},
                {"params": cls, "lr": self.learning_rate * self.cls_lr_multiplier, "weight_decay": self.weight_decay},
            ]
        )


def lam_tag(x):
    return "{:g}".format(float(x)).replace(".", "p")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambda_adv", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    run_name = "resnet50_single_dann_ladv{}_s{}".format(lam_tag(args.lambda_adv), args.seed)
    run_dir = OUT_ROOT / run_name
    ckpt_dir = CKPT_ROOT / run_name
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_done and (run_dir / "summary.json").exists() and not args.dry_run:
        print("skip_done", run_dir / "summary.json")
        return

    spec = json.loads(P2_SPLIT.read_text())
    df = p2.load_master()
    isic = df[df.domain == "isic"].copy()
    pad = df[df.domain == "pad_ufes"].copy()
    train, val, test = p2.isic_splits(isic)
    pad_adv = pad[pad.path.isin(set(spec["pad_adv_paths"]))].reset_index(drop=True)
    pad_hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    hygiene = assert_split_hygiene(train.path, val.path, test.path, pad_adv.path, pad_hold.path)

    train_tf = build_train_transform_robust()
    les_tf = build_lesion_branch_transform_gray()
    eval_tf = build_val_transform_robust()
    train_ds = CombinedTrainDataset(train, pad_adv, transform=train_tf, lesion_transform=les_tf)
    val_ds = SkinDataset(val, transform=eval_tf)

    cfg = {
        "run_name": run_name,
        "lambda_adv": args.lambda_adv,
        "seed": args.seed,
        "architecture": "single_encoder_dann_resnet50",
        "adv_lr_multiplier": 30.0,
        "reviewer_item": 5,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    if args.dry_run:
        print("dry_run", run_name)
        return

    adv_log = run_dir / "adversary_log.jsonl"
    if adv_log.exists() and not args.resume:
        adv_log.write_text("")
    model = ResNet50DannLightning(lambda_adv=args.lambda_adv, log_path=adv_log)
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
    train_loader = p2.make_loader(
        train_ds, args.batch_size, args.num_workers, True, drop_last=True, collate=csg_lite_paired_collate
    )
    val_loader = p2.make_loader(val_ds, args.batch_size, args.num_workers, False)
    last = ckpt_dir / "last.ckpt"
    ckpt_path = str(last) if args.resume and last.exists() else None
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else last

    lit = ResNet50DannLightning.load_from_checkpoint(str(best), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = lit.net.to(device).eval()

    def loader(frame):
        return p2.make_loader(SkinDataset(frame, transform=eval_tf), 64, args.num_workers, False)

    train_pack = collect_all(net, loader(train), device)
    id_pack = collect_all(net, loader(test), device)
    pad_hold_p = collect_all(net, loader(pad_hold), device)

    yid = id_pack["labels"]
    pred = id_pack["logits"].argmax(1)
    hold_ece, hold_conf = ece_from_logits(pad_hold_p["logits"], pad_hold_p["labels"])
    domain_hold = np.concatenate([np.zeros(len(id_pack["z"])), np.ones(len(pad_hold_p["z"]))])
    leak_hold = leakage_probe(np.concatenate([id_pack["z"], pad_hold_p["z"]]), domain_hold)

    ood_det = detector_block(
        train_pack["z"],
        train_pack["labels"],
        id_pack["z"],
        pad_hold_p["z"],
        id_pack["logits"],
        pad_hold_p["logits"],
        n_classes=8,
    )
    rows = []
    if adv_log.exists():
        for line in adv_log.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    gate = gate_from_log(rows, leak_hold["bal_acc_mean"], floor=0.5, lambda_adv=args.lambda_adv)

    summary = {
        "run_name": run_name,
        "seed": args.seed,
        "lambda_adv": args.lambda_adv,
        "architecture": "single_encoder_dann_resnet50",
        "best_checkpoint": str(best),
        "id_balanced_acc": float(balanced_accuracy_score(yid, pred)),
        "ood_ece_pad_heldout": hold_ece,
        "leakage_pad_heldout": leak_hold,
        "ood_pad_heldout": ood_det,
        "gate": gate,
        "hygiene": hygiene,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    maha = ood_det["mahalanobis_classcond"]
    print("saved", run_dir / "summary.json", "maha_pad_heldout", maha)


if __name__ == "__main__":
    main()
