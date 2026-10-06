#!/usr/bin/env python3
"""Phase 15.2 — non-adversarial invariance on the 15.1 single-encoder backbone.

CORAL / MMD: unlabeled alignment on pad_adv (PAD labels stay out of L_cls).
IRM / GroupDRO: require labeled training environments, so pad_adv labels enter
the objective. pad_heldout remains unseen. That protocol difference is recorded.

Coarse weight scan. Do not assume the GRL λ scale transfers.
Outputs under results/paperB/phase15/objectives/.
"""

from __future__ import annotations

import argparse
import json
import math
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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import train_phase2_pad_holdout as p2
from phase15_common import (
    SingleDannNet,
    assert_split_hygiene,
    coral_penalty,
    detector_block,
    ece_from_logits,
    irm_penalty,
    leakage_probe,
    mmd_rbf_penalty,
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

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
P2_SPLIT = PAPERB / "results" / "paperB" / "phase2_pad_holdout" / "pad_patient_split_paths.json"
OUT_ROOT = PAPERB / "results" / "paperB" / "phase15" / "objectives"
CKPT_ROOT = PAPERB / "checkpoints" / "phase15" / "objectives"

# Coarse scan. λ=0 is ERM and is shared (objective=erm).
OBJECTIVES = ("erm", "coral", "mmd", "irm", "groupdro")
WEIGHTS = {
    "erm": (0.0,),
    "coral": (1.0, 10.0, 100.0),
    "mmd": (1.0, 10.0, 100.0),
    "irm": (1.0, 10.0, 100.0),
    "groupdro": (0.1, 1.0, 10.0),
}
SEEDS = (42, 52, 62)

JOBS = []
for obj, ws in WEIGHTS.items():
    for w in ws:
        for s in SEEDS:
            JOBS.append((obj, w, s))
assert len(JOBS) == 3 + 4 * 3 * 3  # 39


def tag_w(x):
    return "{:g}".format(x).replace(".", "p")


class InvariantLightning(pl.LightningModule):
    def __init__(self, objective="coral", weight=1.0, learning_rate=1e-4, weight_decay=1e-4, groupdro_eta=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.objective = objective
        self.weight = float(weight)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.groupdro_eta = float(groupdro_eta)
        self.net = SingleDannNet(n_classes=8, n_domains=2, pretrained=True)
        self.register_buffer("q_group", torch.ones(2) / 2.0)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        images, _les, y_cls, y_true, domain = batch
        logits, _d, z = self.net(images)
        isic = domain == 0
        pad = domain == 1
        loss_cls_isic = F.cross_entropy(logits[isic], y_cls[isic]) if isic.any() else logits.new_tensor(0.0)
        # pad_adv true labels — used only by irm / groupdro
        loss_cls_pad = F.cross_entropy(logits[pad], y_true[pad]) if pad.any() else logits.new_tensor(0.0)

        penalty = logits.new_tensor(0.0)
        uses_pad_labels = False
        if self.objective == "coral" and isic.any() and pad.any():
            penalty = coral_penalty(z[isic], z[pad])
        elif self.objective == "mmd" and isic.any() and pad.any():
            penalty = mmd_rbf_penalty(z[isic], z[pad])
        elif self.objective == "irm":
            uses_pad_labels = True
            p0 = irm_penalty(logits[isic], y_true[isic]) if isic.any() else logits.new_tensor(0.0)
            p1 = irm_penalty(logits[pad], y_true[pad]) if pad.any() else logits.new_tensor(0.0)
            penalty = (p0 + p1) / 2.0
            loss_cls = (loss_cls_isic + loss_cls_pad) / 2.0
        elif self.objective == "groupdro":
            uses_pad_labels = True
            losses = torch.stack([loss_cls_isic.detach(), loss_cls_pad.detach()])
            self.q_group = self.q_group * torch.exp(self.groupdro_eta * losses)
            self.q_group = self.q_group / self.q_group.sum()
            loss_cls = self.q_group[0] * loss_cls_isic + self.q_group[1] * loss_cls_pad
        else:
            loss_cls = loss_cls_isic

        if self.objective in ("coral", "mmd", "erm"):
            loss_cls = loss_cls_isic

        loss = loss_cls + self.weight * penalty
        self.log("train/loss_cls", loss_cls, on_step=False, on_epoch=True)
        self.log("train/penalty", penalty, on_step=False, on_epoch=True)
        self.log("train/uses_pad_labels", float(uses_pad_labels), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits, _, _ = self.net(images)
        loss = F.cross_entropy(logits, targets)
        acc = (logits.argmax(1) == targets).float().mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", type=str, required=True, choices=list(OBJECTIVES))
    p.add_argument("--weight", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output_root", type=Path, default=OUT_ROOT)
    p.add_argument("--ckpt_root", type=Path, default=CKPT_ROOT)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    return p.parse_args()


@torch.no_grad()
def collect_all(net, loader, device):
    net.eval()
    z, lg, y = [], [], []
    for batch in loader:
        images, labels = batch[0], batch[1]
        images = images.to(device)
        logits, _, z_norm = net(images)
        z.append(z_norm.cpu().numpy())
        lg.append(logits.cpu().numpy())
        y.append(labels.numpy() if torch.is_tensor(labels) else np.asarray(labels))
    return {"z": np.concatenate(z), "logits": np.concatenate(lg), "labels": np.concatenate(y)}


def main():
    args = parse_args()
    if args.objective == "erm" and abs(args.weight) > 1e-12:
        raise SystemExit("erm must have weight 0")
    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    run_name = "{}_w{}_s{}".format(args.objective, tag_w(args.weight), args.seed)
    run_dir = args.output_root / run_name
    ckpt_dir = args.ckpt_root / run_name
    args.output_root.mkdir(parents=True, exist_ok=True)
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

    uses_pad_labels = args.objective in ("irm", "groupdro")
    cfg = {
        "run_name": run_name,
        "objective": args.objective,
        "weight": args.weight,
        "seed": args.seed,
        "uses_pad_adv_labels": uses_pad_labels,
        "pad_heldout_in_objective": False,
        "note": (
            "IRM/GroupDRO need labeled training environments; pad_adv labels enter L_cls. "
            "CORAL/MMD/ERM never use PAD labels. pad_heldout is unseen in every objective."
        ),
        "hygiene": hygiene,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    train_tf = build_train_transform_robust()
    les_tf = build_lesion_branch_transform_gray()
    eval_tf = build_val_transform_robust()
    train_ds = CombinedTrainDataset(train, pad_adv, transform=train_tf, lesion_transform=les_tf)
    val_ds = SkinDataset(val, transform=eval_tf)
    if args.dry_run:
        print("dry_run", run_name, "train", len(train_ds), "uses_pad_labels", uses_pad_labels)
        (run_dir / "dry_run_ok.json").write_text(json.dumps({"ok": True, **cfg}, indent=2) + "\n")
        return

    model = InvariantLightning(objective=args.objective, weight=args.weight, learning_rate=args.lr)
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir), filename="best-{epoch:02d}", monitor="val/acc", mode="max",
        save_top_k=1, save_last=True, auto_insert_metric_name=False,
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
    last = ckpt_dir / "last.ckpt"
    ckpt_path = str(last) if args.resume and last.exists() else None
    train_loader = p2.make_loader(train_ds, args.batch_size, args.num_workers, True, drop_last=True, collate=csg_lite_paired_collate)
    val_loader = p2.make_loader(val_ds, args.batch_size, args.num_workers, False)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else last

    lit = InvariantLightning.load_from_checkpoint(str(best), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = lit.net.to(device).eval()

    def loader(frame):
        return p2.make_loader(SkinDataset(frame, transform=eval_tf), 64, args.num_workers, False)

    train_pack = collect_all(net, loader(train), device)
    id_pack = collect_all(net, loader(test), device)
    pad_full = collect_all(net, loader(pad), device)
    pad_hold_p = collect_all(net, loader(pad_hold), device)
    pred = id_pack["logits"].argmax(1)
    yid = id_pack["labels"]
    id_ece, id_conf = ece_from_logits(id_pack["logits"], yid)
    hold_ece, hold_conf = ece_from_logits(pad_hold_p["logits"], pad_hold_p["labels"])
    domain_full = np.concatenate([np.zeros(len(id_pack["z"])), np.ones(len(pad_full["z"]))])
    leak = leakage_probe(np.concatenate([id_pack["z"], pad_full["z"]]), domain_full)
    ood = detector_block(
        train_pack["z"], train_pack["labels"], id_pack["z"], pad_full["z"],
        id_pack["logits"], pad_full["logits"], n_classes=8,
    )
    xfer = six_class_xfer(pad_hold_p["logits"], pad_hold_p["labels"])
    summary = {
        "run_name": run_name,
        "objective": args.objective,
        "weight": args.weight,
        "seed": args.seed,
        "uses_pad_adv_labels": uses_pad_labels,
        "id_balanced_acc": float(balanced_accuracy_score(yid, pred)),
        "id_acc": float(accuracy_score(yid, pred)),
        "id_ece": id_ece,
        "id_mean_confidence": id_conf,
        "ood_ece_pad_heldout": hold_ece,
        "ood_mean_confidence_pad_heldout": hold_conf,
        "leakage": leak,
        "ood_pad_full": ood,
        "xfer_pad_heldout_6class": xfer,
        "best_checkpoint": str(best),
        "plot_x": "achieved_leakage_bal_acc",
        "plot_y": "ood_auroc_maha",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("saved", run_dir / "summary.json", "leak", leak["bal_acc_mean"], "maha", ood["mahalanobis_classcond"])


if __name__ == "__main__":
    main()
