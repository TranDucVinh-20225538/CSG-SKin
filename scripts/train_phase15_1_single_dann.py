#!/usr/bin/env python3
"""Phase 15.1 — plain single-encoder DANN on ISIC↔PAD.

One EffNet-B3, linear classifier, GRL + derm domain head.
No context branch, no orthogonality. Same split / optimiser / adv_lr×30 / GRL ramp
as the dual-encoder Phase 3 runs.

Outputs under results/paperB/phase15/single_dann/. Nothing existing is modified.
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
from torch.utils.data import DataLoader

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import train_phase2_pad_holdout as p2
from phase15_common import (
    LN2,
    SingleDannNet,
    assert_split_hygiene,
    detector_block,
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

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
P2_SPLIT = PAPERB / "results" / "paperB" / "phase2_pad_holdout" / "pad_patient_split_paths.json"
OUT_ROOT = PAPERB / "results" / "paperB" / "phase15" / "single_dann"
CKPT_ROOT = PAPERB / "checkpoints" / "phase15" / "single_dann"

JOBS = []
for _lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    _seeds = (42, 52, 62, 72, 82) if _lam in (0.0, 2.0) else (42, 52, 62)
    for _s in _seeds:
        JOBS.append((_lam, _s))
assert len(JOBS) == 25


def lam_tag(x):
    return "{:g}".format(x).replace(".", "p")


class SingleDannLightning(pl.LightningModule):
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
        self.net = SingleDannNet(n_classes=n_classes, n_domains=n_domains, pretrained=True)
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

    def training_step(self, batch, batch_idx):
        images_ctx, _les, y, _ys, domain = batch
        alpha, progress = self._set_alpha()
        logits, dlog, z = self.net(images_ctx)
        loss_cls = F.cross_entropy(logits, y, ignore_index=-1)
        loss_adv = F.cross_entropy(dlog, domain)
        g = torch.autograd.grad(loss_adv, z, retain_graph=True, allow_unused=True)[0]
        g_norm = float(g.detach().float().norm()) if g is not None else 0.0
        self._g_enc.append(g_norm)
        loss = loss_cls + self.lambda_adv * loss_adv
        with torch.no_grad():
            acc_adv = (dlog.argmax(1) == domain).float().mean()
            isic = y >= 0
            acc_cls = (logits[isic].argmax(1) == y[isic]).float().mean() if isic.any() else logits.new_tensor(0.0)
        self.log("train/loss_cls", loss_cls, on_step=False, on_epoch=True)
        self.log("train/loss_adv", loss_adv, on_step=False, on_epoch=True)
        self.log("train/acc_adv", acc_adv, on_step=False, on_epoch=True)
        self.log("train/acc_cls", acc_cls, on_step=False, on_epoch=True)
        self.log("train/grl_a", alpha, on_step=False, on_epoch=True)
        self.log("train/g_enc_grl", g_norm, on_step=False, on_epoch=True)
        self.log("train/grl_progress", progress, on_step=True, on_epoch=False)
        if self.current_epoch == 0 and batch_idx == 0:
            rec = {
                "event": "first_batch_epoch1",
                "loss_adv": float(loss_adv.detach()),
                "acc_adv": float(acc_adv.detach()),
                "g_enc_grl": g_norm,
                "lnK": LN2,
                "grl_alpha": alpha,
            }
            if self.log_path:
                with self.log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            print("GATE first_batch", rec, flush=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits, _, _ = self.net(images)
        loss = F.cross_entropy(logits, targets)
        acc = (logits.argmax(1) == targets).float().mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics

        def _f(key, default=0.0):
            if key in metrics:
                v = metrics[key]
                return float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
            return float(default)

        row = {
            "epoch": int(self.current_epoch) + 1,
            "acc_adv": _f("train/acc_adv"),
            "loss_adv": _f("train/loss_adv"),
            "acc_cls": _f("train/acc_cls"),
            "grl_alpha": float(self._last_alpha),
            "g_enc_grl_mean": float(np.mean(self._g_enc)) if self._g_enc else 0.0,
            "lnK": LN2,
        }
        self._epoch_rows.append(row)
        self._g_enc = []
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        print(
            "[epoch {:02d}] acc_adv={:.4f} loss_adv={:.4f} acc_cls={:.4f} grl_a={:.4f} g_enc={:.4g}".format(
                row["epoch"], row["acc_adv"], row["loss_adv"], row["acc_cls"], row["grl_alpha"], row["g_enc_grl_mean"]
            ),
            flush=True,
        )

    def configure_optimizers(self):
        adv = list(self.net.domain_head.parameters())
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambda_adv", type=float, required=True)
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


def gate_from_log(rows, leakage_bal, floor=0.5, lambda_adv=0.0):
    from phase15_common import adversary_gate
    return adversary_gate(rows, LN2, lambda_adv, leakage_bal=leakage_bal, floor=floor)


def main():
    args = parse_args()
    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    run_name = "single_dann_ladv{}_s{}".format(lam_tag(args.lambda_adv), args.seed)
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

    train_tf = build_train_transform_robust()
    les_tf = build_lesion_branch_transform_gray()
    eval_tf = build_val_transform_robust()
    train_ds = CombinedTrainDataset(train, pad_adv, transform=train_tf, lesion_transform=les_tf)
    val_ds = SkinDataset(val, transform=eval_tf)

    cfg = {
        "run_name": run_name,
        "lambda_adv": args.lambda_adv,
        "seed": args.seed,
        "architecture": "single_encoder_dann_effb3",
        "no_context_branch": True,
        "no_orthogonality": True,
        "adv_lr_multiplier": 30.0,
        "cls_lr_multiplier": 0.2,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "n_pad_adv": int(len(pad_adv)),
        "n_pad_heldout": int(len(pad_hold)),
        "phase2_split": str(P2_SPLIT),
        "hygiene": hygiene,
        "leakage_floor": 0.5,
        "fit_ood_on": "ISIC train only",
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    if args.dry_run:
        print("dry_run", run_name, "train", len(train_ds), "val", len(val_ds), "pad_adv", len(pad_adv), "pad_hold", len(pad_hold))
        (run_dir / "dry_run_ok.json").write_text(json.dumps({"ok": True, **cfg}, indent=2) + "\n")
        return

    adv_log = run_dir / "adversary_log.jsonl"
    if adv_log.exists() and not args.resume:
        adv_log.write_text("")
    model = SingleDannLightning(
        learning_rate=args.lr,
        lambda_adv=args.lambda_adv,
        log_path=adv_log,
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
    last = ckpt_dir / "last.ckpt"
    ckpt_path = str(last) if args.resume and last.exists() else None
    train_loader = p2.make_loader(train_ds, args.batch_size, args.num_workers, True, drop_last=True, collate=csg_lite_paired_collate)
    val_loader = p2.make_loader(val_ds, args.batch_size, args.num_workers, False)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else last
    print("best", best, flush=True)

    lit = SingleDannLightning.load_from_checkpoint(str(best), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = lit.net.to(device).eval()

    def loader(frame):
        return p2.make_loader(SkinDataset(frame, transform=eval_tf), 64, args.num_workers, False)

    train_pack = collect_all(net, loader(train), device)
    id_pack = collect_all(net, loader(test), device)
    pad_full = collect_all(net, loader(pad), device)
    pad_adv_p = collect_all(net, loader(pad_adv), device)
    pad_hold_p = collect_all(net, loader(pad_hold), device)

    pred = id_pack["logits"].argmax(1)
    yid = id_pack["labels"]
    id_ece, id_conf = ece_from_logits(id_pack["logits"], yid)
    hold_ece, hold_conf = ece_from_logits(pad_hold_p["logits"], pad_hold_p["labels"])

    domain_full = np.concatenate([np.zeros(len(id_pack["z"])), np.ones(len(pad_full["z"]))])
    domain_hold = np.concatenate([np.zeros(len(id_pack["z"])), np.ones(len(pad_hold_p["z"]))])
    leak_full = leakage_probe(np.concatenate([id_pack["z"], pad_full["z"]]), domain_full)
    leak_hold = leakage_probe(np.concatenate([id_pack["z"], pad_hold_p["z"]]), domain_hold)

    ood = {}
    for name, pack in (("pad_full", pad_full), ("pad_heldout", pad_hold_p), ("pad_adv", pad_adv_p)):
        ood[name] = detector_block(
            train_pack["z"], train_pack["labels"], id_pack["z"], pack["z"],
            id_pack["logits"], pack["logits"], n_classes=8,
        )
        ood[name]["n_ood"] = int(len(pack["z"]))

    xfer = six_class_xfer(pad_hold_p["logits"], pad_hold_p["labels"])

    rows = []
    if adv_log.exists():
        for line in adv_log.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    gate = gate_from_log(rows, leak_full["bal_acc_mean"], floor=0.5, lambda_adv=args.lambda_adv)

    summary = {
        "run_name": run_name,
        "seed": args.seed,
        "lambda_adv": args.lambda_adv,
        "architecture": "single_encoder_dann_effb3",
        "best_checkpoint": str(best),
        "id_acc": float(accuracy_score(yid, pred)),
        "id_balanced_acc": float(balanced_accuracy_score(yid, pred)),
        "id_ece": id_ece,
        "id_mean_confidence": id_conf,
        "ood_ece_pad_heldout": hold_ece,
        "ood_mean_confidence_pad_heldout": hold_conf,
        "leakage": {
            "pad_full": leak_full,
            "pad_heldout": leak_hold,
            "primary": "pad_full matches Phase 3 locked protocol; floor 0.5",
        },
        "ood": ood,
        "xfer_pad_heldout_6class": xfer,
        "gate": gate,
        "hygiene": hygiene,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("saved", run_dir / "summary.json")
    print("GATE", gate["status"], gate["reason"], "leak", leak_full["bal_acc_mean"])


if __name__ == "__main__":
    main()
