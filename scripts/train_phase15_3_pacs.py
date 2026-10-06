#!/usr/bin/env python3
"""Phase 15.3 — PACS leave-one-domain-out single-encoder DANN.

Standard 4 domains (photo, art_painting, cartoon, sketch), 7 classes.
Leakage is 3-class balanced accuracy on the training domains; floor 1/3.
OOD = held-out domain, never in the invariance objective.
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
from phase15_common import (
    GradientReversal,
    derm_domain_head,
    detector_block,
    ece_from_logits,
    grl_alpha,
    leakage_probe,
)
from src.utils.seed import seed_everything

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
OUT_ROOT = PAPERB / "results" / "paperB" / "phase15" / "pacs"
CKPT_ROOT = PAPERB / "checkpoints" / "phase15" / "pacs"
PACS_CANDIDATES = [
    Path("/data2/cmdir/home/toandq/data/PACS"),
    Path("/data2/cmdir/home/toandq/data/pacs"),
    Path("/data2/hpcshared/Vinh/data/PACS"),
    Path("/data2/cmdir/home/toandq/OpenMIBOOD/data/PACS"),
    PAPERB / "data" / "PACS",
]
DOMAINS = ("photo", "art_painting", "cartoon", "sketch")
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
LAMBDAS = (0.0, 0.25, 1.0, 2.0, 8.0)
SEEDS = (42, 52, 62)

JOBS = []
for held in DOMAINS:
    for lam in LAMBDAS:
        for s in SEEDS:
            JOBS.append((held, lam, s))
assert len(JOBS) == 60


def lam_tag(x):
    return "{:g}".format(x).replace(".", "p")


def _is_pacs(root: Path):
    if not root.is_dir():
        return False
    names = {c.name.lower() for c in root.iterdir() if c.is_dir()}
    return {"photo", "art_painting", "cartoon", "sketch"}.issubset(names)


def find_pacs_root(explicit=None):
    if explicit:
        p = Path(explicit)
        if _is_pacs(p):
            return p
        raise FileNotFoundError("PACS not found at {}".format(p))
    for p in PACS_CANDIDATES:
        if _is_pacs(p):
            return p
    return None


class PACSFolder(Dataset):
    def __init__(self, root, domains, transform):
        self.samples = []
        root = Path(root)
        want = set(domains)
        for d_i, dname in enumerate(DOMAINS):
            if dname not in want:
                continue
            for c_i, cname in enumerate(CLASSES):
                folder = root / dname / cname
                if not folder.is_dir():
                    alt = root / dname / cname.capitalize()
                    folder = alt if alt.is_dir() else folder
                if not folder.is_dir():
                    continue
                for img in sorted(folder.iterdir()):
                    if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        self.samples.append((str(img), c_i, d_i))
        self.transform = transform
        if not self.samples:
            raise RuntimeError("no PACS images for domains {}".format(domains))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y, d = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), int(y), int(d)


class PacsDann(nn.Module):
    def __init__(self, n_classes=7, n_domains=3, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
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
        return self.classifier(z_norm), self.domain_head(self.grl(z_norm)), z_norm

    def set_grl_lambda(self, lambd):
        self.grl.set_lambd(lambd)


class PacsLightning(pl.LightningModule):
    def __init__(self, n_domains=3, lambda_adv=0.0, learning_rate=1e-4, weight_decay=1e-4, adv_lr_multiplier=30.0, log_path=None):
        super().__init__()
        self.save_hyperparameters(ignore=["log_path"])
        self.net = PacsDann(n_domains=n_domains)
        self.lambda_adv = float(lambda_adv)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adv_lr_multiplier = float(adv_lr_multiplier)
        self.log_path = Path(log_path) if log_path else None
        self._last_alpha = 0.0
        self._g = []

    def training_step(self, batch, batch_idx):
        x, y, d = batch
        total = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        alpha, _ = grl_alpha(self.global_step, total)
        self.net.set_grl_lambda(alpha)
        self._last_alpha = alpha
        logits, dlog, z = self.net(x)
        loss_cls = F.cross_entropy(logits, y)
        loss_adv = F.cross_entropy(dlog, d)
        g = torch.autograd.grad(loss_adv, z, retain_graph=True, allow_unused=True)[0]
        g_norm = float(g.detach().float().norm()) if g is not None else 0.0
        self._g.append(g_norm)
        acc_adv = (dlog.argmax(1) == d).float().mean()
        self.log("train/loss_adv", loss_adv, on_step=False, on_epoch=True)
        self.log("train/acc_adv", acc_adv, on_step=False, on_epoch=True)
        self.log("train/g_enc_grl", g_norm, on_step=False, on_epoch=True)
        if self.current_epoch == 0 and batch_idx == 0 and self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps({
                    "event": "first_batch_epoch1",
                    "loss_adv": float(loss_adv.detach()),
                    "acc_adv": float(acc_adv.detach()),
                    "g_enc_grl": g_norm,
                    "lnK": math.log(3.0),
                }) + "\n")
        return loss_cls + self.lambda_adv * loss_adv

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics

        def _f(k, default=0.0):
            if k in metrics:
                v = metrics[k]
                return float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
            return float(default)

        row = {
            "epoch": int(self.current_epoch) + 1,
            "acc_adv": _f("train/acc_adv"),
            "loss_adv": _f("train/loss_adv"),
            "g_enc_grl_mean": float(np.mean(self._g)) if self._g else 0.0,
            "grl_alpha": float(self._last_alpha),
        }
        self._g = []
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        print(
            "[epoch {:02d}] acc_adv={:.4f} loss_adv={:.4f} g_enc={:.4g}".format(
                row["epoch"], row["acc_adv"], row["loss_adv"], row["g_enc_grl_mean"]
            ),
            flush=True,
        )

    def validation_step(self, batch, batch_idx):
        x, y, _d = batch
        logits, _, _ = self.net(x)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", F.cross_entropy(logits, y), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        adv = list(self.net.domain_head.parameters())
        adv_ids = {id(p) for p in adv}
        main = [p for p in self.net.parameters() if id(p) not in adv_ids]
        return torch.optim.AdamW(
            [
                {"params": main, "lr": self.learning_rate, "weight_decay": self.weight_decay},
                {"params": adv, "lr": self.learning_rate * self.adv_lr_multiplier, "weight_decay": self.weight_decay},
            ]
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--heldout_domain", type=str, required=True, choices=list(DOMAINS))
    p.add_argument("--lambda_adv", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data_root", type=Path, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output_root", type=Path, default=OUT_ROOT)
    p.add_argument("--ckpt_root", type=Path, default=CKPT_ROOT)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    return p.parse_args()


@torch.no_grad()
def collect(net, loader, device):
    net.eval()
    z, lg, y, d = [], [], [], []
    for x, yy, dd in loader:
        x = x.to(device)
        logits, _, zn = net(x)
        z.append(zn.cpu().numpy())
        lg.append(logits.cpu().numpy())
        y.append(yy.numpy())
        d.append(dd.numpy())
    return {
        "z": np.concatenate(z),
        "logits": np.concatenate(lg),
        "labels": np.concatenate(y),
        "domain": np.concatenate(d),
    }


class RemapDomain(Dataset):
    def __init__(self, base, mapping):
        self.base = base
        self.mapping = mapping

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y, d = self.base[idx]
        return x, y, int(self.mapping[DOMAINS[d]])


class Sub(Dataset):
    def __init__(self, base, keep, mapping=None):
        self.base = base
        self.keep = [i for i in range(len(base)) if i in keep]
        self.mapping = mapping

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, i):
        x, y, d = self.base[self.keep[i]]
        if self.mapping is not None:
            d = int(self.mapping[DOMAINS[d]])
        return x, y, d


def main():
    args = parse_args()
    root = find_pacs_root(args.data_root)
    if root is None:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        note = "# PACS data missing\n\nTried:\n" + "\n".join("- " + str(p) for p in PACS_CANDIDATES) + "\n"
        (OUT_ROOT / "DATA_MISSING.md").write_text(note)
        print(json.dumps({"status": "data_missing", "tried": [str(p) for p in PACS_CANDIDATES]}))
        if args.dry_run:
            return
        raise SystemExit("PACS data missing — not a scientific negative")

    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    train_domains = [d for d in DOMAINS if d != args.heldout_domain]
    dmap = {d: i for i, d in enumerate(train_domains)}
    run_name = "pacs_hold{}_ladv{}_s{}".format(args.heldout_domain, lam_tag(args.lambda_adv), args.seed)
    run_dir = args.output_root / run_name
    ckpt_dir = args.ckpt_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_done and (run_dir / "summary.json").exists() and not args.dry_run:
        print("skip_done", run_dir / "summary.json")
        return

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    raw_train = PACSFolder(root, train_domains, train_tf)
    raw_id = PACSFolder(root, train_domains, eval_tf)
    raw_ood = PACSFolder(root, [args.heldout_domain], eval_tf)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(raw_id))
    rng.shuffle(idx)
    n_val = max(int(0.2 * len(idx)), 1)
    val_idx = set(idx[:n_val].tolist())
    train_eval_idx = set(idx[n_val:].tolist())
    train_ds = RemapDomain(raw_train, dmap)
    val_ds = Sub(raw_id, val_idx, dmap)
    id_ds = Sub(raw_id, train_eval_idx, dmap)

    cfg = {
        "run_name": run_name,
        "heldout_domain": args.heldout_domain,
        "train_domains": train_domains,
        "lambda_adv": args.lambda_adv,
        "seed": args.seed,
        "n_train": len(train_ds),
        "n_id_eval": len(id_ds),
        "n_ood": len(raw_ood),
        "data_root": str(root),
        "leakage_floor": 1.0 / 3.0,
        "fit_ood_on": "all train-domain images (eval transform); 20% id split is val only",
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if args.dry_run:
        print("dry_run", cfg)
        (run_dir / "dry_run_ok.json").write_text(json.dumps({"ok": True, **cfg}, indent=2) + "\n")
        return

    adv_log = run_dir / "adversary_log.jsonl"
    model = PacsLightning(n_domains=3, lambda_adv=args.lambda_adv, learning_rate=args.lr, log_path=adv_log)
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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else ckpt_dir / "last.ckpt"
    lit = PacsLightning.load_from_checkpoint(str(best), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = lit.net.to(device).eval()

    fit_ds = RemapDomain(PACSFolder(root, train_domains, eval_tf), dmap)
    fit = collect(net, DataLoader(fit_ds, batch_size=64, shuffle=False, num_workers=args.num_workers), device)
    idp = collect(net, DataLoader(id_ds, batch_size=64, shuffle=False, num_workers=args.num_workers), device)
    oodp = collect(net, DataLoader(raw_ood, batch_size=64, shuffle=False, num_workers=args.num_workers), device)
    pred = idp["logits"].argmax(1)
    leak = leakage_probe(fit["z"], fit["domain"])
    ood = detector_block(fit["z"], fit["labels"], idp["z"], oodp["z"], idp["logits"], oodp["logits"], n_classes=7)
    id_ece, _id_conf = ece_from_logits(idp["logits"], idp["labels"])
    ood_ece, ood_conf = ece_from_logits(oodp["logits"], oodp["labels"])
    summary = {
        "run_name": run_name,
        "heldout_domain": args.heldout_domain,
        "train_domains": train_domains,
        "lambda_adv": args.lambda_adv,
        "seed": args.seed,
        "id_balanced_acc": float(balanced_accuracy_score(idp["labels"], pred)),
        "id_ece": id_ece,
        "ood_ece": ood_ece,
        "ood_mean_confidence": ood_conf,
        "ood_acc": float(accuracy_score(oodp["labels"], oodp["logits"].argmax(1))),
        "ood_balanced_acc": float(balanced_accuracy_score(oodp["labels"], oodp["logits"].argmax(1))),
        "leakage": leak,
        "ood_detectors": ood,
        "chance_floor_leakage": 1.0 / 3.0,
        "best_checkpoint": str(best),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("saved", run_dir / "summary.json", "leak", leak["bal_acc_mean"], "ood_maha", ood["mahalanobis_classcond"])


if __name__ == "__main__":
    main()
