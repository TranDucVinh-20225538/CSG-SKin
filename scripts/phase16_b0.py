#!/usr/bin/env python3
"""Phase 16B0 (+ B0b): ERM on BCN+HAM, binary site probe on frozen features.

Preregistered: results/paperB/phase16/PREREGISTER_B.json
Outputs: results/paperB/phase16/b0/
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
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
REPO = Path("/data2/hpcshared/Vinh/CSG-Skin")
for p in (REPO, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import phase16_bcn_ham_data as bdata
from eval_trivial_ood import features_from_tensor, logistic_probe
from phase15_common import SingleDannNet, leakage_probe
from src.datasets.constants import IMAGENET_MEAN, IMAGENET_STD
from src.datasets.skin_dataset import SkinDataset
from src.utils.seed import seed_everything

OUT = ROOT / "results" / "paperB" / "phase16" / "b0"
CKPT = ROOT / "checkpoints" / "phase16" / "b0"
PREREG = ROOT / "results" / "paperB" / "phase16" / "PREREGISTER_B.json"


def build_train_tf(size: int):
    return transforms.Compose(
        [
            transforms.Resize(size + 64),
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
            transforms.ToTensor(),
            transforms.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD)),
        ]
    )


def build_eval_tf(size: int):
    return transforms.Compose(
        [
            transforms.Resize(size + 64),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD)),
        ]
    )


def build_handcrafted_tf(size: int):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])


def vignette_mask(h: int, w: int, radius_frac=0.45, device="cpu"):
    cy, cx = h / 2.0, w / 2.0
    r = radius_frac * min(h, w)
    yy = torch.arange(h, device=device, dtype=torch.float32).view(-1, 1).expand(h, w)
    xx = torch.arange(w, device=device, dtype=torch.float32).view(1, -1).expand(h, w)
    m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
    return m


class ErmLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, wd=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.net = SingleDannNet(n_classes=8, n_domains=2, pretrained=True)

    def training_step(self, batch, batch_idx):
        x, y, _site = batch
        logits, _, _ = self.net(x)
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _site = batch
        logits, _, _ = self.net(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.net.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)


@torch.no_grad()
def collect_z(net, loader, device, apply_mask=False):
    net.eval()
    z_list, site_list, y_list = [], [], []
    for images, labels, sites in loader:
        images = images.to(device)
        if apply_mask:
            m = vignette_mask(images.shape[2], images.shape[3], device=device)
            images = images * m.unsqueeze(0).unsqueeze(0)
        logits, _, z = net(images)
        z_list.append(z.cpu().numpy())
        site_list.append(sites.numpy())
        y_list.append(labels.numpy())
    return {
        "z": np.concatenate(z_list),
        "site": np.concatenate(site_list),
        "label": np.concatenate(y_list),
    }


class SiteDataset(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label_idx"]), int(row["site_id"])


def label_only_probe_reference(full_df, probe_df):
    """Majority-site-per-class rule (not logistic on label index)."""
    cohort = bdata.label_only_majority_reference(full_df)
    on_probe = bdata.label_only_majority_reference(full_df, eval_frame=probe_df)
    return {"cohort_rule": cohort, "on_probe_split": on_probe}


def handcrafted_site_probes(frame: pd.DataFrame, size: int, masked: bool, workers: int, limit=None):
    tf = build_handcrafted_tf(size)
    rows = []
    n = len(frame) if limit is None else min(limit, len(frame))
    for i in range(n):
        row = frame.iloc[i]
        t = tf(Image.open(row["path"]).convert("RGB"))
        if masked:
            m = vignette_mask(t.shape[1], t.shape[2], device="cpu")
            t = t * m.unsqueeze(0)
        feat = features_from_tensor(t)
        rows.append((feat, int(row["site_id"])))
    out = {}
    for key in ["rgb_moments", "color_hist", "gray_downsample", "hsv_moments", "laplacian_stats"]:
        x = np.stack([r[0][key] for r in rows])
        y = np.array([r[1] for r in rows], dtype=np.int64)
        z0, z1 = x[y == 0], x[y == 1]
        out[key] = logistic_probe(z0, z1)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--handcrafted_workers", type=int, default=8)
    p.add_argument("--handcrafted_limit", type=int, default=0, help="0 = full probe set")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "summary.json"
    if args.skip_done and summary_path.exists() and not args.dry_run:
        print("skip_done", summary_path)
        return

    prereg = json.loads(PREREG.read_text())
    df = bdata.load_bcn_ham_frame()
    train, val, probe = bdata.lesion_split(df, args.seed)
    split_doc = {
        "seed": args.seed,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_probe": int(len(probe)),
        "n_lesions_train": int(train["lesion_id"].nunique()),
        "n_lesions_probe": int(probe["lesion_id"].nunique()),
        "site_counts_train": train["site"].value_counts().to_dict(),
        "site_counts_probe": probe["site"].value_counts().to_dict(),
    }
    (OUT / "split.json").write_text(json.dumps(split_doc, indent=2) + "\n")

    cfg = {
        "phase": "16B0",
        "preregister": str(PREREG),
        "architecture": "single_encoder_erm_effb3 (SingleDannNet, uniform AdamW, no adv loss)",
        "preprocessing": {
            "train": f"Resize({args.image_size}+64), RandomResizedCrop({args.image_size}), robust aug, ImageNet norm",
            "eval": f"Resize({args.image_size}+64), CenterCrop({args.image_size}), ImageNet norm",
            "handcrafted": f"Resize(({args.image_size},{args.image_size})), ToTensor [0,1]",
            "vignette_mask": "circular radius 0.45*min(H,W), zero outside before embed",
        },
        **split_doc,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
    }
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    if args.dry_run:
        print("dry_run ok", split_doc)
        (OUT / "dry_run_ok.json").write_text(json.dumps({"ok": True, **cfg}, indent=2) + "\n")
        return

    train_ds = SiteDataset(train, build_train_tf(args.image_size))
    val_ds = SiteDataset(val, build_eval_tf(args.image_size))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    model = ErmLightning(lr=args.lr)
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(CKPT), filename="best-{epoch:02d}", monitor="val/acc", mode="max",
        save_top_k=1, save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_cb],
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        default_root_dir=str(OUT),
        log_every_n_steps=20,
    )
    trainer.fit(model, train_loader, val_loader)
    best = Path(ckpt_cb.best_model_path or CKPT / "last.ckpt")

    lit = ErmLightning.load_from_checkpoint(str(best), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = lit.net.to(device).eval()
    probe_loader = DataLoader(
        SiteDataset(probe, build_eval_tf(args.image_size)),
        batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    pack = collect_z(net, probe_loader, device, apply_mask=False)
    pack_mask = collect_z(net, probe_loader, device, apply_mask=True)

    site_probe = leakage_probe(pack["z"], pack["site"])
    site_probe_mask = leakage_probe(pack_mask["z"], pack_mask["site"])
    label_ref = label_only_probe_reference(df, probe)

    hc_limit = args.handcrafted_limit if args.handcrafted_limit > 0 else None
    print("handcrafted unmasked...", flush=True)
    hc = handcrafted_site_probes(probe, args.image_size, masked=False, workers=args.handcrafted_workers, limit=hc_limit)
    print("handcrafted masked...", flush=True)
    hc_mask = handcrafted_site_probes(probe, args.image_size, masked=True, workers=args.handcrafted_workers, limit=hc_limit)

    thr = float(prereg["B0_domain_signal_threshold"]["threshold"])
    bal = site_probe["bal_acc_mean"]
    if 0.45 <= bal <= 0.55:
        decision = "stop_no_B1_homogeneous"
    elif bal >= thr:
        decision = "proceed_B1"
    else:
        decision = "report_intermediate_signal"

    summary = {
        "run": "phase16_b0",
        "seed": args.seed,
        "best_checkpoint": str(best),
        "site_probe_frozen_z": site_probe,
        "site_probe_frozen_z_vignette_masked_input": site_probe_mask,
        "label_only_site_reference": label_ref,
        "handcrafted_site_unmasked": hc,
        "handcrafted_site_masked": hc_mask,
        "thresholds": prereg["B0_domain_signal_threshold"],
        "decision": decision,
        "config": cfg,
    }
    # fix id_task - compute cls acc on probe
    with torch.no_grad():
        logits = []
        for images, labels, _s in probe_loader:
            images = images.to(device)
            lg, _, _ = net(images)
            logits.append(lg.cpu())
        logits = torch.cat(logits).numpy()
    pred = logits.argmax(1)
    summary["probe_balanced_acc_8class"] = float(balanced_accuracy_score(pack["label"], pred))
    summary["probe_plain_acc_8class"] = float(accuracy_score(pack["label"], pred))

    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    ref = label_ref["cohort_rule"]
    ref_p = label_ref["on_probe_split"]
    verdict = [
        "# Phase 16 B0 verdict",
        "",
        f"- Site probe (frozen z, unmasked): **{site_probe['bal_acc_mean']:.3f}** bal acc (floor 0.5)",
        f"- Site probe (vignette-masked input): **{site_probe_mask['bal_acc_mean']:.3f}**",
        f"- Label-only majority-class reference (cohort): bal acc **{ref['bal_acc']:.3f}** (plain {ref['plain_acc']:.3f}; BCN rec {ref['bcn_recall']:.3f}, HAM rec {ref['ham_recall']:.3f})",
        f"- Same rule on B0 probe split: bal acc **{ref_p['bal_acc']:.3f}**",
        f"- Decision: **{decision}** (threshold {thr})",
        "",
    ]
    (OUT / "B0_VERDICT.md").write_text("\n".join(verdict))
    print("saved", summary_path, "decision", decision, "site_bal", site_probe["bal_acc_mean"], flush=True)


if __name__ == "__main__":
    main()
