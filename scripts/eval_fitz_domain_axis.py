#!/usr/bin/env python3
"""Phase 1.6d: project Fitzpatrick17k onto the ISIC/PAD z_context k=1 axis. CPU ok."""

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import eval_ood_dual_branch as p1

from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase1_6")
FIG = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/figures")
IMG_DIR = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/data/fitzpatrick17k/images")
CSV = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase11_third_domain/fitzpatrick17k.csv")
FEAT_Z = OUT / "features" / "z_context.npz"


class FolderJpegDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        im = Image.open(p).convert("RGB")
        return self.transform(im), 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


@torch.no_grad()
def extract_z_context(model, loader, device, max_batches=None):
    model.eval()
    chunks = []
    for i, (images, _y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device)
        z = model.context_projector(model.extract_context_features(images))
        chunks.append(z.cpu().numpy())
    return np.concatenate(chunks) if chunks else np.zeros((0, 64), np.float32)


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("device", device)

    if not FEAT_Z.exists():
        raise SystemExit("Need {} from 1.6a first (ISIC/PAD z_context).".format(FEAT_Z))
    pack = np.load(FEAT_Z)
    ztr, zid, zood = pack["train"], pack["id"], pack["ood"]

    pca = PCA(n_components=1, random_state=42)
    pca.fit(ztr)
    id1 = pca.transform(zid)[:, 0]
    ood1 = pca.transform(zood)[:, 0]
    sign = 1.0
    if ood1.mean() < id1.mean():
        sign = -1.0
        id1, ood1 = -id1, -ood1
    # PAD side is positive

    paths = sorted(IMG_DIR.glob("*.jpg"))
    if args.dry_run:
        paths = paths[:64]
    print("fitz images", len(paths))
    if not paths:
        (OUT / "fitz_axis.json").write_text(
            json.dumps({"blocked": True, "reason": "no Fitzpatrick17k images downloaded yet", "n": 0}, indent=2) + "\n"
        )
        print("no images")
        return

    ckpt = p1.find_policy_ckpt(p1.method_ckpt_dir("runB_orth1", 42))
    model, _ = p1.load_model("runB_orth1", ckpt, device)
    tf = build_val_transform_robust()
    ds = FolderJpegDataset(paths, tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    zf = extract_z_context(model, loader, device, max_batches=2 if args.dry_run else None)
    f1 = sign * pca.transform(zf)[:, 0]

    def desc(x, name):
        return {
            "name": name,
            "n": int(len(x)),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "median": float(np.median(x)),
            "p05": float(np.percentile(x, 5)),
            "p95": float(np.percentile(x, 95)),
        }

    # where does fitz sit
    isic_mu, pad_mu = float(id1.mean()), float(ood1.mean())
    fitz_mu = float(f1.mean())
    mid = 0.5 * (isic_mu + pad_mu)
    if abs(fitz_mu - isic_mu) < abs(fitz_mu - pad_mu) and abs(fitz_mu - isic_mu) < abs(fitz_mu - mid):
        land = "ISIC_side"
    elif abs(fitz_mu - pad_mu) < abs(fitz_mu - isic_mu) and abs(fitz_mu - pad_mu) < abs(fitz_mu - mid):
        land = "PAD_side"
    else:
        land = "between"
    # stronger rule: which mode is closer
    land = "PAD_side" if abs(fitz_mu - pad_mu) < abs(fitz_mu - isic_mu) else "ISIC_side"
    if min(isic_mu, pad_mu) < fitz_mu < max(isic_mu, pad_mu):
        land = "between_or_" + land

    stats = {
        "axis": "PCA k=1 of z_context, fit ISIC train, signed so PAD mean > ISIC mean",
        "prediction_P5prime": "Fitzpatrick17k lands on the PAD side, or between, not on the ISIC side",
        "land": land,
        "isic": desc(id1, "ISIC_test"),
        "pad": desc(ood1, "PAD"),
        "fitzpatrick17k": desc(f1, "Fitzpatrick17k"),
        "n_fitz_images": int(len(paths)),
        "licence": "eval only; Groh et al. CC BY-NC-SA 3.0 annotations; images web-scraped, do not redistribute",
    }
    (OUT / "fitz_axis.json").write_text(json.dumps(stats, indent=2) + "\n")
    np.savez_compressed(OUT / "domain_axis_projections.npz", isic=id1, pad=ood1, fitz=f1)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bins = 40
    ax.hist(id1, bins=bins, density=True, alpha=0.45, color="#0072B2", label="ISIC test (ID)")
    ax.hist(ood1, bins=bins, density=True, alpha=0.45, color="#D55E00", label="PAD (OOD)")
    ax.hist(f1, bins=bins, density=True, alpha=0.45, color="#009E73", label="Fitzpatrick17k (unseen)")
    ax.axvline(isic_mu, color="#0072B2", ls="--", lw=1)
    ax.axvline(pad_mu, color="#D55E00", ls="--", lw=1)
    ax.axvline(fitz_mu, color="#009E73", ls="--", lw=1)
    ax.set_xlabel("z_context PC1 (PAD-positive)")
    ax.set_ylabel("density")
    ax.set_title("Unseen-domain projection on the 1-D context axis\n(eval only; Fitzpatrick17k images not redistributed)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / "domain_axis_isic_pad_fitz.{}".format(ext), dpi=300)
    plt.close(fig)
    print("land", land, "fitz_mean", fitz_mu, "isic", isic_mu, "pad", pad_mu)
    print("saved", FIG / "domain_axis_isic_pad_fitz.pdf")


if __name__ == "__main__":
    main()
