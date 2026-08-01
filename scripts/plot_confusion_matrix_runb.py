#!/usr/bin/env python3
"""Generate confusion matrix heatmap for Run B (orth=1.0) on ISIC test set."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.constants import LABELS
from src.datasets.skin_dataset import SkinDataModule
from src.datasets.splits import build_id_ood_test_dataloaders
from src.models.csg_lightning import CSGLiteLightning
from src.utils import ood_metrics
from src.utils.paths import PROJECT_ROOT
from src.utils.seed import seed_everything


def configure_plot_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.2,
        }
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Load best Run B (orth=1.0) checkpoint and plot confusion matrix on ISIC test."
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "master_metadata_lesion_only_soft.csv",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "csg_lite" / "runB_orth1_s42",
        help=".ckpt file or checkpoint directory (newest .ckpt will be picked).",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures" / "fig_confusion_matrix.pdf",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def collect_preds_targets(model, loader, device):
    preds_all, targets_all = [], []
    model.eval()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits, _, _ = model(images)
        preds = torch.argmax(logits, dim=1)
        preds_all.append(preds.cpu())
        targets_all.append(targets.cpu())
    preds_np = torch.cat(preds_all, dim=0).numpy()
    targets_np = torch.cat(targets_all, dim=0).numpy()
    return preds_np, targets_np


def plot_confusion(cm, labels, out_pdf):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    cm = cm.astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    cm_norm = cm / row_sum * 100.0
    annot = np.empty_like(cm_norm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = "{:.1f}%\n({})".format(cm_norm[i, j], int(cm[i, j]))

    fig, ax = plt.subplots(figsize=(8.4, 7.1), dpi=160)
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        cbar_kws={"shrink": 0.85, "pad": 0.02, "label": "Row-normalized (%)"},
        vmin=0.0,
        vmax=100.0,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Normalized Confusion Matrix on ISIC Test Set")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    png_path = out_pdf.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return png_path


def main():
    args = parse_args()
    configure_plot_style()
    if not args.metadata.is_file():
        raise FileNotFoundError("Metadata not found: {}".format(args.metadata))
    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)

    ckpt = ood_metrics.find_checkpoint(args.ckpt)
    if ckpt is None:
        raise FileNotFoundError("Could not resolve checkpoint from: {}".format(args.ckpt))

    lit = CSGLiteLightning.load_from_checkpoint(str(ckpt), strict=False)
    model = lit.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dm = SkinDataModule(
        metadata_csv=args.metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()
    id_loader, _ood_loader = build_id_ood_test_dataloaders(dm)

    preds, targets = collect_preds_targets(model, id_loader, device)
    cm = confusion_matrix(targets, preds, labels=list(range(len(LABELS))))
    png_path = plot_confusion(cm, LABELS, args.output)

    acc = float((preds == targets).mean())
    print("Checkpoint: {}".format(ckpt))
    print("ISIC test samples: {}".format(int(targets.shape[0])))
    print("ISIC test accuracy: {:.4f}".format(acc))
    print("Saved: {}".format(args.output))
    print("Saved: {}".format(png_path))


if __name__ == "__main__":
    main()

