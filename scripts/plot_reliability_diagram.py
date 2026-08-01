#!/usr/bin/env python3
"""Plot reliability diagram (15 bins) for Baseline Soft vs Run B (orth=1.0)."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.skin_dataset import SkinDataModule
from src.datasets.splits import build_id_ood_test_dataloaders
from src.models.baseline import BaselineResNet50
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
        description="Create reliability diagram comparing Baseline Soft and Run B (orth=1.0)."
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "master_metadata_lesion_only_soft.csv",
    )
    p.add_argument(
        "--baseline_ckpt",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "baseline_soft",
        help=".ckpt file or dir.",
    )
    p.add_argument(
        "--runb_ckpt",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "csg_lite" / "runB_orth1_s42",
        help=".ckpt file or dir.",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--bins", type=int, default=15)
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures" / "fig_reliability_baseline_vs_runB.pdf",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _resolve_checkpoint_with_fallback(path_or_dir, keyword):
    """
    Resolve a checkpoint path.
    1) If path_or_dir is a file/dir, use existing helper.
    2) Otherwise, scan PROJECT_ROOT/checkpoints recursively and pick newest *.ckpt
       whose path contains `keyword` (case-insensitive).
    """
    direct = ood_metrics.find_checkpoint(path_or_dir)
    if direct is not None:
        return direct

    ckpt_root = PROJECT_ROOT / "checkpoints"
    if not ckpt_root.is_dir():
        return None

    keyword = str(keyword).lower().strip()
    candidates = []
    for p in ckpt_root.rglob("*.ckpt"):
        if keyword and keyword not in str(p).lower():
            continue
        candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.stat().st_mtime)


@torch.no_grad()
def collect_logits_targets(model, loader, device, model_type):
    logits_all, targets_all = [], []
    model.eval()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        if model_type == "csg":
            logits, _, _ = model(images)
        else:
            logits = model(images)
        logits_all.append(logits.cpu())
        targets_all.append(targets.cpu())
    logits_np = torch.cat(logits_all, dim=0).numpy()
    targets_np = torch.cat(targets_all, dim=0).numpy()
    return logits_np, targets_np


def calibration_curve_from_logits(logits, labels, n_bins=15):
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc, bin_conf, bin_count, bin_ci = [], [], [], []
    ece = 0.0
    n = max(len(labels), 1)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        c = int(mask.sum())
        if c == 0:
            bin_acc.append(np.nan)
            bin_conf.append((lo + hi) / 2.0)
            bin_count.append(0)
            bin_ci.append(np.nan)
            continue
        acc_i = float(correct[mask].mean())
        conf_i = float(conf[mask].mean())
        ci_i = 1.96 * np.sqrt(max(acc_i * (1.0 - acc_i), 1e-8) / c)
        bin_acc.append(acc_i)
        bin_conf.append(conf_i)
        bin_count.append(c)
        bin_ci.append(float(ci_i))
        ece += (c / n) * abs(acc_i - conf_i)

    return {
        "bin_acc": np.asarray(bin_acc, dtype=np.float64),
        "bin_conf": np.asarray(bin_conf, dtype=np.float64),
        "bin_count": np.asarray(bin_count, dtype=np.int64),
        "bin_ci": np.asarray(bin_ci, dtype=np.float64),
        "ece": float(ece),
    }


def plot_reliability(cal_base, cal_runb, out_pdf):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 6.3), dpi=160)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#b0b0b0", linewidth=1.6, label="Perfect calibration")

    m0 = ~np.isnan(cal_base["bin_acc"])
    m1 = ~np.isnan(cal_runb["bin_acc"])
    ax.plot(
        cal_base["bin_conf"][m0],
        cal_base["bin_acc"][m0],
        marker="o",
        linewidth=1.8,
        markersize=5,
        color="#6c757d",
        label="Baseline Soft (ECE={:.3f})".format(cal_base["ece"]),
    )
    ax.plot(
        cal_runb["bin_conf"][m1],
        cal_runb["bin_acc"][m1],
        marker="s",
        linewidth=2.5,
        markersize=6,
        color="#1f77b4",
        label="Run B (orth=1.0) (ECE={:.3f})".format(cal_runb["ece"]),
    )
    ax.fill_between(
        cal_base["bin_conf"][m0],
        np.clip(cal_base["bin_acc"][m0] - cal_base["bin_ci"][m0], 0.0, 1.0),
        np.clip(cal_base["bin_acc"][m0] + cal_base["bin_ci"][m0], 0.0, 1.0),
        color="#6c757d",
        alpha=0.12,
    )
    ax.fill_between(
        cal_runb["bin_conf"][m1],
        np.clip(cal_runb["bin_acc"][m1] - cal_runb["bin_ci"][m1], 0.0, 1.0),
        np.clip(cal_runb["bin_acc"][m1] + cal_runb["bin_ci"][m1], 0.0, 1.0),
        color="#1f77b4",
        alpha=0.14,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Calibration on ISIC Test Set")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right")
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

    base_ckpt = _resolve_checkpoint_with_fallback(args.baseline_ckpt, keyword="baseline")
    runb_ckpt = _resolve_checkpoint_with_fallback(args.runb_ckpt, keyword="runb")
    if base_ckpt is None:
        raise FileNotFoundError(
            "Could not resolve baseline checkpoint from: {}. "
            "Tried direct path/dir and fallback scan in {}/checkpoints/**/*.ckpt containing keyword 'baseline'.".format(
                args.baseline_ckpt, PROJECT_ROOT
            )
        )
    if runb_ckpt is None:
        raise FileNotFoundError(
            "Could not resolve runB checkpoint from: {}. "
            "Tried direct path/dir and fallback scan in {}/checkpoints/**/*.ckpt containing keyword 'runb'.".format(
                args.runb_ckpt, PROJECT_ROOT
            )
        )

    base_lit = BaselineResNet50.load_from_checkpoint(str(base_ckpt))
    runb_lit = CSGLiteLightning.load_from_checkpoint(str(runb_ckpt), strict=False)
    base_model = base_lit.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    runb_model = runb_lit.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = SkinDataModule(
        metadata_csv=args.metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()
    id_loader, _ood_loader = build_id_ood_test_dataloaders(dm)

    logits_base, y_base = collect_logits_targets(base_model, id_loader, device, model_type="baseline")
    logits_runb, y_runb = collect_logits_targets(runb_model, id_loader, device, model_type="csg")
    if y_base.shape[0] != y_runb.shape[0]:
        raise RuntimeError("ID test size mismatch between models.")

    cal_base = calibration_curve_from_logits(logits_base, y_base, n_bins=args.bins)
    cal_runb = calibration_curve_from_logits(logits_runb, y_runb, n_bins=args.bins)
    png_path = plot_reliability(cal_base, cal_runb, args.output)

    print("Baseline checkpoint: {}".format(base_ckpt))
    print("Run B checkpoint: {}".format(runb_ckpt))
    print("Baseline ECE ({} bins): {:.4f}".format(args.bins, cal_base["ece"]))
    print("Run B ECE ({} bins): {:.4f}".format(args.bins, cal_runb["ece"]))
    print("Saved: {}".format(args.output))
    print("Saved: {}".format(png_path))


if __name__ == "__main__":
    main()

