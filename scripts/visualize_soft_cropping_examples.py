#!/usr/bin/env python3
"""Visualize soft preprocessing examples: original, mask, and final crop."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import PROJECT_ROOT


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
        description="Take random images and visualize soft preprocessing (Otsu + margin=0.3)."
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "master_metadata.csv",
    )
    p.add_argument("--n_samples", type=int, default=2)
    p.add_argument("--candidate_pool", type=int, default=24, help="Random candidates to pick best-looking examples from.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--margin", type=float, default=0.30)
    p.add_argument("--min_crop_ratio", type=float, default=0.70)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures" / "fig_soft_crop_examples.pdf",
    )
    return p.parse_args()


def _otsu_threshold(values_u8):
    hist = np.bincount(values_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b2 = ((mu_t * omega - mu) ** 2) / denom
    return int(np.argmax(sigma_b2))


def _border_gray_median(gray):
    h, w = gray.shape
    bh = max(1, int(h * 0.05))
    bw = max(1, int(w * 0.05))
    border = np.concatenate(
        [
            gray[:bh, :].ravel(),
            gray[-bh:, :].ravel(),
            gray[:, :bw].ravel(),
            gray[:, -bw:].ravel(),
        ]
    )
    return float(np.median(border))


def _tight_bbox_and_mask(gray):
    border_med = _border_gray_median(gray)
    diff = np.abs(gray.astype(np.float32) - border_med).clip(0, 255).astype(np.uint8)
    thr = _otsu_threshold(diff)
    mask = diff > max(thr, 8)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        h, w = gray.shape
        y0, y1 = int(0.1 * h), int(0.9 * h)
        x0, x1 = int(0.1 * w), int(0.9 * w)
        return (x0, y0, x1, y1), mask
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), mask


def _expand_bbox(x0, y0, x1, y1, w, h, margin_ratio):
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    mx = int(round(bw * margin_ratio))
    my = int(round(bh * margin_ratio))
    nx0 = max(0, x0 - mx)
    ny0 = max(0, y0 - my)
    nx1 = min(w - 1, x1 + mx)
    ny1 = min(h - 1, y1 + my)
    return nx0, ny0, nx1, ny1


def _enforce_min_crop_ratio(x0, y0, x1, y1, w, h, min_crop_ratio):
    if min_crop_ratio <= 0:
        return x0, y0, x1, y1
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    target_w = int(round(w * min_crop_ratio))
    target_h = int(round(h * min_crop_ratio))
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    nw = max(bw, target_w)
    nh = max(bh, target_h)
    nx0 = max(0, cx - nw // 2)
    ny0 = max(0, cy - nh // 2)
    nx1 = min(w - 1, nx0 + nw - 1)
    ny1 = min(h - 1, ny0 + nh - 1)
    nx0 = max(0, nx1 - nw + 1)
    ny0 = max(0, ny1 - nh + 1)
    return nx0, ny0, nx1, ny1


def run_soft_crop(pil_img, margin, min_crop_ratio, out_size):
    arr = np.array(pil_img.convert("RGB"))
    gray = np.array(pil_img.convert("L"))
    h, w = gray.shape
    (x0, y0, x1, y1), mask = _tight_bbox_and_mask(gray)
    x0, y0, x1, y1 = _expand_bbox(x0, y0, x1, y1, w, h, margin_ratio=margin)
    x0, y0, x1, y1 = _enforce_min_crop_ratio(x0, y0, x1, y1, w, h, min_crop_ratio=min_crop_ratio)
    crop = arr[y0 : y1 + 1, x0 : x1 + 1]
    crop_resized = Image.fromarray(crop).resize((out_size, out_size), Image.Resampling.BILINEAR)
    bbox = (x0, y0, x1, y1)
    mask_ratio = float(mask.mean())
    return arr, mask, np.array(crop_resized), bbox, mask_ratio


def _example_score(mask_ratio):
    # Favor non-trivial and non-overgrown masks (reviewer-friendly visuals).
    target = 0.22
    return abs(mask_ratio - target)


def draw_panel(rows, out_pdf):
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12.5, 3.9 * n), dpi=160)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, row in enumerate(rows):
        ax0, ax1, ax2 = axes[i]
        orig = row["orig"]
        mask = row["mask"]
        crop = row["crop"]
        x0, y0, x1, y1 = row["bbox"]

        ax0.imshow(orig)
        ax0.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="lime", linewidth=1.8)
        ax0.set_title("Original")
        ax0.axis("off")

        ax1.imshow(mask, cmap="gray")
        ax1.set_title("Mask (Otsu)")
        ax1.axis("off")

        ax2.imshow(crop)
        ax2.set_title("Final crop")
        ax2.axis("off")

        # Visual flow arrows between stages.
        ax0.annotate(
            "",
            xy=(1.05, 0.5),
            xycoords="axes fraction",
            xytext=(0.95, 0.5),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=2.0, color="#333333"),
            annotation_clip=False,
        )
        ax1.annotate(
            "",
            xy=(1.05, 0.5),
            xycoords="axes fraction",
            xytext=(0.95, 0.5),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=2.0, color="#333333"),
            annotation_clip=False,
        )

    fig.suptitle("Soft Lesion-Centered Preprocessing Examples")
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
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

    df = pd.read_csv(args.metadata)
    if "path" not in df.columns:
        raise ValueError("Metadata must contain 'path' column.")
    df = df[df["path"].map(lambda p: Path(str(p)).is_file())].copy()
    if df.empty:
        raise ValueError("No existing image paths in metadata.")

    n = min(int(args.n_samples), len(df))
    if n <= 0:
        raise ValueError("n_samples must be > 0")

    pool_n = min(max(int(args.candidate_pool), n), len(df))
    picked = df.sample(n=pool_n, random_state=args.seed).reset_index(drop=True)
    rows = []
    for _, row in picked.iterrows():
        path = Path(str(row["path"]))
        pil = Image.open(path).convert("RGB")
        orig, mask, crop, bbox, mask_ratio = run_soft_crop(
            pil,
            margin=args.margin,
            min_crop_ratio=args.min_crop_ratio,
            out_size=args.crop_size,
        )
        rows.append(
            {
                "path": str(path),
                "orig": orig,
                "mask": mask,
                "crop": crop,
                "bbox": bbox,
                "mask_ratio": mask_ratio,
                "score": _example_score(mask_ratio),
            }
        )

    rows = sorted(rows, key=lambda r: r["score"])[:n]

    png_path = draw_panel(rows, args.output)
    print("Using {} selected samples from {} candidates (metadata: {})".format(n, pool_n, args.metadata))
    for i, row in enumerate(rows, start=1):
        print("  {}. {} | mask_ratio={:.3f}".format(i, row["path"], row["mask_ratio"]))
    print("Saved: {}".format(args.output))
    print("Saved: {}".format(png_path))


if __name__ == "__main__":
    main()

