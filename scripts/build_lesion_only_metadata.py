#!/usr/bin/env python3
# Build lesion-only preprocessed images + metadata for leakage-source verification.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import PROJECT_ROOT


DEFAULT_IN_METADATA = PROJECT_ROOT / "data" / "master_metadata.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "lesion_only_images"
DEFAULT_OUT_METADATA = PROJECT_ROOT / "data" / "master_metadata_lesion_only_soft.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Build lesion-only metadata from master_metadata.csv")
    p.add_argument("--in_metadata", type=Path, default=DEFAULT_IN_METADATA)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--out_metadata", type=Path, default=DEFAULT_OUT_METADATA)
    p.add_argument(
        "--preset",
        type=str,
        default="soft",
        choices=["soft", "aggressive"],
        help="soft keeps more lesion context; aggressive crops tighter.",
    )
    p.add_argument("--size", type=int, default=224, help="Final square size")
    p.add_argument("--margin", type=float, default=-1.0, help="Extra bbox margin ratio (-1 uses preset default)")
    p.add_argument(
        "--min_crop_ratio",
        type=float,
        default=-1.0,
        help="Minimum bbox width/height ratio wrt original image (-1 uses preset default).",
    )
    p.add_argument(
        "--autocontrast",
        action="store_true",
        help="Apply ImageOps.autocontrast on the lesion crop (disabled by default in soft preset).",
    )
    p.add_argument("--limit", type=int, default=0, help="Process first N rows (0 = all)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing processed images")
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


def _tight_bbox_from_diff(gray):
    border_med = _border_gray_median(gray)
    diff = np.abs(gray.astype(np.float32) - border_med).clip(0, 255).astype(np.uint8)
    thr = _otsu_threshold(diff)
    mask = diff > max(thr, 8)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        h, w = gray.shape
        # Fallback center crop (80%) when segmentation fails.
        y0, y1 = int(0.1 * h), int(0.9 * h)
        x0, x1 = int(0.1 * w), int(0.9 * w)
        return x0, y0, x1, y1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


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


def preprocess_one(src_path, dst_path, size, margin_ratio, min_crop_ratio, use_autocontrast):
    img = Image.open(src_path).convert("RGB")
    arr = np.array(img)
    gray = np.array(img.convert("L"))
    h, w = gray.shape

    x0, y0, x1, y1 = _tight_bbox_from_diff(gray)
    x0, y0, x1, y1 = _expand_bbox(x0, y0, x1, y1, w, h, margin_ratio)
    x0, y0, x1, y1 = _enforce_min_crop_ratio(x0, y0, x1, y1, w, h, min_crop_ratio)
    crop = arr[y0 : y1 + 1, x0 : x1 + 1]
    out = Image.fromarray(crop)
    if use_autocontrast:
        out = ImageOps.autocontrast(out)
    out = out.resize((size, size), Image.Resampling.BILINEAR)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst_path, quality=95)


def make_output_path(src_path, out_root):
    p = Path(src_path).resolve()
    name = p.stem + ".jpg"
    # Keep domain-ish folder hint by using parent dir name to avoid collisions.
    parent_tag = p.parent.name
    return out_root / parent_tag / name


def main():
    args = parse_args()
    if args.preset == "soft":
        margin = 0.30 if args.margin < 0 else args.margin
        min_crop_ratio = 0.70 if args.min_crop_ratio < 0 else args.min_crop_ratio
        use_autocontrast = args.autocontrast
    else:
        margin = 0.10 if args.margin < 0 else args.margin
        min_crop_ratio = 0.0 if args.min_crop_ratio < 0 else args.min_crop_ratio
        use_autocontrast = True

    print(
        "Preset={} | margin={:.2f} | min_crop_ratio={:.2f} | autocontrast={}".format(
            args.preset, margin, min_crop_ratio, use_autocontrast
        )
    )
    if not args.in_metadata.is_file():
        raise FileNotFoundError("Input metadata not found: {}".format(args.in_metadata))

    df = pd.read_csv(args.in_metadata)
    if "path" not in df.columns:
        raise ValueError("Input metadata must contain 'path' column.")
    df = df.copy()
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    total = len(df)
    if total == 0:
        raise ValueError("No rows to process.")

    n_ok = 0
    n_fail = 0
    out_paths = []

    print("Building lesion-only images for {} rows...".format(total))
    for i, row in enumerate(df.itertuples(index=False), start=1):
        src = Path(str(row.path))
        dst = make_output_path(src, args.out_dir)
        try:
            if args.overwrite or not dst.is_file():
                preprocess_one(
                    src,
                    dst,
                    size=args.size,
                    margin_ratio=margin,
                    min_crop_ratio=min_crop_ratio,
                    use_autocontrast=use_autocontrast,
                )
            out_paths.append(str(dst.resolve()))
            n_ok += 1
        except Exception:
            out_paths.append("")
            n_fail += 1

        if i % 500 == 0 or i == total:
            print("  {}/{} processed | ok={} fail={}".format(i, total, n_ok, n_fail))

    df["path_lesion_only"] = out_paths
    df = df[df["path_lesion_only"].astype(str) != ""].copy()
    df["path"] = df["path_lesion_only"]
    df = df.drop(columns=["path_lesion_only"])
    args.out_metadata.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_metadata, index=False)

    print("\nSaved lesion-only metadata: {}".format(args.out_metadata))
    print("Rows kept: {} / {} (failed: {})".format(len(df), total, n_fail))
    print("Use this for training: --metadata {}".format(args.out_metadata))


if __name__ == "__main__":
    main()
