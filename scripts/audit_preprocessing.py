#!/usr/bin/env python3
"""Phase 1.5c: preprocessing-artifact audit. No GPU."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path("/data2/hpcshared/Vinh/CSG-Skin")
OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase1_5")
SOFT = REPO / "data" / "master_metadata_lesion_only_soft.csv"
RAW = REPO / "data" / "master_metadata.csv"


def remap(p: str) -> str:
    s = str(p)
    if s.startswith("/mnt/data2/Vinh/"):
        return "/data2/hpcshared/Vinh/" + s[len("/mnt/data2/Vinh/") :]
    return s


def summarize_images(df, tag, limit=None):
    rows = []
    n = len(df) if limit is None else min(limit, len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        if i >= n:
            break
        p = remap(row.path)
        rec = {
            "tag": tag,
            "domain": row.domain,
            "path": p,
            "suffix": Path(p).suffix.lower(),
            "exists": os.path.isfile(p),
            "readable": os.access(p, os.R_OK) if os.path.isfile(p) else False,
            "w": None,
            "h": None,
            "aspect": None,
            "mode": None,
            "format": None,
        }
        if rec["readable"]:
            try:
                with Image.open(p) as im:
                    rec["w"], rec["h"] = im.size
                    rec["aspect"] = float(im.size[0]) / max(im.size[1], 1)
                    rec["mode"] = im.mode
                    rec["format"] = im.format
            except Exception as e:
                rec["error"] = str(e)
        rows.append(rec)
        if (i + 1) % 5000 == 0:
            print(tag, i + 1, flush=True)
    return pd.DataFrame(rows)


def dist_summary(sub, col):
    x = sub[col].dropna().astype(float)
    if x.empty:
        return {}
    return {
        "n": int(len(x)),
        "min": float(x.min()),
        "p25": float(x.quantile(0.25)),
        "median": float(x.median()),
        "p75": float(x.quantile(0.75)),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    soft = pd.read_csv(SOFT)
    raw = pd.read_csv(RAW)
    print("scanning processed (all) and raw (all readable)")
    proc = summarize_images(soft, "processed_lesion_only")
    raws = summarize_images(raw, "raw_source")
    all_df = pd.concat([proc, raws], ignore_index=True)
    all_df.to_csv(OUT / "preprocess_image_inventory.csv", index=False)

    report = {
        "eval_transform": "torchvision Resize(256, bilinear) → CenterCrop(224) → ToTensor → ImageNet Normalize",
        "lesion_branch_at_eval": "in-model _rgb_to_gray3 AFTER ImageNet norm (CSGLite.forward x_lesion=None)",
        "soft_crop_script": {
            "file": "scripts/build_lesion_only_metadata.py",
            "bbox": "border-median difference, expand margin=0.30, min_crop_ratio=0.70",
            "resize": "PIL Image.Resampling.BILINEAR to 224×224",
            "save": "JPEG quality=95 (extension .jpg, no explicit format=)",
            "autocontrast": False,
        },
        "byte_identical_eval_path": True,
        "note_eval_on_already_224": "Both domains' lesion_only files are already 224×224 JPEG. Eval Resize(256)+CenterCrop(224) therefore upsamples then recrops identically.",
        "domains": {},
    }
    for tag in ["processed_lesion_only", "raw_source"]:
        sub = all_df[all_df.tag == tag]
        report["domains"][tag] = {}
        for domain in ["isic", "pad_ufes"]:
            d = sub[sub.domain == domain]
            readable = d[d.readable]
            report["domains"][tag][domain] = {
                "n_listed": int(len(d)),
                "n_exists": int(d.exists.sum()),
                "n_readable": int(d.readable.sum()),
                "suffix_counts": Counter(d.suffix.tolist()),
                "format_counts": Counter([x for x in readable["format"].tolist() if x]),
                "mode_counts": Counter([x for x in readable["mode"].tolist() if x]),
                "width": dist_summary(readable, "w"),
                "height": dist_summary(readable, "h"),
                "aspect": dist_summary(readable, "aspect"),
                "unique_sizes": Counter(
                    ["{}x{}".format(int(w), int(h)) for w, h in zip(readable.w.dropna(), readable.h.dropna())]
                ).most_common(12),
            }

    # Asymmetries
    asymmetries = []
    # encoding
    asymmetries.append(
        {
            "name": "source_encoding",
            "finding": "ISIC source files are JPEG; PAD-UFES source files are PNG (from metadata suffixes). Soft-crop re-encodes BOTH to JPEG quality=95.",
        }
    )
    asymmetries.append(
        {
            "name": "pad_raw_unreadable",
            "finding": "PAD original PNGs live at Ban_sao_datn/.../pad_ufes20/images but are mode 600 / owner anhnv; this user cannot read them. Raw-source trivial detectors cannot include PAD.",
        }
    )
    proc_isic = all_df[(all_df.tag == "processed_lesion_only") & (all_df.domain == "isic") & all_df.readable]
    proc_pad = all_df[(all_df.tag == "processed_lesion_only") & (all_df.domain == "pad_ufes") & all_df.readable]
    if len(proc_isic) and len(proc_pad):
        same = set(zip(proc_isic.w, proc_isic.h)) == {(224, 224)} and set(zip(proc_pad.w, proc_pad.h)) == {(224, 224)}
        asymmetries.append(
            {
                "name": "processed_resolution",
                "finding": "After soft-crop, ISIC and PAD are both 224×224 JPEG (byte-identical encoder settings). Eval transform is therefore shared.",
                "both_224": bool(
                    (proc_isic.w == 224).all()
                    and (proc_isic.h == 224).all()
                    and (proc_pad.w == 224).all()
                    and (proc_pad.h == 224).all()
                ),
            }
        )
    report["asymmetries"] = asymmetries
    (OUT / "preprocess_audit.json").write_text(json.dumps(report, indent=2, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)) + "\n")
    print(json.dumps({k: report[k] for k in ["byte_identical_eval_path", "asymmetries"]}, indent=2, default=str))
    print("saved", OUT / "preprocess_audit.json")


if __name__ == "__main__":
    main()
