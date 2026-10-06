#!/usr/bin/env python3
"""Phase 2.5c: full 1-D axis distributions (ISIC / PAD / Fitzpatrick17k). CPU, no model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NPZ = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase1_6/domain_axis_projections.npz")
OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase2_5")
FIG = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/figures")


def desc(x):
    x = np.asarray(x, dtype=np.float64)
    q25, q75 = np.percentile(x, [25, 75])
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1) if len(x) > 1 else 0.0),
        "median": float(np.median(x)),
        "iqr": float(q75 - q25),
        "p05": float(np.percentile(x, 5)),
        "p25": float(q25),
        "p75": float(q75),
        "p95": float(np.percentile(x, 95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def overlap_coef(a, b, bins=80):
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    width = edges[1] - edges[0]
    # densities integrate to 1: overlap = ∫ min(f,g) dx
    return float(np.minimum(ha, hb).sum() * width)


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = np.sqrt((va + vb) / 2.0)
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    d = np.load(NPZ)
    isic, pad, fitz = d["isic"], d["pad"], d["fitz"]
    # PAD-positive axis from 1.6d
    lo, hi = float(min(isic.mean(), pad.mean())), float(max(isic.mean(), pad.mean()))
    outside_means = float(((fitz < lo) | (fitz > hi)).mean())
    span_lo, span_hi = float(min(isic.min(), pad.min())), float(max(isic.max(), pad.max()))
    outside_minmax = float(((fitz < span_lo) | (fitz > span_hi)).mean())
    # between the two empirical supports' inner gap vs overlap with each
    report = {
        "axis": "PCA k=1 of z_context, ISIC-train fit, PAD-positive",
        "n_fitz_images": int(len(fitz)),
        "download_note": "3887 / 16577 Fitzpatrick17k images (eval only; mixed copyright; not redistributed)",
        "isic": desc(isic),
        "pad": desc(pad),
        "fitzpatrick17k": desc(fitz),
        "pairwise": {
            "ISIC_Fitz": {"overlap_coef": overlap_coef(isic, fitz), "cohens_d": cohens_d(isic, fitz)},
            "Fitz_PAD": {"overlap_coef": overlap_coef(fitz, pad), "cohens_d": cohens_d(fitz, pad)},
            "ISIC_PAD": {"overlap_coef": overlap_coef(isic, pad), "cohens_d": cohens_d(isic, pad)},
        },
        "frac_fitz_outside_mean_interval": outside_means,
        "mean_interval": [lo, hi],
        "frac_fitz_outside_minmax_ISIC_PAD": outside_minmax,
        "minmax_interval": [span_lo, span_hi],
        "claim_check": (
            "Fitz spread overlaps PAD (std {:.1f} vs ISIC–PAD gap {:.1f}); means-only 'between' overstates tightness".format(
                desc(fitz)["std"], abs(pad.mean() - isic.mean())
            )
        ),
    }
    # tighten claim language
    ovl_fp = report["pairwise"]["Fitz_PAD"]["overlap_coef"]
    ovl_if = report["pairwise"]["ISIC_Fitz"]["overlap_coef"]
    if ovl_fp > 0.15:
        report["supported_claim"] = (
            "not on the ISIC side (Fitz overlaps PAD substantially; cannot claim a tight between-mode)"
        )
    else:
        report["supported_claim"] = "between ISIC and PAD, skewed toward PAD"
    (OUT / "axis_distributions.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    bins = 50
    ax.hist(isic, bins=bins, density=True, alpha=0.45, color="#0072B2", label="ISIC test")
    ax.hist(pad, bins=bins, density=True, alpha=0.45, color="#D55E00", label="PAD")
    ax.hist(fitz, bins=bins, density=True, alpha=0.45, color="#009E73", label="Fitzpatrick17k (n={})".format(len(fitz)))
    ax.axvline(isic.mean(), color="#0072B2", ls="--", lw=1)
    ax.axvline(pad.mean(), color="#D55E00", ls="--", lw=1)
    ax.axvline(fitz.mean(), color="#009E73", ls="--", lw=1)
    ax.set_xlabel("z_context PC1 (PAD-positive)")
    ax.set_ylabel("density")
    ax.set_title(
        "1-D domain axis (distributions, not means)\n"
        "OVL ISIC–Fitz={:.3f}  Fitz–PAD={:.3f}  ISIC–PAD={:.3f}".format(ovl_if, ovl_fp, report["pairwise"]["ISIC_PAD"]["overlap_coef"])
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / "domain_axis_distributions.{}".format(ext), dpi=300)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ["supported_claim", "pairwise", "frac_fitz_outside_mean_interval", "isic", "pad", "fitzpatrick17k"] if k in report}, indent=2))
    print("saved", FIG / "domain_axis_distributions.pdf")


if __name__ == "__main__":
    main()
