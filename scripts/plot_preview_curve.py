#!/usr/bin/env python3
"""Phase 1.5d: correlational leakage vs OOD AUROC preview (4 heterogeneous points)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/figures")
# Specified preview numbers (addendum). EffB3 Maha from results/effb3_control (16-d z), not Phase 1 1536-d.
POINTS = [
    {"name": "z_context", "leakage": 0.9997, "auroc": 0.9999, "bal": None, "color": "#0072B2"},
    {"name": "ResNet-50 baseline", "leakage": 0.979, "auroc": 0.868, "bal": 0.655, "color": "#D55E00"},
    {"name": "EffB3 single control", "leakage": 0.802, "auroc": 0.726, "bal": 0.683, "color": "#009E73"},
    {"name": "z_lesion", "leakage": 0.724, "auroc": 0.409, "bal": 0.702, "color": "#CC79A7"},
]


def bootstrap_slope(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    slopes = []
    intercepts = []
    for _ in range(n):
        idx = rng.integers(0, len(x), size=len(x))
        xb, yb = x[idx], y[idx]
        if np.std(xb) < 1e-12:
            continue
        a, b = np.polyfit(xb, yb, 1)
        slopes.append(a)
        intercepts.append(b)
    slopes = np.asarray(slopes)
    intercepts = np.asarray(intercepts)
    return {
        "slope_mean": float(slopes.mean()),
        "slope_ci95": [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))],
        "intercept_mean": float(intercepts.mean()),
        "n_boot": int(len(slopes)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.array([p["leakage"] for p in POINTS], dtype=np.float64)
    y = np.array([p["auroc"] for p in POINTS], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    boot = bootstrap_slope(x, y)
    # colourblind-safe
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for p in POINTS:
        ax.scatter(p["leakage"], p["auroc"], s=70, color=p["color"], zorder=3,
                   label=("z_context (AUROC > 0.9999)" if p["name"] == "z_context" else p["name"]))
    xs = np.linspace(0.68, 1.01, 50)
    ax.plot(xs, slope * xs + intercept, color="#000000", lw=1.2, ls="--", label="OLS fit (n=4)")
    ax.axhline(0.5, color="#888888", lw=0.8, ls=":")
    ax.set_xlabel("Domain-probe accuracy (leakage)")
    ax.set_ylabel("Mahalanobis OOD AUROC (ISIC test vs PAD)")
    ax.set_title("Preview: leakage vs covariate-shift AUROC\n(correlational across heterogeneous models — not an intervention)")
    ax.set_xlim(0.68, 1.02)
    ax.set_ylim(0.35, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"preview_leakage_vs_ood.{ext}", dpi=300)
    plt.close(fig)
    stats = {
        "disclaimer": "Correlational across heterogeneous models (architecture, dim, supervision). Phase 3 λ_adv sweep is the intervention.",
        "n_points": 4,
        "pearson_r": r,
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
        "bootstrap": boot,
        "points": POINTS,
    }
    Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase1_5/preview_curve_stats.json").write_text(
        __import__("json").dumps(stats, indent=2) + "\n"
    )
    print("r=", r, "slope=", slope, "boot", boot)
    print("saved", OUT / "preview_leakage_vs_ood.pdf")


if __name__ == "__main__":
    main()
