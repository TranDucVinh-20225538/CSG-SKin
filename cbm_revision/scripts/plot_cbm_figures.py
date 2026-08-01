#!/usr/bin/env python3
"""Plot CBM revision figures."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "cbm_revision"
FIG_DIR = RESULT_DIR / "figures"


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )


def _load(fp):
    return json.loads(Path(fp).read_text(encoding="utf-8"))


def plot_tradeoff():
    b = _load(RESULT_DIR / "baseline_soft_n5.json")["aggregate"]
    a = _load(RESULT_DIR / "runA_grl_n5.json")["aggregate"]
    r = _load(RESULT_DIR / "runB_orth1_n5.json")["aggregate"]
    rows = [
        ("Baseline Soft", b, "o", "#6c757d"),
        ("Run A / GRL", a, "s", "#1f77b4"),
        ("Run B (orth=1.0)", r, "D", "#2ca02c"),
    ]
    fig, ax = plt.subplots(figsize=(7, 5.4), dpi=160)
    for name, x, mk, c in rows:
        ax.errorbar(
            x["z_lesion_acc_mean"]["mean"],
            x["id_acc"]["mean"],
            xerr=x["z_lesion_acc_mean"]["std"],
            yerr=x["id_acc"]["std"],
            fmt=mk,
            markersize=9,
            capsize=4,
            color=c,
            label=name,
        )
    ax.set_xlabel("Leakage Probe Accuracy (lower is better)")
    ax.set_ylabel("ID Test Accuracy")
    ax.set_title("Utility-Leakage Trade-off (n=5)")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "utility_leakage_tradeoff_cbm.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "utility_leakage_tradeoff_cbm.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_seed_stability():
    objs = {
        "Baseline Soft": _load(RESULT_DIR / "baseline_soft_n5.json"),
        "Run A / GRL": _load(RESULT_DIR / "runA_grl_n5.json"),
        "Run B (orth=1.0)": _load(RESULT_DIR / "runB_orth1_n5.json"),
    }
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    x = np.arange(len(objs))
    means = [objs[k]["aggregate"]["id_balanced_acc"]["mean"] for k in objs]
    stds = [objs[k]["aggregate"]["id_balanced_acc"]["std"] for k in objs]
    ax.bar(x, means, yerr=stds, capsize=4, color=["#6c757d", "#1f77b4", "#2ca02c"], edgecolor="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(list(objs.keys()), rotation=10)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Seed Stability (n=5)")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "seed_stability_cbm.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "seed_stability_cbm.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ood_comparison():
    fp = RESULT_DIR / "ood_comparison.csv"
    if not fp.is_file():
        return
    df = pd.read_csv(fp)
    if df.empty:
        return
    pivot = df.pivot_table(index="ScoreType", columns="Method", values="AUROC", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("AUROC")
    ax.set_title("OOD Detector Comparison (z_lesion)")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ood_detector_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "ood_detector_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    plot_tradeoff()
    plot_seed_stability()
    plot_ood_comparison()
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()

