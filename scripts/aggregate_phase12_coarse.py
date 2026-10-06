#!/usr/bin/env python3
"""Aggregate Camelyon17 coarse λ scan and stop for dense-grid design."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase12/camelyon17")
REPORT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase12/PHASE12_1_COARSE.md")
COARSE = [(0.0, 42), (0.1, 42), (1.0, 42), (10.0, 42)]


def lam_tag(x):
    return "{:g}".format(float(x)).replace(".", "p")


def load_run(lam, seed):
    name = "dann_densenet121_ladv{}_s{}".format(lam_tag(lam), seed)
    p = OUT / "coarse" / name / "summary.json"
    if not p.exists():
        raise SystemExit("missing {}".format(p))
    return json.loads(p.read_text())


def main():
    rows = [load_run(lam, seed) for lam, seed in COARSE]
    primary = "mahalanobis_classcond_sharedcov"
    lines = []
    lines.append("# Phase 12.1 — Camelyon17 coarse λ scan")
    lines.append("")
    lines.append("Single seed (42). DenseNet-121 DANN. Checkpoint on `id_val_select` (n=16,780). ID metrics on `id_val_score` (n=16,780). Not leaderboard-comparable.")
    lines.append("")
    lines.append("**Camelyon17 is binary.** MSP lives in [0.5, 1] and is not the replication criterion. Lead with Mahalanobis (class-conditional, shared covariance) and kNN (k=50).")
    lines.append("")
    lines.append("Hospitals 1 (`val`) and 2 (`test`) were never in the adversary. Composition is 50/50 on train / id_val / test — no class-mix restriction needed.")
    lines.append("")
    lines.append("| λ_adv | ID acc (`id_val_score`) | leak 3-class bal acc | Maha AUROC h2 (test) | kNN AUROC h2 | Maha AUROC h1 (val) | kNN AUROC h1 | xfer acc h2 | xfer acc h1 | MSP h2 |")
    lines.append("|---:|---|---|---|---|---|---|---|---|---|")
    maha_h2 = []
    for r in rows:
        d2 = r["ood"]["test"]["detectors"]
        d1 = r["ood"]["val"]["detectors"]
        maha_h2.append((r["lambda_adv"], d2[primary]))
        lines.append(
            "| {:g} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                r["lambda_adv"],
                r["id_val_score"]["acc"],
                r["leakage_train_hospitals_3class"]["balanced_acc"],
                d2[primary],
                d2["knn_k50"],
                d1[primary],
                d1["knn_k50"],
                r["ood"]["test"]["accuracy"]["acc"],
                r["ood"]["val"]["accuracy"]["acc"],
                d2["MSP"],
            )
        )
    lines.append("")
    # locate transition
    m0 = maha_h2[0][1]
    m_small = maha_h2[1][1]
    collapsed_at_smallest = m_small <= 0.55
    still_high = m_small >= 0.70
    if collapsed_at_smallest:
        interval = (
            "Collapse already complete at the smallest nonzero λ (0.1): Maha h2 {:.3f} vs λ=0 {:.3f}. "
            "Dense grid must extend **downward**: include 0.05, 0.01, 0.001 until the drop begins. "
            "A cliff at λ=0.01 is a stronger result than one at 0.25."
        ).format(m_small, m0)
        rec = [0.0, 0.001, 0.01, 0.05, 0.1, 1.0]
    elif still_high and maha_h2[2][1] <= 0.55:
        interval = (
            "Transition sits between λ=0.1 (Maha h2 {:.3f}) and λ=1 (Maha h2 {:.3f}). "
            "Dense grid: points inside [0.1, 1] plus λ=0 and one value well above (10)."
        ).format(m_small, maha_h2[2][1])
        rec = [0.0, 0.1, 0.25, 0.5, 1.0, 10.0]
    elif still_high and maha_h2[3][1] <= 0.55:
        interval = (
            "Transition sits between λ=1 (Maha h2 {:.3f}) and λ=10 (Maha h2 {:.3f}). "
            "Dense grid: points inside [1, 10] plus λ=0 and 0.1 as a below-transition control."
        ).format(maha_h2[2][1], maha_h2[3][1])
        rec = [0.0, 0.1, 1.0, 2.0, 4.0, 10.0]
    else:
        interval = (
            "No collapse to ≈0.5 on this coarse grid (λ=0 {:.3f}, 0.1 {:.3f}, 1 {:.3f}, 10 {:.3f}). "
            "P12.1 tentatively FAILS on 1 seed. Dense grid is not authorised until this is confirmed "
            "or a wider λ range is proposed. Report as a finding; do not retune."
        ).format(m0, m_small, maha_h2[2][1], maha_h2[3][1])
        rec = None
    lines.append("## Where the transition sits")
    lines.append("")
    lines.append(interval)
    lines.append("")
    if rec:
        lines.append("Proposed dense λ grid (not launched): `{}`".format(rec))
        lines.append("")
    lines.append("**STOP.** Dense sweep is not started. Confirm the grid before using GPUs.")
    lines.append("")
    lines.append("Path verification (Amendment F): see `pathcheck.json`. Camelyon 0 missing / 455,954; iWildCam 0 missing / 203,029.")
    lines.append("")
    REPORT.write_text("\n".join(lines) + "\n")
    dump = {
        "maha_h2": maha_h2,
        "proposed_dense_grid": rec,
        "collapsed_at_smallest_nonzero": collapsed_at_smallest,
        "interval_text": interval,
    }
    (OUT / "coarse" / "aggregate.json").write_text(json.dumps(dump, indent=2) + "\n")
    print(interval)
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
