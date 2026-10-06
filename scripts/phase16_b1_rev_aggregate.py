#!/usr/bin/env python3
"""Aggregate B1-rev coarse (seed 42) and optional dense 3-seed at λ∈{0,0.25,1}."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase16/b1_rev")
COARSE_LAM = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8]
DENSE_LAM = [0, 0.25, 1]
DENSE_SEEDS = [42, 52, 62]


def tag(l):
    return "{:g}".format(l).replace(".", "p")


def load_summary(lam, seed=42):
    p = ROOT / f"ham_bcn_ladv{tag(lam)}_s{seed}" / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def coarse_table():
    rows = []
    for lam in COARSE_LAM:
        o = load_summary(lam)
        if not o:
            continue
        d = o["ood_detectors_bcn_heldout"]
        m = o["ood_auroc_maha"]
        rows.append(
            {
                "lambda": lam,
                "leakage_bal": o["leakage_bcn_heldout_protocol"]["bal_acc_mean"],
                "maha_unrestricted": d["mahalanobis_classcond"],
                "maha_shared": m["shared_classes"],
                "maha_reweighted": m["reweighted_to_ood_class_mix"],
                "knn50": d["knn_k50"],
                "cosine": d["cosine_max"],
                "id_ham_test_bal": o["id_ham_test_balanced_acc"],
                "ood_bcn_hold_bal": o["ood_bcn_heldout_balanced_acc"],
            }
        )
    return rows


def dense_table():
    out = []
    for lam in DENSE_LAM:
        leak, knn, cos, maha = [], [], [], []
        for s in DENSE_SEEDS:
            o = load_summary(lam, s)
            if not o:
                continue
            d = o["ood_detectors_bcn_heldout"]
            leak.append(o["leakage_bcn_heldout_protocol"]["bal_acc_mean"])
            knn.append(d["knn_k50"])
            cos.append(d["cosine_max"])
            maha.append(d["mahalanobis_classcond"])
        if len(leak) < len(DENSE_SEEDS):
            continue
        out.append(
            {
                "lambda": lam,
                "leakage_mean": float(np.mean(leak)),
                "leakage_std": float(np.std(leak, ddof=1)),
                "knn50_mean": float(np.mean(knn)),
                "knn50_std": float(np.std(knn, ddof=1)),
                "cosine_mean": float(np.mean(cos)),
                "cosine_std": float(np.std(cos, ddof=1)),
                "maha_mean": float(np.mean(maha)),
                "maha_std": float(np.std(maha, ddof=1)),
            }
        )
    return out


def main():
    coarse = coarse_table()
    dense = dense_table()
    doc = {
        "coarse_seed42": coarse,
        "coarse_missing_lambdas": [l for l in COARSE_LAM if load_summary(l) is None],
        "dense3seed": dense,
        "dense_missing": len(dense) < len(DENSE_LAM),
    }
    out_json = ROOT / "b1_rev_aggregate.json"
    out_json.write_text(json.dumps(doc, indent=2) + "\n")

    md = ["# B1-rev aggregate (auto-generated)", ""]
    if coarse:
        md.append("## Coarse scan (seed 42, ID=HAM, OOD=BCN heldout)")
        md.append("")
        md.append("| λ | leakage | Maha | Maha rew | kNN-50 | cosine |")
        md.append("|---|--------:|-----:|---------:|-------:|-------:|")
        for r in coarse:
            md.append(
                f"| {r['lambda']:g} | {r['leakage_bal']:.3f} | {r['maha_unrestricted']:.3f} | "
                f"{r['maha_reweighted']:.3f} | {r['knn50']:.3f} | {r['cosine']:.3f} |"
            )
    else:
        md.append("_Coarse: no summaries yet._")
    md.append("")
    if dense:
        md.append("## Dense 3-seed @ λ∈{0, 0.25, 1}")
        md.append("")
        md.append("| λ | leakage | kNN-50 | cosine | Maha |")
        md.append("|---|---------|--------|--------|------|")
        for r in dense:
            md.append(
                f"| {r['lambda']:g} | {r['leakage_mean']:.3f}±{r['leakage_std']:.3f} | "
                f"{r['knn50_mean']:.3f}±{r['knn50_std']:.3f} | "
                f"{r['cosine_mean']:.3f}±{r['cosine_std']:.3f} | "
                f"{r['maha_mean']:.3f}±{r['maha_std']:.3f} |"
            )
    else:
        md.append("_Dense 3-seed: incomplete._")
    (ROOT / "B1_REV_AGGREGATE.md").write_text("\n".join(md) + "\n")
    print("coarse rows", len(coarse), "dense rows", len(dense), "missing λ", doc["coarse_missing_lambdas"])
    print("wrote", out_json)
    if doc["coarse_missing_lambdas"] or doc["dense_missing"]:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
