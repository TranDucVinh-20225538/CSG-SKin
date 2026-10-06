#!/usr/bin/env python3
"""Mean±std for B1 dense seeds at λ ∈ {0, 0.25, 1} — leakage, kNN-50, cosine (no Maha)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase16/b1")
LAMBDAS = [0, 0.25, 1]
SEEDS = [42, 52, 62]


def tag(l):
    return "{:g}".format(l).replace(".", "p")


def main():
    missing = []
    rows = []
    for lam in LAMBDAS:
        leak, knn, cos = [], [], []
        for s in SEEDS:
            p = ROOT / f"bcn_ham_ladv{tag(lam)}_s{s}" / "summary.json"
            if not p.exists():
                missing.append(str(p))
                continue
            o = json.loads(p.read_text())
            leak.append(o["leakage_ham_heldout"]["bal_acc_mean"])
            d = o["ood_detectors_ham_heldout"]
            knn.append(d["knn_k50"])
            cos.append(d["cosine_max"])
        if len(leak) < len(SEEDS):
            continue
        rows.append(
            {
                "lambda": lam,
                "leakage_mean": float(np.mean(leak)),
                "leakage_std": float(np.std(leak, ddof=1)) if len(leak) > 1 else 0.0,
                "knn50_mean": float(np.mean(knn)),
                "knn50_std": float(np.std(knn, ddof=1)) if len(knn) > 1 else 0.0,
                "cosine_mean": float(np.mean(cos)),
                "cosine_std": float(np.std(cos, ddof=1)) if len(cos) > 1 else 0.0,
                "seeds": SEEDS,
                "per_seed_leakage": leak,
                "per_seed_knn": knn,
                "per_seed_cosine": cos,
            }
        )
    out = ROOT / "dense3seed_aggregate.json"
    doc = {"rows": rows, "missing": missing, "note": "Mahalanobis omitted (BCN→HAM baseline below chance)"}
    out.write_text(json.dumps(doc, indent=2) + "\n")
    if missing:
        print("MISSING", len(missing), "files; partial rows", len(rows))
        for m in missing:
            print(" ", m)
        sys.exit(1 if not rows else 0)
    print("λ   leakage          knn50            cosine")
    for r in rows:
        print(
            f"{r['lambda']:g}  "
            f"{r['leakage_mean']:.3f}±{r['leakage_std']:.3f}  "
            f"{r['knn50_mean']:.3f}±{r['knn50_std']:.3f}  "
            f"{r['cosine_mean']:.3f}±{r['cosine_std']:.3f}"
        )
    print("wrote", out)


if __name__ == "__main__":
    main()
