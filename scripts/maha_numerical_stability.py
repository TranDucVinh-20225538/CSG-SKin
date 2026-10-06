#!/usr/bin/env python3
"""Item 2: Mahalanobis AUROC stability vs precision and BLAS threads (pad_heldout, λ=2, seed 42)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
FEAT = ROOT / "results/paperB/phase13/features/runB_orth1_ladv2_s42"
OUT = ROOT / "results/paperB/reviewer_r1"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/data2/hpcshared/Vinh/CSG-Skin")
from src.utils import ood_metrics


def pooled_centered(z, y, n_classes=8):
    means = np.zeros((n_classes, z.shape[1]), dtype=z.dtype)
    centered = []
    for c in range(n_classes):
        m = y == c
        if not m.any():
            continue
        means[c] = z[m].mean(0)
        centered.append(z[m] - means[c])
    return means, np.concatenate(centered, 0)


def maha_auroc(ztr, ytr, zid, zood, dtype, reg_eps=1e-3, ledoit=False):
    ztr = np.asarray(ztr, dtype=dtype)
    zid = np.asarray(zid, dtype=dtype)
    zood = np.asarray(zood, dtype=dtype)
    if ledoit:
        means, centered = pooled_centered(ztr, ytr)
        n = max(len(centered) - 1, 1)
        lw = LedoitWolf().fit(centered)
        prec = np.linalg.inv(lw.covariance_ + reg_eps * np.eye(ztr.shape[1]))
        sid = ood_metrics.mahalanobis_min_squared_distances(zid, means, prec)
        sood = ood_metrics.mahalanobis_min_squared_distances(zood, means, prec)
    else:
        means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(ztr, ytr, 8, reg_eps)
        sid = ood_metrics.mahalanobis_min_squared_distances(zid, means, prec)
        sood = ood_metrics.mahalanobis_min_squared_distances(zood, means, prec)
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    return float(roc_auc_score(y, np.concatenate([sid, sood])))


def main():
    tr = dict(np.load(FEAT / "isic_train.npz"))
    te = dict(np.load(FEAT / "isic_test.npz"))
    hold = dict(np.load(FEAT / "pad_heldout.npz"))
    ztr, ytr = tr["z_lesion_norm"], tr["labels"]
    zid, zood = te["z_lesion_norm"], hold["z_lesion_norm"]
    rows = []
    baseline = maha_auroc(ztr, ytr, zid, zood, np.float64, ledoit=False)
    rows.append({"config": "float64_ridge_eps1e-3", "threads": "default", "auroc": baseline, "ledoit": False})
    rows.append({"config": "float32_ridge_eps1e-3", "threads": "default", "auroc": maha_auroc(ztr, ytr, zid, zood, np.float32, ledoit=False), "ledoit": False})
    rows.append({"config": "float64_ledoit_wolf", "threads": "default", "auroc": maha_auroc(ztr, ytr, zid, zood, np.float64, ledoit=True), "ledoit": True})
    for threads in (1, 4, 8):
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        rows.append(
            {
                "config": "float64_ridge_eps1e-3",
                "threads": threads,
                "auroc": maha_auroc(ztr, ytr, zid, zood, np.float64, ledoit=False),
                "ledoit": False,
            }
        )
    vals = [r["auroc"] for r in rows]
    rep = {
        "checkpoint": "runB_orth1_ladv2_s42",
        "partition": "pad_heldout",
        "baseline_auroc": baseline,
        "max_abs_deviation": float(max(abs(v - baseline) for v in vals)),
        "rows": rows,
        "threshold_note": ">|1e-3| investigate; >|1e-2| fix precision policy in Methods",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "maha_stability.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
