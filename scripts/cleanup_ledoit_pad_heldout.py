#!/usr/bin/env python3
"""Ledoit-Wolf Mahalanobis on pad_heldout (Phase 13 cached z_lesion_norm)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
FEAT = ROOT / "results/paperB/phase13/features"
OUT = ROOT / "results/paperB/cleanup"
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
HEADLINE_EPS = 1e-3


def lam_tag(x):
    return "{:g}".format(float(x)).replace(".", "p")


def auroc_pair(sid, sood):
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    return float(roc_auc_score(y, np.concatenate([sid, sood])))


def pooled_centered(z, y, n_classes=8):
    means = np.zeros((n_classes, z.shape[1]), dtype=np.float64)
    present = []
    centered = []
    for c in range(n_classes):
        m = y == c
        if not m.any():
            continue
        present.append(c)
        means[c] = z[m].mean(0)
        centered.append(z[m] - means[c])
    return means, np.concatenate(centered, 0), present


def maha_min_sq(z, means, prec):
    mins = np.empty(len(z), dtype=np.float64)
    p = prec.astype(np.float64)
    for i, x in enumerate(z.astype(np.float64)):
        best = np.inf
        for c in range(means.shape[0]):
            d = x - means[c]
            dist = float(d @ p @ d)
            if dist < best:
                best = dist
        mins[i] = best
    return mins


def maha_ridge(ztr, ytr, zid, zood, eps):
    means, centered, _ = pooled_centered(ztr, ytr)
    n = max(len(centered) - 1, 1)
    cov = (centered.T @ centered) / n + eps * np.eye(ztr.shape[1])
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    kappa = float(np.linalg.cond(cov))
    sid = maha_min_sq(zid, means, prec)
    sood = maha_min_sq(zood, means, prec)
    return {"auroc": auroc_pair(sid, sood), "condition_number": kappa}


def maha_ledoit(ztr, ytr, zid, zood):
    means, centered, _ = pooled_centered(ztr, ytr)
    lw = LedoitWolf().fit(centered)
    cov = np.asarray(lw.covariance_, dtype=np.float64)
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    sid = maha_min_sq(zid, means, prec)
    sood = maha_min_sq(zood, means, prec)
    return {
        "auroc": auroc_pair(sid, sood),
        "condition_number": float(np.linalg.cond(cov)),
        "shrinkage": float(lw.shrinkage_),
    }


def has_features(tag):
    d = FEAT / tag
    return all((d / f"{s}.npz").exists() for s in ("isic_train", "isic_test", "pad_heldout"))


def load_split(tag, split):
    return dict(np.load(FEAT / tag / f"{split}.npz"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for lam in LAMS:
        seeds = (42, 52, 62, 72, 82) if lam in (0.0, 2.0, 8.0) else (42, 52, 62)
        for seed in seeds:
            tag = f"runB_orth1_ladv{lam_tag(lam)}_s{seed}"
            if not has_features(tag):
                continue
            tr = load_split(tag, "isic_train")
            idp = load_split(tag, "isic_test")
            hold = load_split(tag, "pad_heldout")
            ztr, ytr = tr["z_lesion_norm"], tr["labels"]
            zid, zood = idp["z_lesion_norm"], hold["z_lesion_norm"]
            ridge = maha_ridge(ztr, ytr, zid, zood, HEADLINE_EPS)
            lewo = maha_ledoit(ztr, ytr, zid, zood)
            rows.append(
                {
                    "lambda_adv": float(lam),
                    "seed": int(seed),
                    "headline_auroc": ridge["auroc"],
                    "headline_kappa": ridge["condition_number"],
                    "ledoit_auroc": lewo["auroc"],
                    "ledoit_kappa": lewo["condition_number"],
                    "ledoit_shrinkage": lewo["shrinkage"],
                }
            )
    by_lam = {}
    for lam in LAMS:
        sub = [r for r in rows if r["lambda_adv"] == lam]
        if not sub:
            continue
        hl = [r["headline_auroc"] for r in sub]
        lw = [r["ledoit_auroc"] for r in sub]
        by_lam[str(lam)] = {
            "headline_mean": float(np.mean(hl)),
            "headline_std": float(np.std(hl, ddof=1)) if len(hl) > 1 else 0.0,
            "ledoit_mean": float(np.mean(lw)),
            "ledoit_std": float(np.std(lw, ddof=1)) if len(lw) > 1 else 0.0,
            "kappa_ledoit_mean": float(np.mean([r["ledoit_kappa"] for r in sub])),
            "per_seed": sub,
        }
    doc = {"by_lambda": by_lam, "rows": rows}
    (OUT / "ledoit_pad_heldout.json").write_text(json.dumps(doc, indent=2) + "\n")
    b2 = by_lam.get("2.0", {})
    print("lambda=2 headline", b2.get("headline_mean"), "LW", b2.get("ledoit_mean"))


if __name__ == "__main__":
    main()
