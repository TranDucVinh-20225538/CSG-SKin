#!/usr/bin/env python3
"""Paper B R5 P3: frozen ImageNet ResNet-50 monitor vs the lesion-level model (decisions/paperB_r5_p3.md). CPU.

Frozen side, two sources:
  mapped  (primary now, no extraction): the R2 item-1 features (results/paperB/r2/features/imagenet_resnet50, image-level
          frames) re-indexed by image path onto the lesion-level frames. Lesion-level images that sat in the image-level
          validation split have no cached feature; they are dropped from fit and ID test and the coverage is reported.
          The model side is scored on the same covered ID subset (matched) and on the full test set.
  full    (after r5_frozen_extract.py --encoder imagenet_r50): complete lesion-level frames.
Model side: z_lesion_norm (Table 1 space) of the 35 lesion-level runs; λ = 0 and 2 are the reported cells, all λ listed.
Detectors: Mahalanobis class-conditional (pre-registered), Ledoit-Wolf Mahalanobis (sensitivity), kNN k = 50.
CIs: ID by lesion group, pad_heldout by patient, Fitzpatrick17k by image and (if P13 has run) by near-duplicate cluster.
Usage: r5_p3_frozen_monitor.py [--source mapped|full|both]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402

OUT = R.R5 / "p3"
R50_OLD = R.C.R2 / "features" / "imagenet_resnet50"
FROZEN = R.R5 / "frozen" / "features"
OODS = ("pad_heldout", "fitzpatrick17k")
REPORT_LAMS = (0.0, 2.0)
TARGET = 0.90


class MahaLW:
    def __init__(self, ztr, ytr, n_classes=8):
        z, y = np.asarray(ztr, np.float64), np.asarray(ytr, int)
        self.means = np.stack([z[y == c].mean(0) for c in range(n_classes)])
        lw = LedoitWolf().fit(z - self.means[y])
        self.prec, self.shrinkage = lw.precision_, float(lw.shrinkage_)

    def score(self, z):
        z = np.asarray(z, np.float64)
        zp, mp = z @ self.prec, self.means @ self.prec
        d = np.einsum("nd,nd->n", zp, z)[:, None] + np.einsum("cd,cd->c", mp, self.means)[None] - 2.0 * zp @ self.means.T
        return np.maximum(d, 0.0).min(1)


def frozen_scores(ztr, ytr, zid, zoods):
    m, lw, k = R.C.Maha(ztr, ytr), MahaLW(ztr, ytr), R.C.KNN(ztr)
    return {"maha": (m.score(zid), {o: m.score(z) for o, z in zoods.items()}),
            "maha_lw": (lw.score(zid), {o: lw.score(z) for o, z in zoods.items()}),
            "knn": (k.score(zid), {o: k.score(z) for o, z in zoods.items()})}


def mapped_r50():
    """Old R50 features re-indexed onto lesion-level frames. Returns (ztr, ytr, keep_tr, zte, yte, keep_te, oods)."""
    img = pd.read_csv(R.C.ISIC_SPLIT_CSV)
    old = {s: dict(np.load(R50_OLD / "{}.npz".format(s))) for s in ("isic_train", "isic_test", "pad_heldout", "fitzpatrick17k")}
    for s, key in (("train", "isic_train"), ("test", "isic_test")):
        assert (img.split == s).sum() == len(old[key]["z"]), (s, key)
    where = {r.path: (r.split, int(r.row)) for r in img.itertuples()}
    train, _val, test = R.lesion_frames()

    def take(frame):
        keep, zs = [], []
        for p in frame.path:
            sp, row = where.get(p, ("missing", -1))
            if sp in ("train", "test"):
                keep.append(True)
                zs.append(old["isic_" + sp]["z"][row])
            else:
                keep.append(False)
        return np.stack(zs), np.asarray(keep)

    ztr, ktr = take(train)
    zte, kte = take(test)
    ytr, yte = train.label_idx.to_numpy()[ktr], test.label_idx.to_numpy()[kte]
    return {"ztr": ztr, "ytr": ytr, "keep_tr": ktr, "zte": zte, "yte": yte, "keep_te": kte,
            "oods": {o: old[o]["z"] for o in OODS}}


def full_encoder(enc):
    d = FROZEN / enc
    f = {s: dict(np.load(d / "{}.npz".format(s))) for s in ("isic_train", "isic_test") + OODS}
    n = len(f["isic_test"]["z"])
    return {"ztr": f["isic_train"]["z"], "ytr": f["isic_train"]["labels"], "keep_tr": np.ones(len(f["isic_train"]["z"]), bool),
            "zte": f["isic_test"]["z"], "yte": f["isic_test"]["labels"], "keep_te": np.ones(n, bool),
            "oods": {o: f[o]["z"] for o in OODS}}


def fitz_kinds():
    return ["image"] + (["cluster_phash", "cluster_union"] if R.P13_CLUSTERS.exists() else [])


def score_frozen(pack, label):
    gid = R.id_test_groups()[pack["keep_te"]]
    sc = frozen_scores(pack["ztr"], pack["ytr"], pack["zte"], pack["oods"])
    res = {"encoder": label, "n_fit": int(len(pack["ztr"])), "n_id": int(len(pack["zte"])),
           "coverage_train": float(pack["keep_tr"].mean()), "coverage_test": float(pack["keep_te"].mean()), "cells": {}}
    for det, (sid, so) in sc.items():
        for o in OODS:
            for kind in (fitz_kinds() if o == "fitzpatrick17k" else ["patient"]):
                good = R.ood_groups(o, kind if o == "fitzpatrick17k" else "image")
                lo, hi = R.cluster_ci(sid, gid, so[o], good)
                res["cells"]["{}|{}|{}".format(det, o, kind)] = {"auroc": R.auroc(sid, so[o]), "ci_lo": lo, "ci_hi": hi}
    return res


def score_model(keep_te=None):
    """z_lesion_norm Mahalanobis and kNN of every run, on the full ID test and on the covered subset."""
    rows = []
    for lam, seed, tag in R.runs():
        tr, te = R.load(tag, "isic_train"), R.load(tag, "isic_test")
        oo = {o: R.load(tag, o) for o in OODS}
        m, k = R.C.Maha(tr["z_lesion_norm"], tr["labels"].astype(int)), R.C.KNN(tr["z_lesion_norm"])
        rec = {"lambda": lam, "seed": seed, "run": tag}
        for det, f in (("maha", m), ("knn", k)):
            sid = f.score(te["z_lesion_norm"])
            for o in OODS:
                so = f.score(oo[o]["z_lesion_norm"])
                rec["{}|{}|full".format(det, o)] = R.auroc(sid, so)
                if keep_te is not None:
                    rec["{}|{}|covered".format(det, o)] = R.auroc(sid[keep_te], so)
        rows.append(rec)
    return rows


def model_ci(lam, det, o):
    """Pooled bootstrap (PREREGISTER draws) of the model's mean AUROC at one λ, full ID test."""
    pre = R.prereg()
    rng = np.random.default_rng(int(pre["rng_seed"]))
    per = {}
    for l, s, tag in R.runs():
        if l != lam:
            continue
        tr, te, oo = R.load(tag, "isic_train"), R.load(tag, "isic_test"), R.load(tag, o)
        f = R.C.Maha(tr["z_lesion_norm"], tr["labels"].astype(int)) if det == "maha" else R.C.KNN(tr["z_lesion_norm"])
        per[s] = (f.score(te["z_lesion_norm"]), f.score(oo["z_lesion_norm"]))
    n = int(pre["n_resamples"])
    return R.pooled_bootstrap(per, R.draws(R.id_test_groups(), n, rng), R.draws(R.ood_groups(o), n, rng))


def report(frozen, model_rows, out=OUT, title="Paper B R5 P3 — frozen ImageNet monitor vs the lesion-level model",
           decisions="decisions/paperB_r5_p3.md"):
    md = pd.DataFrame(model_rows)
    flat = []
    for fr in frozen:
        for key, c in fr["cells"].items():
            det, o, kind = key.split("|")
            flat.append({"encoder": fr["encoder"], "detector": det, "ood": o, "resample": kind, "auroc": c["auroc"],
                         "ci_lo": c["ci_lo"], "ci_hi": c["ci_hi"], "n_fit": fr["n_fit"], "n_id": fr["n_id"],
                         "coverage_test": fr["coverage_test"], "meets_0.90": c["auroc"] >= TARGET})
    for r in model_rows:
        for k, v in r.items():
            if "|" in k:
                det, o, idv = k.split("|")
                flat.append({"encoder": "model_z_lesion_norm_lambda{:g}".format(r["lambda"]), "seed": r["seed"], "detector": det,
                             "ood": o, "resample": idv, "auroc": v})
    pd.DataFrame(flat).to_csv(out / "summary.csv", index=False)
    L = ["# " + title, "", "Design: `{}`. Commit `{}`. ID = lesion-level ISIC test; fit = lesion-level ISIC train; target "
         "(R2 pre-registration carried over): Mahalanobis ≥ {:.2f} on pad_heldout and on Fitzpatrick17k.".format(
             decisions, R.git_head(), TARGET), ""]
    L += ["## Frozen encoders", "", "| Encoder | n fit / n ID (coverage) | Detector | pad_heldout [95% CI] | Fitzpatrick17k, image [95% CI] |"
          " Fitzpatrick17k, cluster [95% CI] |", "|---|---|---|---|---|---|"]
    tex = []
    for fr in frozen:
        for det in ("maha", "maha_lw", "knn"):
            c = fr["cells"]
            pad, fi = c["{}|pad_heldout|patient".format(det)], c["{}|fitzpatrick17k|image".format(det)]
            cl = c.get("{}|fitzpatrick17k|cluster_union".format(det))
            L.append("| {} | {} / {} ({:.3f}) | {} | {:.3f} [{:.3f}, {:.3f}] | {:.3f} [{:.3f}, {:.3f}] | {} |".format(
                fr["encoder"], fr["n_fit"], fr["n_id"], fr["coverage_test"], det, pad["auroc"], pad["ci_lo"], pad["ci_hi"],
                fi["auroc"], fi["ci_lo"], fi["ci_hi"], "{:.3f} [{:.3f}, {:.3f}]".format(cl["auroc"], cl["ci_lo"], cl["ci_hi"]) if cl else "—"))
            if det == "maha":
                tex.append("{} & {:.3f} [{:.3f}, {:.3f}] & {:.3f} [{:.3f}, {:.3f}] \\\\".format(
                    fr["encoder"].replace("_", "\\_"), pad["auroc"], pad["ci_lo"], pad["ci_hi"], fi["auroc"], fi["ci_lo"], fi["ci_hi"]))
    L += ["", "## Model (z_lesion^norm), mean ± s.d. over seeds", "", "| λ | n | Maha pad (full ID) | Maha pad (covered ID) | "
          "Maha Fitz (full) | kNN pad (full) | kNN Fitz (full) |", "|---:|---:|---|---|---|---|---|"]
    for lam in sorted(md["lambda"].unique()):
        d = md[md["lambda"] == lam]
        g = lambda k: R.fmt(d[k].mean(), d[k].std(ddof=1)) if k in d else "—"  # noqa: E731
        L.append("| {:g} | {} | {} | {} | {} | {} | {} |".format(lam, len(d), g("maha|pad_heldout|full"), g("maha|pad_heldout|covered"),
                                                             g("maha|fitzpatrick17k|full"), g("knn|pad_heldout|full"), g("knn|fitzpatrick17k|full")))
        if lam in REPORT_LAMS:
            tex.append("model $\\lambda={:g}$ & {} & {} \\\\".format(lam, R.tex_pm(d["maha|pad_heldout|full"].mean(), d["maha|pad_heldout|full"].std(ddof=1)),
                                                                 R.tex_pm(d["maha|fitzpatrick17k|full"].mean(), d["maha|fitzpatrick17k|full"].std(ddof=1))))
    ci = {(lam, o): model_ci(lam, "maha", o) for lam in REPORT_LAMS for o in OODS}
    L += ["", "Model pooled bootstrap (PREREGISTER draws), Mahalanobis: " + "; ".join(
        "λ = {:g} {}: {:.3f} [{:.3f}, {:.3f}]".format(l, o, v["mean"], v["pooled_ci_lo"], v["pooled_ci_hi"]) for (l, o), v in ci.items()), ""]
    for fr in frozen:
        c = fr["cells"]
        p, f = c["maha|pad_heldout|patient"]["auroc"], c["maha|fitzpatrick17k|image"]["auroc"]
        d0, d2 = md[md["lambda"] == 0.0], md[md["lambda"] == 2.0]
        L.append("{}: Mahalanobis {:.3f} on pad_heldout and {:.3f} on Fitzpatrick17k (target {:.2f}); the model gives {:.3f} / {:.3f} "
                 "at λ = 0 and {:.3f} / {:.3f} at λ = 2 on the same ID test.".format(
                     fr["encoder"], p, f, TARGET, d0["maha|pad_heldout|full"].mean(), d0["maha|fitzpatrick17k|full"].mean(),
                     d2["maha|pad_heldout|full"].mean(), d2["maha|fitzpatrick17k|full"].mean()))
        L.append("")
    L += ["## LaTeX rows (encoder & Maha pad_heldout [CI] & Maha Fitzpatrick [CI])", "", "```latex"] + tex + ["```", ""]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    R.write_json(out / "results.json", {"commit": R.git_head(), "frozen": frozen, "model_ci": {"{:g}|{}".format(*k): v for k, v in ci.items()}})
    print("wrote", out / "REPORT.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("mapped", "full", "both"), default="both")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen, keep = [], None
    if a.source in ("mapped", "both"):
        mp = mapped_r50()
        keep = mp["keep_te"]
        frozen.append(score_frozen(mp, "imagenet_r50_mapped_R2_features"))
    if a.source in ("full", "both") and (FROZEN / "imagenet_r50" / "fitzpatrick17k.npz").exists():
        frozen.append(score_frozen(full_encoder("imagenet_r50"), "imagenet_r50_full_frames"))
    report(frozen, score_model(keep))


if __name__ == "__main__":
    main()
