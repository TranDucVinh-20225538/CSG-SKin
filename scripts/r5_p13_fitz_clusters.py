#!/usr/bin/env python3
"""Paper B R5 P13: Fitzpatrick17k bootstrap over near-duplicate clusters instead of images (decisions/paperB_r5_p13.md).

The 3,887 Fitzpatrick17k images used as OOD all come from atlasdermatologico.com.br (url ...img?imageId=N) and carry no
patient or case identifier. Cluster definitions (all written to results/paperB/r5/p13/fitz_clusters.csv, keyed by md5hash):
  cluster_phash       connected components of 64-bit DCT perceptual hash, Hamming distance ≤ 8 (primary)
  cluster_embed       connected components of cosine ≥ 0.95 on frozen ImageNet ResNet-50 features (R2 item 1 cache)
  cluster_union       components of the union of both edge sets (primary for the bootstrap)
  cluster_imageid_run same diagnosis label and consecutive imageId (gap ≤ 1) — atlas upload runs (sensitivity, coarse)
  cluster_diagnosis   diagnosis label (90 groups; most conservative bound)
Stage --hash (CPU, reads the 3,887 JPEGs once): writes the cluster CSV and cluster statistics.
Stage --bootstrap (CPU, cached features): Mahalanobis / kNN / cosine on z_lesion_norm of the 35 lesion-level runs,
PREREGISTER draws per λ, Fitzpatrick side resampled by each cluster definition; CI widths and seeds with CI < 0.5.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402

OUT = R.R5 / "p13"
FITZ_CSV = R.RES / "phase11_third_domain" / "fitzpatrick17k.csv"
R50_FITZ = R.C.R2 / "features" / "imagenet_resnet50" / "fitzpatrick17k.npz"
HAMMING = 8
COS = 0.95
KINDS = ("image", "cluster_phash", "cluster_embed", "cluster_union", "cluster_imageid_run", "cluster_diagnosis")
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def phash(path):
    im = Image.open(path).convert("L").resize((32, 32), Image.LANCZOS)
    d = dctn(np.asarray(im, np.float64), norm="ortho")[:8, :8].ravel()
    bits = d > np.median(d[1:])
    return np.packbits(bits)


def edges_hamming(H, thr):
    rows, cols = [], []
    for i in range(len(H)):
        dist = POP[np.bitwise_xor(H[i + 1:], H[i])].sum(1)
        j = np.flatnonzero(dist <= thr) + i + 1
        rows += [i] * len(j)
        cols += j.tolist()
    return np.asarray(rows, int), np.asarray(cols, int)


def edges_cosine(Z, thr, chunk=512):
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    rows, cols = [], []
    for a in range(0, len(Z), chunk):
        S = Z[a:a + chunk] @ Z.T
        i, j = np.nonzero(S >= thr)
        i = i + a
        k = j > i
        rows.append(i[k])
        cols.append(j[k])
    return np.concatenate(rows), np.concatenate(cols)


def components(n, r, c):
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    return connected_components(A, directed=False)[1]


def imageid_runs(meta):
    iid = meta.url.str.extract(r"imageId=(\d+)")[0].astype(float)
    out = np.full(len(meta), -1, int)
    k = 0
    for _lab, d in meta.assign(iid=iid).groupby("label"):
        d = d.sort_values("iid")
        prev = None
        for idx, v in zip(d.index, d.iid):
            if prev is None or not np.isfinite(v) or not np.isfinite(prev) or v - prev > 1:
                k += 1
            out[idx] = k
            prev = v
    return out


def stage_hash():
    OUT.mkdir(parents=True, exist_ok=True)
    stems = R.fitz_stems()
    paths = sorted(R.T.p3.FITZ_DIR.glob("*.jpg"))
    assert len(stems) == 3887
    meta = pd.read_csv(FITZ_CSV).drop_duplicates("md5hash").set_index("md5hash").loc[stems].reset_index()
    H = np.stack([phash(p) for p in paths])
    rp, cp = edges_hamming(H, HAMMING)
    z = np.load(R50_FITZ)["z"].astype(np.float64)
    assert len(z) == len(stems), "R50 Fitzpatrick feature order"
    re_, ce = edges_cosine(z, COS)
    n = len(stems)
    df = pd.DataFrame({"md5hash": stems, "label": meta.label, "image_id": meta.url.str.extract(r"imageId=(\d+)")[0],
                       "phash": ["".join("{:02x}".format(b) for b in h) for h in H]})
    df["cluster_phash"] = components(n, rp, cp)
    df["cluster_embed"] = components(n, re_, ce)
    df["cluster_union"] = components(n, np.r_[rp, re_], np.r_[cp, ce])
    df["cluster_imageid_run"] = imageid_runs(meta)
    df["cluster_diagnosis"] = pd.factorize(meta.label)[0]
    df.to_csv(R.P13_CLUSTERS, index=False)
    stats = {"n_images": n, "hamming_threshold": HAMMING, "cosine_threshold": COS, "n_edges_phash": int(len(rp)),
             "n_edges_embed": int(len(re_)), "hosts": meta.url.str.extract(r"https?://([^/]+)/")[0].value_counts().to_dict()}
    for k in KINDS[1:]:
        sz = df[k].value_counts()
        stats[k] = {"n_clusters": int(len(sz)), "n_singletons": int((sz == 1).sum()), "largest": int(sz.max()),
                    "images_in_multi": int(sz[sz > 1].sum()), "effective_n": float(n ** 2 / (sz ** 2).sum())}
    R.write_json(OUT / "cluster_stats.json", stats)
    print(stats)


def stage_bootstrap():
    pre = R.prereg()
    n = int(pre["n_resamples"])
    cl = pd.read_csv(R.P13_CLUSTERS, dtype=str)
    rr = R.runs()
    idg = R.id_test_groups()
    scores = {}
    for lam, s, tag in rr:
        tr, te, fz = R.load(tag, "isic_train"), R.load(tag, "isic_test"), R.load(tag, "fitzpatrick17k")
        sc, _ = R.detector_scores(tr, te, {"fitzpatrick17k": fz})
        for det in R.BOOT_DETS:
            scores[(det, lam, s)] = (sc[det][0], sc[det][1]["fitzpatrick17k"])
    rows, out = [], {}
    for kind in KINDS:
        g = R.fitz_groups(kind)
        rng = np.random.default_rng(int(pre["rng_seed"]))
        idd, odd = R.draws(idg, n, rng), R.draws(g, n, rng)
        for det in R.BOOT_DETS:
            for lam in sorted({l for _d, l, _s in scores}):
                per = {s: scores[(det, l, s)] for (d, l, s) in scores if d == det and l == lam}
                b = R.pooled_bootstrap(per, idd, odd)
                out["{}|{}|{:g}".format(kind, det, lam)] = b
                rows.append({"resample": kind, "n_groups": int(len(set(g))), "detector": det, "lambda": lam, "n_seeds": b["n_seeds"],
                             "mean": b["mean"], "ci_lo": b["pooled_ci_lo"], "ci_hi": b["pooled_ci_hi"],
                             "ci_width": b["pooled_ci_hi"] - b["pooled_ci_lo"], "k_ci_below_0.5": b["k_seeds_ci_below_0.5"],
                             "rule_A": b["rule_A"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    R.write_json(OUT / "bootstrap.json", {"commit": R.git_head(), "cells": out})
    report(df)


def report(df):
    st = json.loads((OUT / "cluster_stats.json").read_text())
    L = ["# Paper B R5 P13 — Fitzpatrick17k bootstrap over near-duplicate clusters", "",
         "Design: `decisions/paperB_r5_p13.md`. Commit `{}`. ID side: lesion-level ISIC test by lesion group; Fitzpatrick side "
         "resampled by the cluster column; PREREGISTER draws (2,000, seed 20261005), z_lesion_norm.".format(R.git_head()), "",
         "| Resample unit | clusters | singletons | largest | effective n |", "|---|---:|---:|---:|---:|", "| image | 3887 | 3887 | 1 | 3887 |"]
    for k in KINDS[1:]:
        s = st[k]
        L.append("| {} | {} | {} | {} | {:.0f} |".format(k, s["n_clusters"], s["n_singletons"], s["largest"], s["effective_n"]))
    L += ["", "## Mahalanobis, pooled mean and 95% CI by resample unit", "", "| λ | n | " + " | ".join(KINDS) + " |", "|---:|---:|" + "---|" * len(KINDS)]
    tex = []
    m = df[df.detector == "mahalanobis_classcond"]
    for lam in sorted(m["lambda"].unique()):
        d = m[m["lambda"] == lam].set_index("resample")
        cells = ["{:.3f} [{:.3f}, {:.3f}] k={}".format(d.loc[k, "mean"], d.loc[k, "ci_lo"], d.loc[k, "ci_hi"], d.loc[k, "k_ci_below_0.5"]) for k in KINDS]
        L.append("| {:g} | {} | {} |".format(lam, d["n_seeds"].iloc[0], " | ".join(cells)))
        tex.append("{:g} & ".format(lam) + " & ".join("[{:.3f}, {:.3f}]".format(d.loc[k, "ci_lo"], d.loc[k, "ci_hi"]) for k in ("image", "cluster_union", "cluster_diagnosis")) + " \\\\")
    L += ["", "## CI width ratio vs image resampling (all detectors, λ = 2)", "", "| Detector | " + " | ".join(KINDS[1:]) + " |", "|---|" + "---|" * (len(KINDS) - 1)]
    for det in R.BOOT_DETS:
        d = df[(df.detector == det) & (df["lambda"] == 2.0)].set_index("resample")
        if len(d):
            L.append("| {} | {} |".format(R.DET_LABEL[det], " | ".join("{:.2f}".format(d.loc[k, "ci_width"] / d.loc["image", "ci_width"]) for k in KINDS[1:])))
    d = m[m["lambda"] == 2.0].set_index("resample")
    if len(d):
        L += ["", "## Plain reading", "", "At λ = 2 the Fitzpatrick17k Mahalanobis CI is [{:.3f}, {:.3f}] resampling images, [{:.3f}, {:.3f}] resampling "
              "near-duplicate clusters (union) and [{:.3f}, {:.3f}] resampling whole diagnoses; seeds with CI < 0.5: {}, {}, {} of {}.".format(
                  d.loc["image", "ci_lo"], d.loc["image", "ci_hi"], d.loc["cluster_union", "ci_lo"], d.loc["cluster_union", "ci_hi"],
                  d.loc["cluster_diagnosis", "ci_lo"], d.loc["cluster_diagnosis", "ci_hi"], d.loc["image", "k_ci_below_0.5"],
                  d.loc["cluster_union", "k_ci_below_0.5"], d.loc["cluster_diagnosis", "k_ci_below_0.5"], d["n_seeds"].iloc[0])]
    L += ["", "## LaTeX rows (λ & image CI & cluster-union CI & diagnosis CI), Mahalanobis", "", "```latex"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--bootstrap", action="store_true")
    a = ap.parse_args()
    if a.hash:
        stage_hash()
    if a.bootstrap:
        stage_bootstrap()


if __name__ == "__main__":
    main()
