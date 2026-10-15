#!/usr/bin/env python3
"""Paper B audit A1 (decisions/paperB_r5_audit2.md): BCN↔HAM bootstrap by lesion on both sides.

Replaces the image-level `bootstrap_phase16_reverse.py` interval ([0.666, 0.699], HAM→BCN λ = 1, 3 seeds). Features
from r5_s_extract.py (results/paperB/r5/s_flip/features/<run>/{id_train,id_test,ood}.npz, z + lesion_id). Detectors as
in phase16 (`bootstrap_ood_ci`: Mahalanobis class-conditional + 1e-3·I, kNN k=50, cosine max; fit on ID train).
PREREGISTER draws (2,000, seed 20261005), draws shared across seeds; lesion on both sides, image on both sides alongside.
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import bootstrap_ood_ci as B  # noqa: E402
import r5cpu_common as R  # noqa: E402

F16 = R.R5 / "s_flip" / "features"
OUT = R.R5 / "audit" / "a1_bcnham_lesion"
PAIR = {"bcn_ham": "BCN→HAM", "ham_bcn": "HAM→BCN"}
DETS = ("mahalanobis_classcond", "knn_k50", "cosine_max")
OLD = {"lo": 0.666, "hi": 0.699, "source": "reviewer_r1/bootstrap_phase16_ham_bcn_ladv1.json (image both sides)"}


def scores(d):
    P = {k: dict(np.load(d / "{}.npz".format(k), allow_pickle=True)) for k in ("id_train", "id_test", "ood")}
    ztr, ytr, zid, zood = P["id_train"]["z"], P["id_train"]["labels"].astype(int), P["id_test"]["z"], P["ood"]["z"]
    out = {"mahalanobis_classcond": B.maha_scores(ztr, ytr, zid, zood), "cosine_max": B.cosine_scores(ztr, ytr, zid, zood),
           "knn_k50": B.knn50_scores(ztr, zid, zood)}
    return out, P["id_test"]["group"].astype(str), P["ood"]["group"].astype(str)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pre = R.prereg()
    n = int(pre["n_resamples"])
    cells, groups = {}, {}
    for d in sorted(F16.glob("*_ladv*_s*")):
        m = re.fullmatch(r"(bcn_ham|ham_bcn)_ladv([0-9p]+)_s(\d+)", d.name)
        if not m:
            continue
        pair, lam, seed = m.group(1), float(m.group(2).replace("p", ".")), int(m.group(3))
        sc, gi, go = scores(d)
        if pair in groups:
            assert np.array_equal(groups[pair][0], gi) and np.array_equal(groups[pair][1], go), d.name
        groups[pair] = (gi, go)
        cells.setdefault((pair, lam), {})[seed] = sc
        print("scored", d.name, flush=True)
    assert sum(len(v) for v in cells.values()) == 28, "expected 28 phase16 runs"
    rows, res = [], {}
    for pair in PAIR:
        gi, go = groups[pair]
        draws = {}
        for unit, (a, b) in (("lesion", (gi, go)), ("image", (np.arange(len(gi)).astype(str), np.arange(len(go)).astype(str)))):
            rng = np.random.default_rng(int(pre["rng_seed"]))
            draws[unit] = (R.draws(a, n, rng), R.draws(b, n, rng))
        for lam in sorted(l for p, l in cells if p == pair):
            per = cells[(pair, lam)]
            for det in DETS:
                for unit, (idd, odd) in draws.items():
                    bb = R.pooled_bootstrap({s: per[s][det] for s in sorted(per)}, idd, odd)
                    res["{}|{:g}|{}|{}".format(pair, lam, det, unit)] = bb
                    rows.append({"pair": pair, "lambda": lam, "detector": det, "resample": unit, "n_seeds": bb["n_seeds"],
                                 "seeds": " ".join(map(str, sorted(per))), "n_id": len(gi), "n_ood": len(go),
                                 "n_id_lesions": len(set(gi)), "n_ood_lesions": len(set(go)), "mean": bb["mean"],
                                 "sd": bb["sd"], "ci_lo": bb["pooled_ci_lo"], "ci_hi": bb["pooled_ci_hi"],
                                 "k_seeds_ci_below_0.5": bb["k_seeds_ci_below_0.5"],
                                 "k_seeds_ci_above_0.5": sum(v["lo"] > 0.5 for v in bb["per_seed"].values())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    R.write_json(OUT / "results.json", {"commit": R.git_head(), "old_interval": OLD, "cells": res})
    key = df[(df.pair == "ham_bcn") & (df["lambda"] == 1.0) & (df.detector == "mahalanobis_classcond")].set_index("resample")
    L = ["# Paper B audit A1: BCN↔HAM bootstrap by lesion on both sides", "",
         "Design: `decisions/paperB_r5_audit2.md`. Commit `{}`. PREREGISTER draws (2,000, seed 20261005), shared across seeds. "
         "Features re-extracted from the phase16 checkpoints (`r5_s_extract.py`; recomputed vs stored Mahalanobis AUROC in "
         "`results/paperB/r5/s_flip/features/*/check.json`). Lesions: ID test {} / OOD {} (BCN→HAM), {} / {} (HAM→BCN).".format(
             R.git_head(), len(set(groups["bcn_ham"][0])), len(set(groups["bcn_ham"][1])), len(set(groups["ham_bcn"][0])),
             len(set(groups["ham_bcn"][1]))), "",
         "**Headline replacement (HAM→BCN, λ = 1, Mahalanobis, 3 seeds):** lesion-clustered 95% CI [{:.3f}, {:.3f}] "
         "(seeds with CI entirely above 0.5: {}/3); image-level on the same draws protocol [{:.3f}, {:.3f}]; the manuscript's "
         "[0.666, 0.699] came from `bootstrap_phase16_reverse.py` (image both sides).".format(
             key.loc["lesion", "ci_lo"], key.loc["lesion", "ci_hi"], key.loc["lesion", "k_seeds_ci_above_0.5"],
             key.loc["image", "ci_lo"], key.loc["image", "ci_hi"]), "",
         "| direction | λ | detector | seeds | mean ± s.d. | CI (lesion) | CI (image) | k CI>0.5 (lesion) | k CI<0.5 (lesion) |",
         "|---|---:|---|---:|---|---|---|---:|---:|"]
    tex = []
    piv = df.set_index(["pair", "lambda", "detector", "resample"])
    for pair in PAIR:
        for lam in sorted(df[df.pair == pair]["lambda"].unique()):
            for det in DETS:
                a, b = piv.loc[(pair, lam, det, "lesion")], piv.loc[(pair, lam, det, "image")]
                ms = "{:.3f} ± {:.3f}".format(a["mean"], a["sd"]) if a.n_seeds > 1 else "{:.3f}".format(a["mean"])
                L.append("| {} | {:g} | {} | {} | {} | [{:.3f}, {:.3f}] | [{:.3f}, {:.3f}] | {} | {} |".format(
                    PAIR[pair], lam, R.DET_LABEL[det], a.n_seeds, ms, a.ci_lo, a.ci_hi, b.ci_lo, b.ci_hi,
                    a["k_seeds_ci_above_0.5"], a["k_seeds_ci_below_0.5"]))
                if det == "mahalanobis_classcond":
                    tex.append("{} & {:g} & {} & {} & [{:.3f}, {:.3f}] \\\\".format(
                        PAIR[pair].replace("→", "$\\to$"), lam, a.n_seeds,
                        R.tex_pm(a["mean"], a["sd"] if a.n_seeds > 1 else None), a.ci_lo, a.ci_hi))
    L += ["", "## LaTeX rows (direction & λ & seeds & Mahalanobis & lesion CI)", "", "```latex"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
