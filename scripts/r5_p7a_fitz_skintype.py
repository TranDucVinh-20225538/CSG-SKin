#!/usr/bin/env python3
"""Paper B R5 P7(a) addition: Fitzpatrick17k OOD stratified by Fitzpatrick skin type (decisions/paperB_r5_p7a.md).

Strata from the dataset's `fitzpatrick_scale` label: I–II, III–IV, V–VI (−1 = unlabelled, reported as a count only).
Mahalanobis (class-conditional, z_lesion_norm, fit on ISIC train) on the 3,887 OOD Fitzpatrick images of the 35 lesion-level
Table 1 runs; per stratum AUROC vs the lesion-level ISIC test at λ = 0 and λ = 2. CI: PREREGISTER draws (2,000, seed
20261005), ID side by lesion group, Fitzpatrick side by the P13 near-duplicate clusters (cluster_union, restricted to
the stratum); image-level CI alongside. CPU only, cached features.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402

OUT = R.R5 / "p7a" / "fitz_skintype"
FITZ_CSV = R.RES / "phase11_third_domain" / "fitzpatrick17k.csv"
STRATA = {"I–II": (1, 2), "III–IV": (3, 4), "V–VI": (5, 6)}
LAMS = (0.0, 2.0)
DET = "mahalanobis_classcond"
MAP = {"melanoma": "MEL", "basal cell carcinoma": "BCC", "squamous cell carcinoma": "SCC", "actinic keratosis": "AK",
       "dermatofibroma": "DF", "congenital nevus": "NV"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pre = R.prereg()
    n = int(pre["n_resamples"])
    stems = R.fitz_stems()
    meta = pd.read_csv(FITZ_CSV).drop_duplicates("md5hash").set_index("md5hash").loc[stems]
    ftype = meta.fitzpatrick_scale.to_numpy()
    clus = R.fitz_groups("cluster_union")
    idg = R.id_test_groups()
    scores = {}
    for lam, s, tag in R.runs():
        if lam not in LAMS:
            continue
        sc, _ = R.detector_scores(R.load(tag, "isic_train"), R.load(tag, "isic_test"),
                                  {"fitzpatrick17k": R.load(tag, "fitzpatrick17k")})
        scores[(lam, s)] = (sc[DET][0], sc[DET][1]["fitzpatrick17k"])
    rows, cells = [], {}
    strata = dict(STRATA, **{"all labelled": (1, 2, 3, 4, 5, 6)})
    for name, types in strata.items():
        m = np.isin(ftype, types)
        idx = np.flatnonzero(m)
        mapped = meta.label.to_numpy()[idx]
        info = {"n_images": int(m.sum()), "n_clusters": int(len(set(clus[idx]))),
                "share_mapped_to_isic_class": float(pd.Series(mapped).map(MAP).notna().mean())}
        for unit, g in (("cluster_union", clus[idx]), ("image", np.asarray(stems)[idx])):
            rng = np.random.default_rng(int(pre["rng_seed"]))
            idd, odd = R.draws(idg, n, rng), R.draws(g, n, rng)
            for lam in LAMS:
                per = {s: (sid, so[idx]) for (l, s), (sid, so) in scores.items() if l == lam}
                b = R.pooled_bootstrap(per, idd, odd)
                cells["{}|{}|{:g}".format(name, unit, lam)] = b
                rows.append({"stratum": name, "resample": unit, "lambda": lam, **info, "n_seeds": b["n_seeds"],
                             "mean": b["mean"], "sd": b["sd"], "ci_lo": b["pooled_ci_lo"], "ci_hi": b["pooled_ci_hi"],
                             "k_ci_below_0.5": b["k_seeds_ci_below_0.5"], "rule_A": b["rule_A"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    R.write_json(OUT / "results.json", {"commit": R.git_head(), "n_unlabelled_type": int((ftype == -1).sum()),
                                        "n_total": int(len(ftype)), "cells": cells})
    c = df[df["resample"] == "cluster_union"].set_index(["stratum", "lambda"])
    im = df[df["resample"] == "image"].set_index(["stratum", "lambda"])
    L = ["# Paper B R5 P7(a): Fitzpatrick17k by skin type", "",
         "Design: `decisions/paperB_r5_p7a.md` (addition of 2026-10-15). Commit `{}`. Mahalanobis (class-conditional) on "
         "z_lesion_norm, fit on ISIC train; ID = lesion-level ISIC test. Mean ± s.d. over 5 seeds; 95% CI of the seed mean, "
         "PREREGISTER draws, Fitzpatrick side resampled by P13 `cluster_union` within the stratum (image-level CI in the "
         "last column). {} of {} images have no skin-type label (−1) and are excluded.".format(
             R.git_head(), int((ftype == -1).sum()), len(ftype)), "",
         "| stratum | n images | n clusters | mapped to an ISIC class | λ | Mahalanobis | CI (clusters) | k seeds CI<0.5 | rule A | CI (images) |",
         "|---|---:|---:|---:|---:|---|---|---:|---|---|"]
    tex = []
    for name in strata:
        for lam in LAMS:
            r, q = c.loc[(name, lam)], im.loc[(name, lam)]
            L.append("| {} | {} | {} | {:.1%} | {:g} | {:.3f} ± {:.3f} | [{:.3f}, {:.3f}] | {}/5 | {} | [{:.3f}, {:.3f}] |".format(
                name, r.n_images, r.n_clusters, r.share_mapped_to_isic_class, lam, r["mean"], r.sd, r.ci_lo, r.ci_hi,
                r["k_ci_below_0.5"], "yes" if r.rule_A else "no", q.ci_lo, q.ci_hi))
            tex.append("{} & {} & {:g} & {:.3f}\\,$\\pm$\\,{:.3f} & [{:.3f}, {:.3f}] \\\\".format(
                name.replace("–", "--"), r.n_images, lam, r["mean"], r.sd, r.ci_lo, r.ci_hi))
    L += ["", "## LaTeX rows (stratum & n & λ & Mahalanobis & cluster CI)", "", "```latex"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
