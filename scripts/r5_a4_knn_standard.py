#!/usr/bin/env python3
"""Paper B audit A4 (decisions/paperB_r5_audit2.md): standard kNN (Sun et al. 2022: distance to the k-th neighbour)
next to the mean-of-k variant used in Table 1, k ∈ {10, 50, 200}.

35 lesion-level Table 1 runs; OOD sets pad_heldout, Fitzpatrick17k, MED-NODE, SD-198, SCIN. Features z_lesion_norm,
L2-normalised again, Euclidean neighbours among L2-normalised ISIC-train features (`bootstrap_ood_ci.fit_knn`
convention). Score larger = more OOD. The mean-of-50 column is asserted equal to `bootstrap_ood_ci.knn50_scores`.
CI: PREREGISTER pooled bootstrap, units as in S1 (ID lesion; OOD patient / P13 cluster / image / image / case).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import bootstrap_ood_ci as B  # noqa: E402
import r5cpu_common as R  # noqa: E402
import r5_s_flip as S  # noqa: E402

OUT = R.R5 / "audit" / "a4_knn_standard"
KS = (10, 50, 200)
SETS = S.LES_SETS
VARIANTS = [("kth", k) for k in KS] + [("mean", k) for k in KS]


def l2n(z):
    z = np.asarray(z, np.float64)
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pre = R.prereg()
    n = int(pre["n_resamples"])
    cells, check = {}, []
    runs = R.runs(need=("isic_train", "isic_test") + tuple(SETS))
    assert len(runs) == 35, len(runs)
    for lam, seed, tag in runs:
        tr, te = R.load(tag, "isic_train"), R.load(tag, "isic_test")
        nn = NearestNeighbors(n_neighbors=max(KS), metric="euclidean").fit(l2n(tr[R.PRIMARY_REP]))
        did = nn.kneighbors(l2n(te[R.PRIMARY_REP]))[0]
        for name in SETS:
            oo = R.load(tag, name)
            dood = nn.kneighbors(l2n(oo[R.PRIMARY_REP]))[0]
            if name == "pad_heldout":
                ref_id, ref_ood = B.knn50_scores(tr[R.PRIMARY_REP], te[R.PRIMARY_REP], oo[R.PRIMARY_REP])
                check.append(max(np.abs(did[:, :50].mean(1) - ref_id).max(), np.abs(dood[:, :50].mean(1) - ref_ood).max()))
            for v, k in VARIANTS:
                f = (lambda D: D[:, k - 1]) if v == "kth" else (lambda D: D[:, :k].mean(1))
                cells.setdefault((name, lam, v, k), {})[seed] = (f(did), f(dood))
        print("scored", tag, "max|mean50 - knn50_scores| so far {:.2e}".format(max(check)), flush=True)
    assert max(check) < 1e-5, max(check)
    rows, draws = [], {}
    for (name, lam, v, k), per in sorted(cells.items(), key=lambda kv: (list(SETS).index(kv[0][0]), kv[0][1], kv[0][2], kv[0][3])):
        if name not in draws:
            rng = np.random.default_rng(int(pre["rng_seed"]))
            draws[name] = (R.draws(R.id_test_groups(), n, rng), R.draws(S.ood_groups(name), n, rng))
        b = R.pooled_bootstrap(per, *draws[name])
        rows.append({"set": name, "lambda": lam, "variant": v, "k": k, "n_seeds": b["n_seeds"], "mean": b["mean"],
                     "sd": b["sd"], "ci_lo": b["pooled_ci_lo"], "ci_hi": b["pooled_ci_hi"],
                     "k_seeds_ci_below_0.5": b["k_seeds_ci_below_0.5"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    R.write_json(OUT / "results.json", {"commit": R.git_head(), "max_abs_dev_mean50_vs_knn50_scores": max(check),
                                        "rows": rows})
    L = ["# Paper B audit A4: standard kNN (k-th neighbour) vs mean-of-k", "",
         "Design: `decisions/paperB_r5_audit2.md`. Commit `{}`. 35 lesion-level runs, z_lesion_norm (L2-normalised), "
         "Euclidean, fit on ISIC train; ID = ISIC lesion test. Mean ± s.d. over 5 seeds [95% CI of the seed mean]. "
         "Check: mean-of-50 equals `bootstrap_ood_ci.knn50_scores` to {:.1e}.".format(R.git_head(), max(check)), ""]
    tex = []
    for name in SETS:
        L += ["## {}".format(S.SET_LABEL[name]), "",
              "| λ | " + " | ".join("{} k={}".format("k-th" if v == "kth" else "mean", k) for v, k in VARIANTS) + " |",
              "|---:|" + "---|" * len(VARIANTS)]
        for lam in sorted(df["lambda"].unique()):
            sub = df[(df.set == name) & (df["lambda"] == lam)].set_index(["variant", "k"])
            L.append("| {:g} | {} |".format(lam, " | ".join("{:.3f} ± {:.3f} [{:.3f}, {:.3f}]".format(
                sub.loc[(v, k), "mean"], sub.loc[(v, k), "sd"], sub.loc[(v, k), "ci_lo"], sub.loc[(v, k), "ci_hi"])
                for v, k in VARIANTS)))
            tex.append("{} & {:g} & {} \\\\".format(S.SET_LABEL[name], lam, " & ".join(
                R.tex_pm(sub.loc[(v, k), "mean"], sub.loc[(v, k), "sd"]) for v, k in VARIANTS)))
        L.append("")
    L += ["## LaTeX rows (set & λ & k-th 10/50/200 & mean 10/50/200)", "", "```latex"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
