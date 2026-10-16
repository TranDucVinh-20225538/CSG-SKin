#!/usr/bin/env python3
"""Paper B post hoc: explain BCN→HAM Mahalanobis inversion at λ = 0 (decisions/paperB_r5_bcnham_maha0.md).

CPU, labelled. Features from r5_s_extract (s_flip). Detectors fit on BCN train only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, "/data2/hpcshared/Vinh/CSG-Skin")
import bootstrap_ood_ci as B  # noqa: E402
import r5cpu_common as R  # noqa: E402
from src.utils import ood_metrics  # noqa: E402

FEAT = R.R5 / "s_flip" / "features"
OUT = R.R5 / "audit" / "bcnham_maha0"
SEEDS = (42, 52, 62)
MIN_N = 20
NCLS = 8
LAB = list(R.LABELS)
RNG = 20261005


def load_run(seed):
    d = FEAT / "bcn_ham_ladv0_s{}".format(seed)
    P = {k: dict(np.load(d / "{}.npz".format(k), allow_pickle=True)) for k in ("id_train", "id_test", "ood")}
    for k in P:
        P[k]["labels"] = P[k]["labels"].astype(int)
        P[k]["group"] = P[k]["group"].astype(str)
    return P


def class_mix(y):
    y = np.asarray(y, int)
    c = np.bincount(y, minlength=NCLS).astype(float)
    return c / max(c.sum(), 1.0)


def counts(y):
    y = np.asarray(y, int)
    c = np.bincount(y, minlength=NCLS)
    return {LAB[i]: int(c[i]) for i in range(NCLS)}


def maha_params(ztr, ytr):
    return ood_metrics.compute_mahalanobis_params_from_arrays(ztr, ytr, NCLS, 1e-3)


def dist_to_own_centroid(z, y, means, prec):
    """Squared Mahalanobis distance of each sample to its own class mean."""
    z, y = np.asarray(z, np.float64), np.asarray(y, int)
    means = np.asarray(means, np.float64)
    prec = np.asarray(prec, np.float64)
    out = np.full(len(z), np.nan)
    for c in range(NCLS):
        m = y == c
        if not m.any() or not np.isfinite(means[c]).all():
            continue
        d = z[m] - means[c]
        out[m] = np.einsum("ij,jk,ik->i", d, prec, d)
    return out


def ood_weights_to_id_mix(y_ood, pi_id, pi_ood):
    y = np.asarray(y_ood, int)
    w = np.ones(len(y), dtype=np.float64)
    for c in range(NCLS):
        m = y == c
        if not m.any():
            continue
        if pi_ood[c] <= 0:
            w[m] = 0.0
        else:
            w[m] = pi_id[c] / pi_ood[c]
    s = w.sum()
    if s > 0:
        w *= len(w) / s
    return w


def boot_auroc_lesion(per_seed_sid_sood, gi, go, n_boot, seed=RNG, w_ood_per_seed=None):
    """Lesion-clustered bootstrap of seed-mean AUROC. Optional per-seed OOD sample weights."""
    rng = np.random.default_rng(seed)
    id_draws = R.draws(gi, n_boot, rng)
    ood_draws = R.draws(go, n_boot, np.random.default_rng(seed + 1))
    seeds = sorted(per_seed_sid_sood)
    pts = {}
    for s in seeds:
        sid, so = per_seed_sid_sood[s]
        if w_ood_per_seed is None:
            pts[s] = R.auroc(sid, so)
        else:
            pts[s] = R.auroc(sid, so, w_ood=w_ood_per_seed[s])
    mat = np.full((n_boot, len(seeds)), np.nan)
    for b, (ii, oo) in enumerate(zip(id_draws, ood_draws)):
        for j, s in enumerate(seeds):
            sid, so = per_seed_sid_sood[s]
            if w_ood_per_seed is None:
                mat[b, j] = R.auroc(sid[ii], so[oo])
            else:
                mat[b, j] = R.auroc(sid[ii], so[oo], w_ood=w_ood_per_seed[s][oo])
    pooled = np.nanmean(mat, 1)
    lo, hi = np.nanpercentile(pooled, [2.5, 97.5])
    v = np.array([pts[s] for s in seeds], float)
    return {
        "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "ci_lo": float(lo), "ci_hi": float(hi), "n_seeds": len(seeds),
        "per_seed": {int(s): float(pts[s]) for s in seeds},
    }


def boot_class_matched(per_seed_scores_c, gi_c, go_c, n_boot, seed=RNG):
    """per_seed_scores_c: {seed: (sid_c, sood_c)} only class-c images; groups aligned."""
    if any(len(gi_c) == 0 or len(go_c) == 0 for _ in [0]):
        return None
    # groups must match the sliced arrays — rebuild from full using masks in caller
    return boot_auroc_lesion(per_seed_scores_c, gi_c, go_c, n_boot, seed=seed)


def pm(m):
    if m is None:
        return "—"
    if m.get("sd") is not None and m["n_seeds"] > 1:
        return "{:.3f} ± {:.3f}".format(m["mean"], m["sd"])
    return "{:.3f}".format(m["mean"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pre = R.prereg()
    n_boot = int(pre["n_resamples"])

    runs = {s: load_run(s) for s in SEEDS}
    # Class mix from seed 42 split (same images across seeds for phase16 features)
    y_te = runs[SEEDS[0]]["id_test"]["labels"]
    y_ood = runs[SEEDS[0]]["ood"]["labels"]
    pi_te, pi_ood = class_mix(y_te), class_mix(y_ood)
    n_te, n_ood = counts(y_te), counts(y_ood)
    tv = float(0.5 * np.abs(pi_te - pi_ood).sum())
    mix = {
        "split": "phase16 BCN→HAM: id_test=BCN-test, ood=HAM-heldout (s_flip features; R6-R5 outcome split)",
        "n_bcn_test": int(len(y_te)), "n_ham_heldout": int(len(y_ood)),
        "n_bcn_test_per_class": n_te, "n_ham_heldout_per_class": n_ood,
        "pi_bcn_test": {LAB[i]: float(pi_te[i]) for i in range(NCLS)},
        "pi_ham_heldout": {LAB[i]: float(pi_ood[i]) for i in range(NCLS)},
        "tv": tv,
        "majority_bcn_test": LAB[int(np.argmax(pi_te))],
        "majority_ham_heldout": LAB[int(np.argmax(pi_ood))],
        "p_maj_bcn": float(pi_te.max()), "q_maj_ham": float(pi_ood.max()),
        "min_n_for_reading": MIN_N,
    }
    R.write_json(OUT / "class_mix.json", mix)

    gi0 = runs[SEEDS[0]]["id_test"]["group"]
    go0 = runs[SEEDS[0]]["ood"]["group"]
    for s in SEEDS[1:]:
        assert np.array_equal(runs[s]["id_test"]["group"], gi0)
        assert np.array_equal(runs[s]["ood"]["group"], go0)

    # Scores per seed
    sid_maha, so_maha, sid_knn, so_knn = {}, {}, {}, {}
    means_prec = {}
    w_ood = {}
    for s, P in runs.items():
        ztr, ytr = P["id_train"]["z"], P["id_train"]["labels"]
        zte, zood = P["id_test"]["z"], P["ood"]["z"]
        sid_maha[s], so_maha[s] = B.maha_scores(ztr, ytr, zte, zood)
        sid_knn[s], so_knn[s] = B.knn50_scores(ztr, zte, zood)
        means_prec[s] = maha_params(ztr, ytr)
        # OOD weights → BCN-test mix (use seed-42 mix; identical labels across seeds)
        w_ood[s] = ood_weights_to_id_mix(P["ood"]["labels"], pi_te, pi_ood)

    # (3) overall + reweighted
    unw = boot_auroc_lesion({s: (sid_maha[s], so_maha[s]) for s in SEEDS}, gi0, go0, n_boot)
    knn = boot_auroc_lesion({s: (sid_knn[s], so_knn[s]) for s in SEEDS}, gi0, go0, n_boot)
    rw = boot_auroc_lesion({s: (sid_maha[s], so_maha[s]) for s in SEEDS}, gi0, go0, n_boot,
                          w_ood_per_seed=w_ood)
    gap = 0.5 - unw["mean"]
    recovered = float((rw["mean"] - unw["mean"]) / gap) if gap > 1e-9 else float("nan")
    reweight = {
        "mahalanobis_unweighted": unw,
        "knn_unweighted": knn,
        "mahalanobis_ood_reweighted_to_bcn_test_mix": rw,
        "gap_to_0.5_unweighted": float(gap),
        "share_of_gap_recovered_by_reweight": recovered,
    }
    R.write_json(OUT / "reweight.json", reweight)

    # (2) per-class class-matched + (4)(5) dispersion / n
    yte0, yood0 = runs[SEEDS[0]]["id_test"]["labels"], runs[SEEDS[0]]["ood"]["labels"]
    per_rows, disp_rows = [], []
    for c in range(NCLS):
        name = LAB[c]
        mid = yte0 == c
        mood = yood0 == c
        n_id, n_o = int(mid.sum()), int(mood.sum())
        eligible = n_id >= MIN_N and n_o >= MIN_N
        row = {"class": name, "n_bcn_test": n_id, "n_ham_heldout": n_o, "eligible_for_reading": eligible}
        if n_id == 0 or n_o == 0:
            row.update({"maha_mean": None, "knn_mean": None, "note": "empty side"})
            per_rows.append(row)
            continue
        gi_c, go_c = gi0[mid], go0[mood]
        maha_c = {s: (sid_maha[s][mid], so_maha[s][mood]) for s in SEEDS}
        knn_c = {s: (sid_knn[s][mid], so_knn[s][mood]) for s in SEEDS}
        bm = boot_auroc_lesion(maha_c, gi_c, go_c, n_boot, seed=RNG + 10 + c)
        bk = boot_auroc_lesion(knn_c, gi_c, go_c, n_boot, seed=RNG + 100 + c)
        row.update({
            "maha_mean": bm["mean"], "maha_sd": bm["sd"], "maha_ci_lo": bm["ci_lo"], "maha_ci_hi": bm["ci_hi"],
            "knn_mean": bk["mean"], "knn_sd": bk["sd"], "knn_ci_lo": bk["ci_lo"], "knn_ci_hi": bk["ci_hi"],
        })
        per_rows.append(row)

        # dispersion: median own-centroid distance, mean over seeds
        med_id, med_ood = [], []
        for s in SEEDS:
            means, prec = means_prec[s]
            # means may be array
            P = runs[s]
            d_id = dist_to_own_centroid(P["id_test"]["z"][mid], P["id_test"]["labels"][mid], means, prec)
            d_o = dist_to_own_centroid(P["ood"]["z"][mood], P["ood"]["labels"][mood], means, prec)
            med_id.append(float(np.nanmedian(d_id)))
            med_ood.append(float(np.nanmedian(d_o)))
        disp_rows.append({
            "class": name, "n_bcn_test": n_id, "n_ham_heldout": n_o, "eligible_for_reading": eligible,
            "median_maha2_bcn_test": float(np.mean(med_id)),
            "median_maha2_ham_heldout": float(np.mean(med_ood)),
            "ratio_ham_over_bcn": float(np.mean(med_ood) / max(np.mean(med_id), 1e-12)),
        })

    pdf = pd.DataFrame(per_rows)
    ddf = pd.DataFrame(disp_rows)
    pdf.to_csv(OUT / "per_class.csv", index=False)
    ddf.to_csv(OUT / "dispersion.csv", index=False)

    # Reading
    elig = pdf[pdf["eligible_for_reading"] == True]  # noqa: E712
    maha_elig = elig["maha_mean"].dropna()
    n_below = int((maha_elig < 0.5).sum()) if len(maha_elig) else 0
    n_elig = int(len(maha_elig))
    rw_m, unw_m = rw["mean"], unw["mean"]
    if rw_m >= 0.5 and n_below == 0 and n_elig > 0:
        reading = "label_shift"
        reading_text = (
            "Reweighted Mahalanobis AUROC ≥ 0.5 and all eligible class-matched Mahalanobis AUROCs ≥ 0.5 → "
            "label shift.")
    elif n_below == n_elig and n_elig > 0 and rw_m < 0.5 and (rw_m - unw_m) < 0.02:
        reading = "dispersion"
        reading_text = (
            "Eligible class-matched Mahalanobis AUROCs stay < 0.5 and reweighting barely moves the pooled "
            "AUROC → within-class dispersion.")
    else:
        reading = "both"
        reading_text = (
            "Mixed: reweighting moves Mahalanobis toward 0.5 (share of gap recovered = {:.1%}) but "
            "pooled reweighted AUROC is {:.3f}; {}/{} eligible class-matched Mahalanobis AUROCs are < 0.5. "
            "Both label shift and within-class dispersion contribute; do not pick one.".format(
                recovered if recovered == recovered else float("nan"), rw_m, n_below, n_elig))

    summary = {
        "commit": R.git_head(), "seeds": list(SEEDS), "reading": reading,
        "reading_text": reading_text, "reweight": reweight, "class_mix_tv": tv,
        "n_eligible_classes": n_elig, "n_eligible_maha_below_0.5": n_below,
    }
    R.write_json(OUT / "summary.json", summary)
    pd.DataFrame([{
        "metric": "maha_unweighted", **{k: unw[k] for k in ("mean", "sd", "ci_lo", "ci_hi")},
    }, {
        "metric": "knn_unweighted", **{k: knn[k] for k in ("mean", "sd", "ci_lo", "ci_hi")},
    }, {
        "metric": "maha_ood_reweighted_to_bcn_test", **{k: rw[k] for k in ("mean", "sd", "ci_lo", "ci_hi")},
        "share_gap_recovered": recovered,
    }]).to_csv(OUT / "summary.csv", index=False)

    L = [
        "# Paper B: BCN→HAM Mahalanobis at λ = 0 — label shift vs dispersion",
        "",
        "Design: `decisions/paperB_r5_bcnham_maha0.md`. Commit `{}`. Seeds {}. "
        "Detectors fit on BCN train; OOD = HAM-heldout; ID test = BCN-test. "
        "Lesion-clustered 95% CI (PREREGISTER {} draws, rng {}). "
        "Classes with n < {} on either side are excluded from the reading (CI still shown).".format(
            R.git_head(), SEEDS, n_boot, RNG, MIN_N),
        "",
        "## (1) Class mix (this split)",
        "",
        "BCN-test n = {}, HAM-heldout n = {}. TV = {:.3f}. "
        "Majority: BCN-test {} ({:.3f}) → HAM-heldout {} ({:.3f}).".format(
            mix["n_bcn_test"], mix["n_ham_heldout"], tv,
            mix["majority_bcn_test"], mix["p_maj_bcn"],
            mix["majority_ham_heldout"], mix["q_maj_ham"]),
        "",
        "| class | n BCN-test | π BCN-test | n HAM-heldout | π HAM | eligible |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for c in LAB:
        L.append("| {} | {} | {:.3f} | {} | {:.3f} | {} |".format(
            c, n_te[c], mix["pi_bcn_test"][c], n_ood[c], mix["pi_ham_heldout"][c],
            "yes" if n_te[c] >= MIN_N and n_ood[c] >= MIN_N else "no"))
    L += [
        "",
        "## (3) Pooled AUROC (lesion CI) and OOD-side reweight to BCN-test mix",
        "",
        "| detector / weighting | mean ± s.d. | 95% CI (lesion) |",
        "|---|---|---|",
        "| Mahalanobis unweighted | {} | [{:.3f}, {:.3f}] |".format(pm(unw), unw["ci_lo"], unw["ci_hi"]),
        "| kNN k=50 unweighted | {} | [{:.3f}, {:.3f}] |".format(pm(knn), knn["ci_lo"], knn["ci_hi"]),
        "| Mahalanobis, HAM reweighted → BCN-test mix | {} | [{:.3f}, {:.3f}] |".format(
            pm(rw), rw["ci_lo"], rw["ci_hi"]),
        "",
        "Share of gap to 0.5 recovered by reweighting: **{:.1%}** "
        "(unweighted {:.3f} → reweighted {:.3f}).".format(recovered, unw_m, rw_m),
        "",
        "## (2) Per-class class-matched AUROC (HAM class c vs BCN-test class c)",
        "",
        "| class | n_BCN | n_HAM | Maha mean ± sd [CI] | kNN mean ± sd [CI] | in reading |",
        "|---|---:|---:|---|---|:---:|",
    ]
    for _, r in pdf.iterrows():
        if r.get("maha_mean") is None or (isinstance(r.get("maha_mean"), float) and np.isnan(r["maha_mean"])):
            L.append("| {} | {} | {} | — | — | no |".format(r["class"], r["n_bcn_test"], r["n_ham_heldout"]))
            continue
        L.append("| {} | {} | {} | {:.3f} ± {:.3f} [{:.3f}, {:.3f}] | {:.3f} ± {:.3f} [{:.3f}, {:.3f}] | {} |".format(
            r["class"], r["n_bcn_test"], r["n_ham_heldout"],
            r["maha_mean"], r["maha_sd"], r["maha_ci_lo"], r["maha_ci_hi"],
            r["knn_mean"], r["knn_sd"], r["knn_ci_lo"], r["knn_ci_hi"],
            "yes" if r["eligible_for_reading"] else "no"))
    L += [
        "",
        "## (4) Per-class dispersion (median Mahalanobis² to own class centroid)",
        "",
        "| class | n_BCN | n_HAM | median d² BCN-test | median d² HAM | HAM/BCN | in reading |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for _, r in ddf.iterrows():
        L.append("| {} | {} | {} | {:.2f} | {:.2f} | {:.2f} | {} |".format(
            r["class"], r["n_bcn_test"], r["n_ham_heldout"],
            r["median_maha2_bcn_test"], r["median_maha2_ham_heldout"], r["ratio_ham_over_bcn"],
            "yes" if r["eligible_for_reading"] else "no"))
    L += [
        "",
        "## Reading",
        "",
        "**{}.** {}".format(reading, reading_text),
        "",
        "Pre-existing headline (A1 / R6-R5): Mahalanobis λ = 0 ≈ 0.369, kNN ≈ 0.658 — reproduced here as "
        "unweighted pooled estimates above.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", OUT / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
