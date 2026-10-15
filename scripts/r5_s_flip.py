#!/usr/bin/env python3
"""Paper B R5 S1/S2: sign-flipped Mahalanobis and the threshold view (decisions/paperB_r5_s.md).

Lesion-level block: 35 Table 1 runs × {pad_heldout, Fitzpatrick17k, MED-NODE, SD-198, SCIN}, ID test = ISIC lesion
test, threshold on ISIC lesion val. BCN↔HAM block: 28 phase16 runs (features from r5_s_extract.py). Mahalanobis as in
Table 1 (bootstrap_ood_ci.maha_scores: class means + shared covariance + 1e-3·I, fit on ID train). CPU only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402
import r5_p7a_fitz_skintype as FS  # noqa: E402
from src.utils import ood_metrics  # noqa: E402

OUT = R.R5 / "s_flip"
F16 = OUT / "features"
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LES_SETS = {"pad_heldout": "pad_heldout", "fitzpatrick17k": "Fitzpatrick17k", "mednode": "MED-NODE", "sd198": "SD-198",
            "scin": "SCIN"}
P16_SETS = {"bcn_ham": "BCN→HAM", "ham_bcn": "HAM→BCN"}
SET_LABEL = dict(LES_SETS, **P16_SETS)
LABELS = list(R.LABELS)
NV = "NV"
MIN_N = 20


def maha_fit(ztr, ytr):
    means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(ztr, np.asarray(ytr).astype(int), 8, 1e-3)
    return lambda z: ood_metrics.mahalanobis_min_squared_distances(z, means, prec).astype(np.float64)


def ood_groups(name):
    if name == "pad_heldout":
        return R.pad_heldout_patients()
    if name == "fitzpatrick17k":
        return R.fitz_groups("cluster_union")
    return pd.read_csv(R.R5 / "p7a" / "sets" / "{}.csv".format(name), keep_default_na=False).group.astype(str).to_numpy()


def ood_classes(name, feats):
    """Mapped ISIC class name per OOD image ('' = unmapped)."""
    if name == "pad_heldout":
        return np.array([LABELS[int(i)] for i in feats["labels"]])
    if name == "fitzpatrick17k":
        meta = pd.read_csv(FS.FITZ_CSV).drop_duplicates("md5hash").set_index("md5hash").loc[R.fitz_stems()]
        return meta.label.map(FS.MAP).fillna("").to_numpy()
    return pd.read_csv(R.R5 / "p7a" / "sets" / "{}.csv".format(name), keep_default_na=False).mapped.to_numpy()


def thresholds(s_val):
    return float(np.quantile(s_val, 0.95)), float(np.quantile(-s_val, 0.95))


def run_record(sid, sood, s_val, cls_id, cls_ood):
    a = R.auroc(sid, sood)
    af = R.auroc(-sid, -sood)
    t, tf = thresholds(s_val)
    pc = {}
    idn, oon = cls_id == NV, cls_ood == NV
    oom = (cls_ood != NV) & (cls_ood != "")
    for key, mi, mo in (("matched_NV", idn, oon), ("matched_nonNV", ~idn, oom), ("oodonly_NV", None, oon),
                        ("oodonly_nonNV", None, oom)):
        si = sid if mi is None else sid[mi]
        so = sood[mo]
        pc[key] = {"n_id": int(len(si)), "n_ood": int(len(so)), "auroc": R.auroc(si, so) if len(so) else float("nan")}
    return {"auroc": a, "auroc_flip": af, "flip_identity_dev": abs(af - (1.0 - a)), "t_val": t, "t_val_flip": tf,
            "id_acc": float(np.mean(sid <= t)), "ood_acc": float(np.mean(sood <= t)),
            "id_acc_flip": float(np.mean(-sid <= tf)), "ood_acc_flip": float(np.mean(-sood <= tf)),
            "fpr95_testcal": float(np.mean(sood <= np.quantile(sid, 0.95))),
            "fpr95_testcal_flip": float(np.mean(-sood <= np.quantile(-sid, 0.95))),
            "n_unmapped_ood": int(np.sum(cls_ood == "")), "per_class": pc}


def lesion_block():
    """{(set, lam): {seed: dict(sid, sood, sval, rec)}}"""
    cells, cls_cache = {}, {}
    for lam, seed, tag in R.runs(need=("isic_train", "isic_val", "isic_test") + tuple(LES_SETS)):
        tr, va, te = (R.load(tag, k) for k in ("isic_train", "isic_val", "isic_test"))
        f = maha_fit(tr[R.PRIMARY_REP], tr["labels"])
        sid, sval = f(te[R.PRIMARY_REP]), f(va[R.PRIMARY_REP])
        cls_id = np.array([LABELS[int(i)] for i in te["labels"]])
        for name in LES_SETS:
            oo = R.load(tag, name)
            if name not in cls_cache:
                cls_cache[name] = ood_classes(name, oo)
            sood = f(oo[R.PRIMARY_REP])
            assert len(sood) == len(cls_cache[name]), (tag, name)
            rec = run_record(sid, sood, sval, cls_id, cls_cache[name])
            cells.setdefault((name, lam), {})[seed] = {"sid": sid, "sood": sood, "rec": rec}
        print("lesion", tag, flush=True)
    assert sum(len(v) for v in cells.values()) == 35 * len(LES_SETS), "expected 35 runs with isic_val"
    return cells


def p16_block():
    cells, checks, groups = {}, {}, {}
    for d in sorted(F16.glob("*_ladv*_s*")):
        m = re.fullmatch(r"(bcn_ham|ham_bcn)_ladv([0-9p]+)_s(\d+)", d.name)
        if not m or not (d / "check.json").exists():
            continue
        pair, lam, seed = m.group(1), float(m.group(2).replace("p", ".")), int(m.group(3))
        P = {k: dict(np.load(d / "{}.npz".format(k), allow_pickle=True)) for k in ("id_train", "id_val", "id_test", "ood")}
        f = maha_fit(P["id_train"]["z"], P["id_train"]["labels"])
        sid, sood, sval = f(P["id_test"]["z"]), f(P["ood"]["z"]), f(P["id_val"]["z"])
        g = (P["id_test"]["group"].astype(str), P["ood"]["group"].astype(str))
        if pair in groups:
            assert all(np.array_equal(a, b) for a, b in zip(groups[pair], g)), d.name
        groups[pair] = g
        lab = lambda y: np.array([LABELS[int(i)] for i in y])  # noqa: E731
        rec = run_record(sid, sood, sval, lab(P["id_test"]["labels"]), lab(P["ood"]["labels"]))
        rec["stored_check"] = json.loads((d / "check.json").read_text())
        checks[d.name] = rec["stored_check"]["abs_diff"]
        cells.setdefault((pair, lam), {})[seed] = {"sid": sid, "sood": sood, "rec": rec}
        print("phase16", d.name, flush=True)
    assert sum(len(v) for v in cells.values()) == 28, "expected 28 phase16 runs"
    return cells, checks, groups


def aggregate(cells, id_groups_of, ood_groups_of, transfer_from):
    """transfer_from {pair_or_block: {seed: (t, t_flip)}} at λ = 0."""
    pre = R.prereg()
    n = int(pre["n_resamples"])
    rows, draws = [], {}
    for (name, lam), per in sorted(cells.items(), key=lambda kv: (list(SET_LABEL).index(kv[0][0]), kv[0][1])):
        if name not in draws:
            rng = np.random.default_rng(int(pre["rng_seed"]))
            draws[name] = (R.draws(id_groups_of(name), n, rng), R.draws(ood_groups_of(name), n, rng))
        idd, odd = draws[name]
        seeds = sorted(per)
        b = R.pooled_bootstrap({s: (per[s]["sid"], per[s]["sood"]) for s in seeds}, idd, odd)
        acc = {}
        for key, sgn, tk in (("ood_acc", 1.0, "t_val"), ("ood_acc_flip", -1.0, "t_val_flip")):
            ind = {s: (sgn * per[s]["sood"] <= per[s]["rec"][tk]) for s in seeds}
            v = np.array([np.mean([ind[s][oo].mean() for s in seeds]) for oo in odd])
            acc[key] = tuple(float(x) for x in np.percentile(v, [2.5, 97.5]))
        t0 = transfer_from[name]
        t0m, t0fm = float(np.mean([v[0] for v in t0.values()])), float(np.mean([v[1] for v in t0.values()]))
        recs = [per[s]["rec"] for s in seeds]
        ms = lambda xs: R.mean_sd([float(x) for x in xs])[:2]  # noqa: E731
        row = {"set": name, "lambda": lam, "n_seeds": len(seeds), "seeds": " ".join(map(str, seeds)),
               "n_id": len(per[seeds[0]]["sid"]), "n_ood": len(per[seeds[0]]["sood"]),
               "auroc_mean": b["mean"], "auroc_sd": b["sd"], "auroc_ci_lo": b["pooled_ci_lo"],
               "auroc_ci_hi": b["pooled_ci_hi"], "k_seeds_ci_below_0.5": b["k_seeds_ci_below_0.5"],
               "flip_mean": 1.0 - b["mean"], "flip_sd": b["sd"], "flip_ci_lo": 1.0 - b["pooled_ci_hi"],
               "flip_ci_hi": 1.0 - b["pooled_ci_lo"], "oracle_dir": "flip" if b["mean"] < 0.5 else "keep",
               "oracle_auroc": max(b["mean"], 1.0 - b["mean"]),
               "max_flip_identity_dev": max(r["flip_identity_dev"] for r in recs)}
        for k in ("id_acc", "ood_acc", "id_acc_flip", "ood_acc_flip", "fpr95_testcal", "fpr95_testcal_flip", "t_val",
                  "t_val_flip"):
            row[k + "_mean"], row[k + "_sd"] = ms(r[k] for r in recs)
        row["ood_acc_ci_lo"], row["ood_acc_ci_hi"] = acc["ood_acc"]
        row["ood_acc_flip_ci_lo"], row["ood_acc_flip_ci_hi"] = acc["ood_acc_flip"]
        row["t0"], row["t0_flip"] = t0m, t0fm
        row["t0_per_seed"] = json.dumps({str(k): v[0] for k, v in t0.items()})
        for k, sgn, t in (("x_id_acc", 1.0, t0m), ("x_id_acc_flip", -1.0, t0fm)):
            row[k + "_mean"], row[k + "_sd"] = ms(np.mean(sgn * per[s]["sid"] <= t) for s in seeds)
        for k, sgn, t in (("x_ood_acc", 1.0, t0m), ("x_ood_acc_flip", -1.0, t0fm)):
            row[k + "_mean"], row[k + "_sd"] = ms(np.mean(sgn * per[s]["sood"] <= t) for s in seeds)
        for key in ("matched_NV", "matched_nonNV", "oodonly_NV", "oodonly_nonNV"):
            pcs = [r["per_class"][key] for r in recs]
            row["pc_{}_n_id".format(key)], row["pc_{}_n_ood".format(key)] = pcs[0]["n_id"], pcs[0]["n_ood"]
            row["pc_{}_mean".format(key)], row["pc_{}_sd".format(key)] = ms(p["auroc"] for p in pcs)
        row["n_unmapped_ood"] = recs[0]["n_unmapped_ood"]
        rows.append(row)
        print("agg", name, lam, "{:.3f} flip {:.3f} fpr95 {:.3f}/{:.3f}".format(
            row["auroc_mean"], row["flip_mean"], row["ood_acc_mean"], row["ood_acc_flip_mean"]), flush=True)
    return rows


def f3(m, s=None):
    if m is None or not np.isfinite(m):
        return "—"
    return "{:.3f}".format(m) if s is None or not np.isfinite(s) else "{:.3f} ± {:.3f}".format(m, s)


def write_report(df, d_dir, d_src, checks):
    L = ["# Paper B R5 S1/S2: sign-flipped Mahalanobis and the threshold view", "",
         "Design: `decisions/paperB_r5_s.md`. Commit `{}`. Mahalanobis (class-conditional, shared covariance, fit on ID "
         "train); AUROC = `sklearn.metrics.roc_auc_score`, ID = 0, OOD = 1, larger = more OOD; flipped score = −score. "
         "Mean ± s.d. over seeds; 95% CI of the seed mean with PREREGISTER draws (2,000, seed 20261005), ID by lesion, "
         "OOD by patient (pad_heldout), P13 cluster (Fitzpatrick17k), image (MED-NODE, SD-198), case (SCIN), lesion "
         "(BCN↔HAM). Lesion block: 35 Table 1 runs. BCN↔HAM: phase16 EfficientNet-B3 runs, 3 seeds at λ ∈ {{0, 0.25, 1}}, "
         "1 seed elsewhere (single-seed cells: no s.d.).".format(R.git_head()), "",
         "Pipeline checks: max |AUROC(−s) − (1 − AUROC(s))| over all runs = {:.1e}; phase16 recomputed vs stored "
         "Mahalanobis AUROC, max |diff| over 28 runs = {:.1e}.".format(df.max_flip_identity_dev.max(),
                                                                    max(checks.values()) if checks else float("nan")), "",
         "## S1. Original vs flipped AUROC", "",
         "Transferred direction d, chosen on pad_heldout at λ = 2 (mean AUROC {:.3f}): **{}**, applied unchanged to every "
         "other cell. Oracle = max(A, 1 − A), the direction chosen on the test data itself (upper bound).".format(d_src, d_dir),
         "", "| set | λ | seeds | original | CI | flipped | CI | under d | oracle |", "|---|---:|---:|---|---|---|---|---|---|"]
    tex1, tex2 = [], []
    under = lambda r: r.flip_mean if d_dir == "flip" else r.auroc_mean  # noqa: E731
    for r in df.itertuples():
        L.append("| {} | {:g} | {} | {} | [{:.3f}, {:.3f}] | {} | [{:.3f}, {:.3f}] | {:.3f} | {:.3f} |".format(
            SET_LABEL[r.set], r.lambda_, r.n_seeds, f3(r.auroc_mean, r.auroc_sd if r.n_seeds > 1 else None),
            r.auroc_ci_lo, r.auroc_ci_hi, f3(r.flip_mean, r.flip_sd if r.n_seeds > 1 else None), r.flip_ci_lo,
            r.flip_ci_hi, under(r), r.oracle_auroc))
        tex1.append("{} & {:g} & {} & {} & {:.3f} \\\\".format(
            SET_LABEL[r.set].replace("→", "$\\to$"), r.lambda_, R.tex_pm(r.auroc_mean, r.auroc_sd if r.n_seeds > 1 else None),
            R.tex_pm(r.flip_mean, r.flip_sd if r.n_seeds > 1 else None), under(r)))
    L += ["", "Cells with AUROC < 0.5 under d, per λ (lesion sets / BCN↔HAM):", "",
          "| λ | lesion sets below 0.5 under d | BCN↔HAM below 0.5 under d |", "|---:|---|---|"]
    for lam in sorted(df["lambda"].unique()):
        sub = df[df["lambda"] == lam]
        les = [SET_LABEL[r.set] for r in sub.itertuples() if r.set in LES_SETS and under(r) < 0.5]
        p16 = [SET_LABEL[r.set] for r in sub.itertuples() if r.set in P16_SETS and under(r) < 0.5]
        nl, np16 = int(sub.set.isin(list(LES_SETS)).sum()), int(sub.set.isin(list(P16_SETS)).sum())
        L.append("| {:g} | {}/{} {} | {}/{} {} |".format(lam, len(les), nl, ", ".join(les), len(p16), np16, ", ".join(p16)))
    L += ["", "## S1. Per class (nevus vs others), original AUROC (flipped = 1 − value)", "",
          "Class-matched: ID NV vs OOD NV, ID non-NV vs OOD mapped non-NV. OOD-only: full ID test vs OOD NV / OOD mapped "
          "non-NV. n = OOD images in the cell; † = n_OOD < {}. Unmapped OOD images are excluded (count in the last "
          "column).".format(MIN_N), "",
          "| set | λ | matched NV | n | matched non-NV | n | OOD-only NV | OOD-only non-NV | n unmapped |",
          "|---|---:|---|---:|---|---:|---|---|---:|"]
    for r in df.itertuples():
        def pc(k):
            m, s, nn = getattr(r, "pc_{}_mean".format(k)), getattr(r, "pc_{}_sd".format(k)), getattr(r, "pc_{}_n_ood".format(k))
            return f3(m, s if r.n_seeds > 1 else None) + ("†" if nn < MIN_N else "")
        L.append("| {} | {:g} | {} | {} | {} | {} | {} | {} | {} |".format(
            SET_LABEL[r.set], r.lambda_, pc("matched_NV"), r.pc_matched_NV_n_ood, pc("matched_nonNV"),
            r.pc_matched_nonNV_n_ood, pc("oodonly_NV"), pc("oodonly_nonNV"), r.n_unmapped_ood))
    L += ["", "## S2. Threshold at 95% ID acceptance on ID validation", "",
          "Accept if score ≤ t, t = 95th percentile of the run's ID validation scores (ISIC lesion val; BCN or HAM val). "
          "OOD acceptance = FPR@95TPR. CI over OOD groups with t fixed.", "",
          "| set | λ | ID test acc (orig) | OOD acc (orig) | CI | ID test acc (flip) | OOD acc (flip) | CI |",
          "|---|---:|---|---|---|---|---|---|"]
    for r in df.itertuples():
        sd = r.n_seeds > 1
        L.append("| {} | {:g} | {} | {} | [{:.3f}, {:.3f}] | {} | {} | [{:.3f}, {:.3f}] |".format(
            SET_LABEL[r.set], r.lambda_, f3(r.id_acc_mean, r.id_acc_sd if sd else None),
            f3(r.ood_acc_mean, r.ood_acc_sd if sd else None), r.ood_acc_ci_lo, r.ood_acc_ci_hi,
            f3(r.id_acc_flip_mean, r.id_acc_flip_sd if sd else None),
            f3(r.ood_acc_flip_mean, r.ood_acc_flip_sd if sd else None), r.ood_acc_flip_ci_lo, r.ood_acc_flip_ci_hi))
    L += ["", "## S2. Threshold transferred from λ = 0 (monitor calibrated before DA)", "",
          "t0 = mean over the λ = 0 seeds of the validation thresholds (lesion: 5 seeds; BCN↔HAM: 3 seeds), applied "
          "unchanged to each λ > 0 run. Distances are not on a common scale across runs.", "",
          "| set | λ | t0 | ID test acc (orig) | OOD acc (orig) | ID test acc (flip) | OOD acc (flip) |",
          "|---|---:|---:|---|---|---|---|"]
    for r in df[df["lambda"] > 0].itertuples():
        sd = r.n_seeds > 1
        L.append("| {} | {:g} | {:.2f} | {} | {} | {} | {} |".format(
            SET_LABEL[r.set], r.lambda_, r.t0, f3(r.x_id_acc_mean, r.x_id_acc_sd if sd else None),
            f3(r.x_ood_acc_mean, r.x_ood_acc_sd if sd else None), f3(r.x_id_acc_flip_mean, r.x_id_acc_flip_sd if sd else None),
            f3(r.x_ood_acc_flip_mean, r.x_ood_acc_flip_sd if sd else None)))
        tex2.append("{} & {:g} & {} & {} & {} & {} \\\\".format(
            SET_LABEL[r.set].replace("→", "$\\to$"), r.lambda_, R.tex_pm(r.ood_acc_mean, r.ood_acc_sd if sd else None),
            R.tex_pm(r.ood_acc_flip_mean, r.ood_acc_flip_sd if sd else None),
            R.tex_pm(r.x_id_acc_mean, r.x_id_acc_sd if sd else None), R.tex_pm(r.x_ood_acc_mean, r.x_ood_acc_sd if sd else None)))
    L += ["", "λ = 0 per-seed validation thresholds (original score):", ""]
    for name in df.set.unique():
        r = df[(df.set == name) & (df["lambda"] == 0)]
        if len(r):
            L.append("- {}: {}".format(SET_LABEL[name], r.iloc[0].t0_per_seed))
    L += ["", "## LaTeX rows: S1 (set & λ & original & flipped & under d)", "", "```latex"] + tex1 + ["```", "",
          "## LaTeX rows: S2, λ > 0 (set & λ & FPR@95 original & FPR@95 flipped & transferred-t0 ID acc & transferred-t0 OOD acc)",
          "", "```latex"] + tex2 + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    les = lesion_block()
    p16, checks, p16_groups = p16_block()
    t0_les = {s: (v["rec"]["t_val"], v["rec"]["t_val_flip"]) for s, v in les[("pad_heldout", 0.0)].items()}
    t0 = {name: t0_les for name in LES_SETS}
    for pair in P16_SETS:
        t0[pair] = {s: (v["rec"]["t_val"], v["rec"]["t_val_flip"]) for s, v in p16[(pair, 0.0)].items()}
    rows = aggregate(les, lambda _n: R.id_test_groups(), ood_groups, t0)
    rows += aggregate(p16, lambda n: p16_groups[n][0], lambda n: p16_groups[n][1], t0)
    df = pd.DataFrame(rows)
    src = float(df[(df.set == "pad_heldout") & (df["lambda"] == 2.0)].auroc_mean.iloc[0])
    d_dir = "flip" if src < 0.5 else "keep"
    df["transfer_dir"] = d_dir
    df["auroc_under_d"] = df.flip_mean if d_dir == "flip" else df.auroc_mean
    df.to_csv(OUT / "summary.csv", index=False)
    runs = OUT / "runs"
    runs.mkdir(exist_ok=True)
    for block in (les, p16):
        for (name, lam), per in block.items():
            for s, v in per.items():
                R.write_json(runs / "{}_lam{:g}_s{}.json".format(name, lam, s), {"set": name, "lambda": lam, "seed": s,
                                                                                **v["rec"]})
    R.write_json(OUT / "results.json", {"commit": R.git_head(), "transfer_dir": d_dir, "pad_heldout_lam2_mean": src,
                                        "phase16_recompute_abs_diff": checks})
    write_report(df.rename(columns={"lambda": "lambda_"}).assign(**{"lambda": df["lambda"]}), d_dir, src, checks)
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
