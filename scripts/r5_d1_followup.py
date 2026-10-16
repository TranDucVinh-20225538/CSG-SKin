#!/usr/bin/env python3
"""Paper B R5 D1 follow-up: placebo (λ0→λ0 other seed) and dose (λ0→λk).

Decisions: decisions/paperB_r5_d1_followup.md (committed before any number). CPU; existing item4 features only.
Outputs: results/paperB/r5/d1_followup/{placebo.json,dose.json,summary.csv,REPORT.md}
Also amends parent d1_domain/REPORT.md with backbone / pull definitions (no number change).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402
import r5_d1_domain_direction as D1  # noqa: E402
from r5_p6_leakage_probe import pad_heldout_mask  # noqa: E402

OUT = R.R5 / "d1_followup"
PARENT = R.R5 / "d1_domain"
SEEDS = D1.SEEDS
PARTNER = {42: 52, 52: 62, 62: 72, 72: 82, 82: 42}
LAMS_DOSE = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SETS = D1.LESION_SETS
REPS = ("latent", "backbone")
REP_KEY = D1.REPS
DET = D1.DET_REP
DELTA_MIN = 0.15  # DANN-specific floor


def _rho_dom_orth(dom, orth, pull, idx):
    rd, ro, rp = rankdata(dom[idx]), rankdata(orth[idx]), rankdata(pull[idx])
    return D1._r(rd, rp), D1._r(ro, rp)


def load_pack(lam, seed, splits):
    return {s: R.load(R.run_tag(lam, seed), s) for s in splits}


def domain_scores(A0, hmask):
    """Fit ISIC↔pad_adv on λ = 0 pack A0; return {rep: (dom_by_set, orth_by_set)} for SETS + scoring fns."""
    out = {}
    for rep_name, rep in REP_KEY.items():
        X0, X1 = A0["isic_train"][rep], A0["pad_full"][rep][~hmask]
        _clf, _sc, score = D1.domain_fit(X0, X1)
        dom, orth = {}, {}
        for s in SETS:
            d, o = score(A0[s][rep])
            dom[s], orth[s] = d, o
        out[rep_name] = (dom, orth)
    return out


def pull_pair(A0, Ak, s):
    y0 = A0["isic_train"]["labels"].astype(int)
    yk = Ak["isic_train"]["labels"].astype(int)
    sid0, sood0 = R.B.maha_scores(A0["isic_train"][DET], y0, A0["isic_test"][DET], A0[s][DET])
    sidk, soodk = R.B.maha_scores(Ak["isic_train"][DET], yk, Ak["isic_test"][DET], Ak[s][DET])
    return D1.pulls(sid0, sood0, sidk, soodk)


def paired_delta_ci(per_seed_fn, groups, n_boot, rng):
    """per_seed_fn(seed, idx) -> float. Returns mean over seeds on full data + CI of seed-mean over draws."""
    seeds = list(SEEDS)
    full = np.arange(len(groups))
    point = float(np.mean([per_seed_fn(s, full) for s in seeds]))
    draws = R.draws(groups, n_boot, rng)
    mat = np.array([np.mean([per_seed_fn(s, ii) for s in seeds]) for ii in draws])
    lo, hi = np.nanpercentile(mat, [2.5, 97.5])
    return {"mean": point, "ci": [float(lo), float(hi)], "excludes_0": bool(lo > 0 or hi < 0),
            "n_seeds": len(seeds)}


def analyse_placebo(dom_by_seed, pull_dann, pull_plc, groups_by_set, n_boot, seed):
    """pull_*: {source_seed: {set: pull array}} precomputed."""
    out = {"pairs": [{"source": s, "partner": PARTNER[s]} for s in SEEDS], "sets": {}}
    for sname in SETS:
        g = groups_by_set[sname]
        blk = {"n": int(len(g)), "n_groups": int(len(np.unique(g))), "reps": {}}
        for rep in REPS:
            dann_dom, dann_orth, plc_dom, plc_orth = [], [], [], []
            for s in SEEDS:
                dom, orth = dom_by_seed[s][rep][0][sname], dom_by_seed[s][rep][1][sname]
                full = np.arange(len(dom))
                rd, ro = _rho_dom_orth(dom, orth, pull_dann[s][sname], full)
                pd_, po = _rho_dom_orth(dom, orth, pull_plc[s][sname], full)
                dann_dom.append(rd); dann_orth.append(ro); plc_dom.append(pd_); plc_orth.append(po)

            def delta_fn(src, idx, _rep=rep, _set=sname):
                dom = dom_by_seed[src][_rep][0][_set]
                orth = dom_by_seed[src][_rep][1][_set]
                rd, _ = _rho_dom_orth(dom, orth, pull_dann[src][_set], idx)
                pd_, _ = _rho_dom_orth(dom, orth, pull_plc[src][_set], idx)
                return rd - pd_

            def arm_fn(arm):
                pulls = pull_dann if arm == "dann" else pull_plc

                def f(src, idx, _rep=rep, _set=sname, _p=pulls):
                    dom = dom_by_seed[src][_rep][0][_set]
                    orth = dom_by_seed[src][_rep][1][_set]
                    rd, _ = _rho_dom_orth(dom, orth, _p[src][_set], idx)
                    return rd
                return f

            dlt = paired_delta_ci(delta_fn, g, n_boot, np.random.default_rng(seed))
            ci_dann = paired_delta_ci(arm_fn("dann"), g, n_boot, np.random.default_rng(seed))
            ci_plc = paired_delta_ci(arm_fn("plc"), g, n_boot, np.random.default_rng(seed))
            met = bool(dlt["mean"] >= DELTA_MIN and dlt["ci"][0] > 0)
            blk["reps"][rep] = {
                "rho_dom_dann": float(np.mean(dann_dom)), "rho_orth_dann": float(np.mean(dann_orth)),
                "rho_dom_placebo": float(np.mean(plc_dom)), "rho_orth_placebo": float(np.mean(plc_orth)),
                "rho_dom_dann_ci": ci_dann["ci"], "rho_dom_placebo_ci": ci_plc["ci"],
                "delta_rho_dom": dlt["mean"], "delta_rho_dom_ci": dlt["ci"],
                "delta_excludes_0": dlt["excludes_0"], "dann_specific_set": met,
            }
            print("placebo", sname, rep, "Δρ_dom={:+.3f} CI=[{:+.3f},{:+.3f}] met={}".format(
                dlt["mean"], *dlt["ci"], met), flush=True)
        out["sets"][sname] = blk
    k_lat = sum(1 for s in SETS if out["sets"][s]["reps"]["latent"]["dann_specific_set"])
    k_bb = sum(1 for s in SETS if out["sets"][s]["reps"]["backbone"]["dann_specific_set"])
    out["verdict"] = {
        "latent_n_met": k_lat, "backbone_n_met": k_bb,
        "dann_specific": bool(k_lat >= 4),
        "rule": "Δρ_dom ≥ 0.15 and CI entirely above 0 in ≥ 4/5 sets (latent); paired Δ only",
    }
    return out


def analyse_dose(dom_by_seed, pull_by_lam, groups_by_set, n_boot, seed):
    """pull_by_lam: {lam: {source_seed: {set: pull}}}."""
    out = {"lambdas": list(LAMS_DOSE), "sets": {}}
    for sname in SETS:
        g = groups_by_set[sname]
        blk = {"reps": {}}
        for rep in REPS:
            by_lam = {}
            for lam in LAMS_DOSE:
                rhos = []
                for s in SEEDS:
                    dom = dom_by_seed[s][rep][0][sname]
                    orth = dom_by_seed[s][rep][1][sname]
                    rd, ro = _rho_dom_orth(dom, orth, pull_by_lam[lam][s][sname], np.arange(len(dom)))
                    rhos.append((rd, ro))

                def arm_fn(src, idx, _lam=lam, _rep=rep, _set=sname):
                    dom = dom_by_seed[src][_rep][0][_set]
                    orth = dom_by_seed[src][_rep][1][_set]
                    rd, _ = _rho_dom_orth(dom, orth, pull_by_lam[_lam][src][_set], idx)
                    return rd

                ci = paired_delta_ci(arm_fn, g, n_boot, np.random.default_rng(seed))
                by_lam[str(lam)] = {
                    "rho_dom": float(np.mean([r[0] for r in rhos])),
                    "rho_orth": float(np.mean([r[1] for r in rhos])),
                    "rho_dom_ci": ci["ci"],
                }

            def delta_2_vs_025(src, idx, _rep=rep, _set=sname):
                dom = dom_by_seed[src][_rep][0][_set]
                orth = dom_by_seed[src][_rep][1][_set]
                r2, _ = _rho_dom_orth(dom, orth, pull_by_lam[2.0][src][_set], idx)
                r025, _ = _rho_dom_orth(dom, orth, pull_by_lam[0.25][src][_set], idx)
                return r2 - r025

            dlt = paired_delta_ci(delta_2_vs_025, g, n_boot, np.random.default_rng(seed))
            met = bool(dlt["mean"] > 0 and dlt["ci"][0] > 0)
            blk["reps"][rep] = {
                "by_lambda": by_lam,
                "delta_rho_dom_l2_minus_l025": dlt["mean"],
                "delta_ci": dlt["ci"], "excludes_0": dlt["excludes_0"],
                "l2_exceeds_l025": met,
            }
            print("dose", sname, rep, "δ(2−0.25)={:+.3f} CI=[{:+.3f},{:+.3f}] met={}".format(
                dlt["mean"], *dlt["ci"], met), flush=True)
        out["sets"][sname] = blk
    k_lat = sum(1 for s in SETS if out["sets"][s]["reps"]["latent"]["l2_exceeds_l025"])
    k_bb = sum(1 for s in SETS if out["sets"][s]["reps"]["backbone"]["l2_exceeds_l025"])
    out["verdict"] = {
        "latent_n_met": k_lat, "backbone_n_met": k_bb,
        "l2_exceeds_l025": bool(k_lat >= 4),
        "rule": "ρ_dom(λ0→λ2) − ρ_dom(λ0→λ0.25) > 0 with CI excluding 0 in ≥ 4/5 sets (latent); no monotonicity claim",
    }
    return out


def fci(mean, ci):
    return "{:+.3f} [{:+.3f}, {:+.3f}]".format(mean, *ci)


def write_report(placebo, dose):
    L = ["# Paper B R5 D1 follow-up — placebo and dose", "",
         "Decisions: `decisions/paperB_r5_d1_followup.md`. Commit `{}`. Parent D1 unchanged in criterion.".format(
             R.git_head()), "",
         "backbone = `backbone_raw_lesion` (pre-projection features of the same DANN-trained model; they receive the "
         "reversed gradient). pull = change from λ = 0 to λ = k (Mahalanobis; primary formula as D1). Domain score "
         "always from the source λ = 0 model.", "",
         "## (1) Placebo: λ0→λ2 vs λ0→λ0 (different seed)", "",
         "Seed pairs (source → partner): {}. Primary: paired bootstrap of "
         "Δρ_dom = ρ_dom(λ0→λ2) − ρ_dom(placebo). DANN-specific if Δρ_dom ≥ {:.2f} and CI entirely above 0 in "
         "≥ 4/5 sets (latent). Separate arm CIs are descriptive only.".format(
             ", ".join("{}→{}".format(s, PARTNER[s]) for s in SEEDS), DELTA_MIN), "",
         "| set | rep | ρ_dom DANN | ρ_dom placebo | Δρ_dom [CI] | DANN-specific |",
         "|---|---|---|---|---|---|"]
    rows = []
    for s in SETS:
        for rep in REPS:
            r = placebo["sets"][s]["reps"][rep]
            L.append("| {} | {} | {:+.3f} | {:+.3f} | {} | {} |".format(
                D1.SET_LABEL[s], rep, r["rho_dom_dann"], r["rho_dom_placebo"],
                fci(r["delta_rho_dom"], r["delta_rho_dom_ci"]),
                "met" if r["dann_specific_set"] else "not met"))
            rows.append({"item": "placebo", "set": s, "rep": rep, **{k: r[k] for k in (
                "rho_dom_dann", "rho_orth_dann", "rho_dom_placebo", "rho_orth_placebo",
                "delta_rho_dom", "dann_specific_set")}})
    v = placebo["verdict"]
    L += ["", "Placebo verdict (latent): **{}/5** sets met → DANN-specific {}.".format(
        v["latent_n_met"], "supported" if v["dann_specific"] else "not supported"),
          "Backbone (secondary): {}/5.".format(v["backbone_n_met"]), "",
          "## (2) Dose: λ0 → λ ∈ {{{}}}".format(", ".join(str(x) for x in LAMS_DOSE)), "",
          "Prediction / test only: ρ_dom(λ0→λ2) exceeds ρ_dom(λ0→λ0.25) (paired). No monotonicity prediction.", "",
          "| set | rep | " + " | ".join("ρ_dom λ={:g}".format(x) for x in LAMS_DOSE) +
          " | δ(2−0.25) [CI] | 2>0.25 |",
          "|---|---|" + "|".join(["---:"] * len(LAMS_DOSE)) + "|---|---|"]
    for s in SETS:
        for rep in REPS:
            r = dose["sets"][s]["reps"][rep]
            cells = ["{:+.3f}".format(r["by_lambda"][str(lam)]["rho_dom"]) for lam in LAMS_DOSE]
            L.append("| {} | {} | {} | {} | {} |".format(
                D1.SET_LABEL[s], rep, " | ".join(cells),
                fci(r["delta_rho_dom_l2_minus_l025"], r["delta_ci"]),
                "met" if r["l2_exceeds_l025"] else "not met"))
            for lam in LAMS_DOSE:
                b = r["by_lambda"][str(lam)]
                rows.append({"item": "dose", "set": s, "rep": rep, "lambda": lam,
                             "rho_dom": b["rho_dom"], "rho_orth": b["rho_orth"]})
            rows.append({"item": "dose_test", "set": s, "rep": rep,
                         "delta_l2_minus_l025": r["delta_rho_dom_l2_minus_l025"],
                         "l2_exceeds_l025": r["l2_exceeds_l025"]})
    vd = dose["verdict"]
    L += ["", "Dose test (latent): **{}/5** sets with λ=2 > λ=0.25 (CI excludes 0) → {}.".format(
        vd["latent_n_met"], "supported" if vd["l2_exceeds_l025"] else "not supported"),
          "Backbone (secondary): {}/5. λ=4 and λ=8 are descriptive only (Table 1 non-monotone).".format(
              vd["backbone_n_met"]), "",
          "## Predictions vs observed", "",
          "- Placebo DANN-specific (pred. 4/5): observed {}/5 → {}.".format(
              v["latent_n_met"], "supported" if v["dann_specific"] else "not supported"),
          "- Dose λ2 > λ0.25 (pred. ≥4/5): observed {}/5 → {}.".format(
              vd["latent_n_met"], "supported" if vd["l2_exceeds_l025"] else "not supported"), ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print("\n".join(L), flush=True)


def amend_parent_report():
    path = PARENT / "REPORT.md"
    if not path.exists():
        return
    text = path.read_text()
    note = ("**Definitions (follow-up clarification):** backbone = `backbone_raw_lesion` (pre-projection features of "
            "the same DANN-trained model; they receive the reversed gradient). pull = change from λ = 0 to λ = k "
            "(here k = 2), same primary/secondary formulae as above.\n\n")
    if "follow-up clarification" in text:
        return
    # insert after the first blank line following the title block / criterion paragraph
    marker = "Domain classifier validity"
    if marker in text:
        text = text.replace(marker, note + marker, 1)
    else:
        text = note + text
    path.write_text(text)


def main():
    pre = R.prereg()
    n_boot, seed = int(pre["n_resamples"]), int(pre["rng_seed"])
    hmask = pad_heldout_mask()
    need = ("isic_train", "isic_test", "pad_full") + SETS
    print("loading all λ packs…", flush=True)
    cache0 = {s: load_pack(0.0, s, need) for s in SEEDS}
    cache_k = {lam: {s: load_pack(lam, s, need) for s in SEEDS} for lam in LAMS_DOSE}
    print("domain scores…", flush=True)
    dom_by_seed = {s: domain_scores(cache0[s], hmask) for s in SEEDS}
    groups_by_set = {s: D1.set_groups(s, len(cache0[SEEDS[0]][s]["labels"])) for s in SETS}
    print("precompute pulls…", flush=True)
    pull_dann = {s: {sn: pull_pair(cache0[s], cache_k[2.0][s], sn)["pull"] for sn in SETS} for s in SEEDS}
    pull_plc = {s: {sn: pull_pair(cache0[s], cache0[PARTNER[s]], sn)["pull"] for sn in SETS} for s in SEEDS}
    pull_by_lam = {lam: {s: {sn: pull_pair(cache0[s], cache_k[lam][s], sn)["pull"] for sn in SETS}
                         for s in SEEDS} for lam in LAMS_DOSE}
    print("placebo…", flush=True)
    placebo = analyse_placebo(dom_by_seed, pull_dann, pull_plc, groups_by_set, n_boot, seed)
    print("dose…", flush=True)
    dose = analyse_dose(dom_by_seed, pull_by_lam, groups_by_set, n_boot, seed)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"commit": R.git_head(), "n_boot": n_boot, "rng_seed": seed, "placebo": placebo, "dose": dose}
    R.write_json(OUT / "placebo.json", placebo)
    R.write_json(OUT / "dose.json", dose)
    R.write_json(OUT / "D1_FOLLOWUP.json", payload)
    write_report(placebo, dose)
    amend_parent_report()
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
