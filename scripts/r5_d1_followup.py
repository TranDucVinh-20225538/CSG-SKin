#!/usr/bin/env python3
"""Paper B R5 D1 follow-up: placebo + dose (lesion-level item4 features only).

Decisions: decisions/paperB_r5_d1_followup.md. Never reads phase13/features.
Usage: r5_d1_followup.py [--stage placebo|dose|all]
  placebo = matched λ0 pairs + λ0→λ2 DANN arm (main test; unchanged).
  dose    = cross-seed: λ0 s42 → mid-λ s43–46 / λ2 s52–82 vs λ0 s42 → λ0 s52–82.
"""

from __future__ import annotations

import argparse
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
FEAT = R.FEAT_L  # item4 only
SEEDS = D1.SEEDS
PARTNER = {42: 52, 52: 62, 62: 72, 72: 82, 82: 42}
# Cross-seed dose (matched mid-λ not possible: R4 mid-λ seeds 42–46 only).
DOSE_SRC = 42
MID_PARTNERS = (43, 44, 45, 46)
END_PARTNERS = (52, 62, 72, 82)
LAMS_MID = (0.25, 0.5, 1.0, 4.0, 8.0)
LAMS_DOSE = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SETS = D1.LESION_SETS
REPS = ("latent", "backbone")
REP_KEY = D1.REPS
DET = D1.DET_REP
DELTA_MIN = 0.15
NEED = ("isic_train", "isic_test", "pad_full") + SETS


def _rho_dom_orth(dom, orth, pull, idx):
    rd, ro, rp = rankdata(dom[idx]), rankdata(orth[idx]), rankdata(pull[idx])
    return D1._r(rd, rp), D1._r(ro, rp)


def feat_ok(lam, seed):
    d = FEAT / R.run_tag(lam, seed)
    return all((d / "{}.npz".format(s)).exists() for s in NEED)


def load_pack(lam, seed):
    if not feat_ok(lam, seed):
        raise FileNotFoundError("missing item4 lesion-level features λ={} seed={}".format(lam, seed))
    tag = R.run_tag(lam, seed)
    return {s: R.load(tag, s, feat_root=FEAT) for s in NEED}


def domain_scores(A0, hmask):
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


def pull_arr(A0, Ak, s):
    y0 = A0["isic_train"]["labels"].astype(int)
    yk = Ak["isic_train"]["labels"].astype(int)
    sid0, sood0 = R.B.maha_scores(A0["isic_train"][DET], y0, A0["isic_test"][DET], A0[s][DET])
    sidk, soodk = R.B.maha_scores(Ak["isic_train"][DET], yk, Ak["isic_test"][DET], Ak[s][DET])
    return D1.pulls(sid0, sood0, sidk, soodk)["pull"]


def paired_delta_ci(per_seed_fn, seeds, groups, n_boot, rng):
    full = np.arange(len(groups))
    point = float(np.mean([per_seed_fn(s, full) for s in seeds]))
    draws = R.draws(groups, n_boot, rng)
    mat = np.array([np.mean([per_seed_fn(s, ii) for s in seeds]) for ii in draws])
    lo, hi = np.nanpercentile(mat, [2.5, 97.5])
    return {"mean": point, "ci": [float(lo), float(hi)], "excludes_0": bool(lo > 0 or hi < 0),
            "n_seeds": len(seeds), "seeds": list(seeds)}


def image_boot_ci(stat_fn, groups, n_boot, rng):
    """Paired bootstrap over images/groups; statistic is a scalar from index array."""
    full = np.arange(len(groups))
    point = float(stat_fn(full))
    draws = R.draws(groups, n_boot, rng)
    mat = np.array([float(stat_fn(ii)) for ii in draws])
    lo, hi = np.nanpercentile(mat, [2.5, 97.5])
    return {"mean": point, "ci": [float(lo), float(hi)], "excludes_0": bool(lo > 0 or hi < 0)}


def analyse_placebo(hmask, n_boot, seed):
    cache0 = {s: load_pack(0.0, s) for s in SEEDS}
    cache2 = {s: load_pack(2.0, s) for s in SEEDS}
    dom_by = {s: domain_scores(cache0[s], hmask) for s in SEEDS}
    pull_dann = {s: {sn: pull_arr(cache0[s], cache2[s], sn) for sn in SETS} for s in SEEDS}
    pull_plc = {s: {sn: pull_arr(cache0[s], cache0[PARTNER[s]], sn) for sn in SETS} for s in SEEDS}
    groups = {sn: D1.set_groups(sn, len(cache0[SEEDS[0]][sn]["labels"])) for sn in SETS}
    out = {"pairs": [{"source": s, "partner": PARTNER[s]} for s in SEEDS], "sets": {},
           "feature_root": str(FEAT), "split": "lesion-level item4"}
    for sname in SETS:
        g = groups[sname]
        blk = {"n": int(len(g)), "n_groups": int(len(np.unique(g))), "reps": {}}
        for rep in REPS:
            pts = {k: [] for k in ("dd", "do", "pd", "po")}
            for s in SEEDS:
                dom, orth = dom_by[s][rep][0][sname], dom_by[s][rep][1][sname]
                full = np.arange(len(dom))
                rd, ro = _rho_dom_orth(dom, orth, pull_dann[s][sname], full)
                pd_, po = _rho_dom_orth(dom, orth, pull_plc[s][sname], full)
                pts["dd"].append(rd); pts["do"].append(ro); pts["pd"].append(pd_); pts["po"].append(po)

            def delta_fn(src, idx, _rep=rep, _set=sname):
                dom = dom_by[src][_rep][0][_set]
                orth = dom_by[src][_rep][1][_set]
                rd, _ = _rho_dom_orth(dom, orth, pull_dann[src][_set], idx)
                pd_, _ = _rho_dom_orth(dom, orth, pull_plc[src][_set], idx)
                return rd - pd_

            dlt = paired_delta_ci(delta_fn, SEEDS, g, n_boot, np.random.default_rng(seed))
            met = bool(dlt["mean"] >= DELTA_MIN and dlt["ci"][0] > 0)
            blk["reps"][rep] = {
                "rho_dom_dann": float(np.mean(pts["dd"])), "rho_orth_dann": float(np.mean(pts["do"])),
                "rho_dom_placebo": float(np.mean(pts["pd"])), "rho_orth_placebo": float(np.mean(pts["po"])),
                "delta_rho_dom": dlt["mean"], "delta_rho_dom_ci": dlt["ci"],
                "delta_excludes_0": dlt["excludes_0"], "dann_specific_set": met,
            }
            print("placebo", sname, rep, "Δρ={:+.3f}".format(dlt["mean"]), dlt["ci"],
                  "met" if met else "no", flush=True)
        out["sets"][sname] = blk
    k = sum(1 for s in SETS if out["sets"][s]["reps"]["latent"]["dann_specific_set"])
    kb = sum(1 for s in SETS if out["sets"][s]["reps"]["backbone"]["dann_specific_set"])
    out["verdict"] = {"latent_n_met": k, "backbone_n_met": kb, "dann_specific": k >= 4,
                      "rule": "Δρ_dom ≥ 0.15 and CI entirely above 0 in ≥ 4/5 sets (latent); paired Δ only"}
    # also stash λ0→λ2 ρ_dom for the dose table
    out["lambda0_to_2"] = {
        s: {rep: {"rho_dom": out["sets"][s]["reps"][rep]["rho_dom_dann"],
                  "rho_orth": out["sets"][s]["reps"][rep]["rho_orth_dann"]} for rep in REPS}
        for s in SETS
    }
    return out


def analyse_dose(hmask, n_boot, seed, placebo=None):
    """Cross-seed dose: λ0 s42 → partners; Δρ_dom vs λ0 s42 → λ0 s52–82; test ρ(λ2)>ρ(λ0.25)."""
    for s in (DOSE_SRC,) + MID_PARTNERS:
        for lam in LAMS_MID:
            if not feat_ok(lam, s) and s in MID_PARTNERS:
                raise FileNotFoundError("missing item4 mid-λ features λ={} seed={}".format(lam, s))
    for s in (DOSE_SRC,) + END_PARTNERS:
        if not feat_ok(0.0, s):
            raise FileNotFoundError("missing item4 λ0 features seed={}".format(s))
        if not feat_ok(2.0, s) and s in END_PARTNERS:
            raise FileNotFoundError("missing item4 λ2 features seed={}".format(s))

    A0 = load_pack(0.0, DOSE_SRC)
    dom_src = domain_scores(A0, hmask)
    cache0_end = {s: load_pack(0.0, s) for s in END_PARTNERS}
    cache2_end = {s: load_pack(2.0, s) for s in END_PARTNERS}
    cache_mid = {lam: {s: load_pack(lam, s) for s in MID_PARTNERS} for lam in LAMS_MID}

    out = {
        "design": "cross-seed",
        "source": {"lambda": 0.0, "seed": DOSE_SRC},
        "mid_partners": list(MID_PARTNERS),
        "end_partners": list(END_PARTNERS),
        "lambdas": list(LAMS_DOSE),
        "sets": {},
        "feature_root": str(FEAT),
        "split": "lesion-level item4",
        "note": "matched mid-λ not possible (R4 mid-λ seeds 42–46); main §1 matched placebo unchanged",
    }

    for sname in SETS:
        g = D1.set_groups(sname, len(A0[sname]["labels"]))
        pulls_plc = {s: pull_arr(A0, cache0_end[s], sname) for s in END_PARTNERS}
        pulls_l2 = {s: pull_arr(A0, cache2_end[s], sname) for s in END_PARTNERS}
        pulls_mid = {lam: {s: pull_arr(A0, cache_mid[lam][s], sname) for s in MID_PARTNERS}
                     for lam in LAMS_MID}
        arm_pulls = {0.0: pulls_plc, 2.0: pulls_l2, **pulls_mid}
        blk = {"n": int(len(g)), "n_groups": int(len(np.unique(g))), "reps": {}}

        for rep in REPS:
            dom = dom_src[rep][0][sname]
            orth = dom_src[rep][1][sname]

            def mean_rho_dom(pulls, idx, _dom=dom, _orth=orth):
                return float(np.mean([_rho_dom_orth(_dom, _orth, pulls[s], idx)[0] for s in pulls]))

            def mean_rho_orth(pulls, idx, _dom=dom, _orth=orth):
                return float(np.mean([_rho_dom_orth(_dom, _orth, pulls[s], idx)[1] for s in pulls]))

            by_lam = {}
            rng_base = int(seed)
            for lam in (0.0,) + LAMS_DOSE:
                pulls = arm_pulls[lam]
                partners = list(pulls.keys())
                full = np.arange(len(dom))
                rd0, ro0 = mean_rho_dom(pulls, full), mean_rho_orth(pulls, full)

                def rho_fn(idx, _p=pulls):
                    return mean_rho_dom(_p, idx)

                rho_ci = image_boot_ci(rho_fn, g, n_boot, np.random.default_rng(rng_base + int(lam * 100)))

                if lam == 0.0:
                    dlt = {"mean": 0.0, "ci": [0.0, 0.0], "excludes_0": False}
                else:
                    def dlt_fn(idx, _p=pulls):
                        return mean_rho_dom(_p, idx) - mean_rho_dom(pulls_plc, idx)

                    dlt = image_boot_ci(dlt_fn, g, n_boot, np.random.default_rng(rng_base + 1000 + int(lam * 100)))

                by_lam[str(lam)] = {
                    "rho_dom": rd0, "rho_orth": ro0, "rho_dom_ci": rho_ci["ci"],
                    "delta_rho_dom_vs_xseed_placebo": dlt["mean"],
                    "delta_ci": dlt["ci"], "delta_excludes_0": dlt["excludes_0"],
                    "n_partners": len(partners), "partners": partners,
                    "target_lambda": lam,
                }

            def delta_l2_l025(idx):
                return mean_rho_dom(pulls_l2, idx) - mean_rho_dom(pulls_mid[0.25], idx)

            dlt25 = image_boot_ci(delta_l2_l025, g, n_boot, np.random.default_rng(rng_base + 2000))
            met = bool(dlt25["mean"] > 0 and dlt25["ci"][0] > 0)
            blk["reps"][rep] = {
                "by_lambda": by_lam,
                "delta_rho_dom_l2_minus_l025": dlt25["mean"], "delta_ci": dlt25["ci"],
                "l2_exceeds_l025": met,
                "partners_l2": list(END_PARTNERS), "partners_l025": list(MID_PARTNERS),
            }
            print("dose", sname, rep,
                  "ρ2={:+.3f}".format(by_lam["2.0"]["rho_dom"]),
                  "ρ0.25={:+.3f}".format(by_lam["0.25"]["rho_dom"]),
                  "δ={:+.3f}".format(dlt25["mean"]), dlt25["ci"],
                  "met" if met else "no", flush=True)
        out["sets"][sname] = blk

    k = sum(1 for s in SETS if out["sets"][s]["reps"]["latent"]["l2_exceeds_l025"])
    kb = sum(1 for s in SETS if out["sets"][s]["reps"]["backbone"]["l2_exceeds_l025"])
    out["verdict"] = {
        "latent_n_met": k, "backbone_n_met": kb, "n_sets": len(SETS),
        "l2_exceeds_l025": k >= 4,
        "rule": ("cross-seed: ρ_dom(λ2 s52–82)−ρ_dom(λ0.25 s43–46)>0 CI excludes 0 in ≥4/5 sets "
                 "(source λ0 s42; paired image bootstrap)"),
    }
    return out


def fci(mean, ci):
    if mean is None or ci is None or ci[0] is None:
        return "—"
    return "{:+.3f} [{:+.3f}, {:+.3f}]".format(mean, *ci)


def write_report(placebo, dose):
    L = ["# Paper B R5 D1 follow-up — placebo and dose", "",
         "Decisions: `decisions/paperB_r5_d1_followup.md`. Commit `{}`. "
         "**Features: lesion-level item4 only** (no phase13).".format(R.git_head()), "",
         "backbone = `backbone_raw_lesion` (pre-projection features of the same DANN-trained model; reversed gradient). "
         "pull = change from λ = 0 to λ = k. Domain score from the source λ = 0 model.", "",
         "## (1) Placebo: λ0→λ2 vs λ0→λ0 (different seed)", "",
         "Pairs: {}. DANN-specific if Δρ_dom ≥ {:.2f} and CI entirely above 0 in ≥ 4/5 sets (latent).".format(
             ", ".join("{}→{}".format(s, PARTNER[s]) for s in SEEDS), DELTA_MIN), "",
         "| set | rep | ρ_dom DANN (λ0→λ2) | ρ_dom placebo | Δρ_dom [CI] | DANN-specific |",
         "|---|---|---|---|---|---|"]
    rows = []
    for s in SETS:
        for rep in REPS:
            r = placebo["sets"][s]["reps"][rep]
            L.append("| {} | {} | {:+.3f} | {:+.3f} | {} | {} |".format(
                D1.SET_LABEL[s], rep, r["rho_dom_dann"], r["rho_dom_placebo"],
                fci(r["delta_rho_dom"], r["delta_rho_dom_ci"]),
                "met" if r["dann_specific_set"] else "not met"))
            rows.append({"item": "placebo", "set": s, "rep": rep, **{k: r[k] for k in r if k != "delta_rho_dom_ci"},
                         "delta_lo": r["delta_rho_dom_ci"][0], "delta_hi": r["delta_rho_dom_ci"][1]})
    v = placebo["verdict"]
    L += ["", "Placebo verdict: **{}/5** latent → DANN-specific {} (backbone {}/5).".format(
        v["latent_n_met"], "supported" if v["dann_specific"] else "not supported", v["backbone_n_met"]), ""]
    if dose is None:
        L += ["## (2) Dose — not run in this stage", "",
              "Cross-seed dose (λ0 s42 → mid-λ s43–46 / λ2 s52–82 vs λ0→λ0 s52–82) runs with `--stage dose`.", ""]
    else:
        L += ["## (2) Dose (cross-seed; deviation — matched mid-λ not possible)", "",
              "Source: λ = 0 seed {}. Mid-λ partners s{}; λ = 2 / placebo partners s{}. "
              "Δρ_dom = ρ_dom(level) − ρ_dom(λ0→λ0 cross-seed placebo); paired image bootstrap. "
              "Test: ρ_dom(λ2) > ρ_dom(λ0.25) (no monotonicity claim).".format(
                  DOSE_SRC, list(MID_PARTNERS), list(END_PARTNERS)), "",
              "### ρ_dom by λ (partner mean)", "",
              "| set | rep | xseed plc (λ0) | " + " | ".join("λ={:g}".format(x) for x in LAMS_DOSE)
              + " | δ(2−0.25) | 2>0.25 |",
              "|---|---|---:|" + "|".join(["---:"] * len(LAMS_DOSE)) + "|---|---|"]
        for s in SETS:
            for rep in REPS:
                r = dose["sets"][s]["reps"][rep]
                plc = r["by_lambda"].get("0.0", {})
                cells = []
                for lam in LAMS_DOSE:
                    b = r["by_lambda"].get(str(lam))
                    cells.append("—" if not b else "{:+.3f}".format(b["rho_dom"]))
                L.append("| {} | {} | {} | {} | {} | {} |".format(
                    D1.SET_LABEL[s], rep,
                    "—" if not plc else "{:+.3f}".format(plc["rho_dom"]),
                    " | ".join(cells),
                    fci(r["delta_rho_dom_l2_minus_l025"], r["delta_ci"]),
                    "met" if r["l2_exceeds_l025"] else "not met"))
                for lam in (0.0,) + LAMS_DOSE:
                    b = r["by_lambda"][str(lam)]
                    rows.append({
                        "item": "dose_xseed", "set": s, "rep": rep, "lambda": lam,
                        "rho_dom": b["rho_dom"], "rho_orth": b["rho_orth"],
                        "delta_vs_xseed_placebo": b["delta_rho_dom_vs_xseed_placebo"],
                        "delta_lo": b["delta_ci"][0], "delta_hi": b["delta_ci"][1],
                    })
        L += ["", "### Δρ_dom vs cross-seed placebo", "",
              "| set | rep | " + " | ".join("λ={:g}".format(x) for x in LAMS_DOSE) + " |",
              "|---|---|" + "|".join(["---:"] * len(LAMS_DOSE)) + "|"]
        for s in SETS:
            for rep in REPS:
                r = dose["sets"][s]["reps"][rep]
                cells = [fci(r["by_lambda"][str(lam)]["delta_rho_dom_vs_xseed_placebo"],
                             r["by_lambda"][str(lam)]["delta_ci"]) for lam in LAMS_DOSE]
                L.append("| {} | {} | {} |".format(D1.SET_LABEL[s], rep, " | ".join(cells)))
        vd = dose["verdict"]
        L += ["", "Dose test (cross-seed): **{}/5** latent → λ2>λ0.25 {} (backbone {}/5).".format(
            vd["latent_n_met"], "supported" if vd["l2_exceeds_l025"] else "not supported",
            vd["backbone_n_met"]), ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print("\n".join(L), flush=True)


def amend_parent_report():
    path = PARENT / "REPORT.md"
    if not path.exists() or "follow-up clarification" in path.read_text():
        return
    note = ("**Definitions (follow-up clarification):** backbone = `backbone_raw_lesion` (pre-projection features of "
            "the same DANN-trained model; they receive the reversed gradient). pull = change from λ = 0 to λ = k "
            "(here k = 2), same primary/secondary formulae as above.\n\n")
    text = path.read_text()
    path.write_text(text.replace("Domain classifier validity", note + "Domain classifier validity", 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("placebo", "dose", "all"), default="placebo",
                    help="placebo: matched §1; dose: cross-seed §2; all: both")
    a = ap.parse_args()
    pre = R.prereg()
    n_boot, seed = int(pre["n_resamples"]), int(pre["rng_seed"])
    hmask = pad_heldout_mask()
    OUT.mkdir(parents=True, exist_ok=True)
    placebo = dose = None
    if a.stage in ("placebo", "all"):
        print("placebo (matched item4)…", flush=True)
        placebo = analyse_placebo(hmask, n_boot, seed)
        R.write_json(OUT / "placebo.json", placebo)
    if a.stage in ("dose", "all"):
        print("dose (cross-seed item4)…", flush=True)
        dose = analyse_dose(hmask, n_boot, seed, placebo)
        R.write_json(OUT / "dose.json", dose)
    if placebo is None:
        import json
        if not (OUT / "placebo.json").exists():
            raise SystemExit("need placebo.json (run --stage placebo first) to write combined REPORT")
        placebo = json.loads((OUT / "placebo.json").read_text())
    write_report(placebo, dose)
    R.write_json(OUT / "D1_FOLLOWUP.json", {
        "commit": R.git_head(), "n_boot": n_boot, "rng_seed": seed, "stage": a.stage,
        "feature_root": str(FEAT), "placebo": placebo, "dose": dose,
    })
    amend_parent_report()
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
