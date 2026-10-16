#!/usr/bin/env python3
"""Paper B R5 D1 follow-up: placebo + dose (lesion-level item4 features only).

Decisions: decisions/paperB_r5_d1_followup.md. Never reads phase13/features.
Usage: r5_d1_followup.py [--stage placebo|dose|all]
  placebo = λ0 pairs + λ0→λ2 DANN arm (available now on item4).
  dose    = all λ levels (mid-λ: item4 seeds 42–46 exist; matched with λ0 only seed 42 → pending).
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
    """Dose over LAMS_DOSE; skip λ with <3 matched lesion-level seeds. λ=2 always available."""
    cache0 = {s: load_pack(0.0, s) for s in SEEDS}
    dom_by = {s: domain_scores(cache0[s], hmask) for s in SEEDS}
    out = {"lambdas": list(LAMS_DOSE), "sets": {}, "pending_lams": [],
           "feature_root": str(FEAT), "split": "lesion-level item4"}
    for lam in LAMS_DOSE:
        seeds = [s for s in SEEDS if feat_ok(lam, s) and feat_ok(0.0, s)]
        if len(seeds) < 3:
            mid = [s for s in (42, 43, 44, 45, 46) if feat_ok(lam, s)]
            out["pending_lams"].append({
                "lambda": lam, "matched_seeds": seeds, "mid_lam_item4_seeds": mid,
                "reason": ("R4/item4 mid-λ features exist for seeds 42–46; λ0 only for 42–82; "
                           "matched λ0∩mid-λ={} (need ≥3). No lesion mid-λ at 52–82.".format(seeds)),
            })
    for sname in SETS:
        g = D1.set_groups(sname, len(cache0[SEEDS[0]][sname]["labels"]))
        blk = {"reps": {}}
        for rep in REPS:
            by_lam, pull_cache = {}, {}
            for lam in LAMS_DOSE:
                seeds = [s for s in SEEDS if feat_ok(lam, s)]
                if len(seeds) < 3:
                    by_lam[str(lam)] = None
                    continue
                pulls, rhos = {}, []
                cache_k = {s: load_pack(lam, s) for s in seeds}
                for s in seeds:
                    pulls[s] = pull_arr(cache0[s], cache_k[s], sname)
                    dom = dom_by[s][rep][0][sname]
                    orth = dom_by[s][rep][1][sname]
                    rd, ro = _rho_dom_orth(dom, orth, pulls[s], np.arange(len(dom)))
                    rhos.append((rd, ro))
                pull_cache[lam] = pulls

                def arm_fn(src, idx, _lam=lam, _rep=rep, _set=sname):
                    dom = dom_by[src][_rep][0][_set]
                    orth = dom_by[src][_rep][1][_set]
                    rd, _ = _rho_dom_orth(dom, orth, pull_cache[_lam][src], idx)
                    return rd

                ci = paired_delta_ci(arm_fn, seeds, g, n_boot, np.random.default_rng(seed))
                by_lam[str(lam)] = {
                    "rho_dom": float(np.mean([r[0] for r in rhos])),
                    "rho_orth": float(np.mean([r[1] for r in rhos])),
                    "rho_dom_ci": ci["ci"], "n_seeds": len(seeds), "seeds": seeds,
                }

            seeds_test = []
            if by_lam.get("2.0") and by_lam.get("0.25"):
                seeds_test = sorted(set(by_lam["2.0"]["seeds"]) & set(by_lam["0.25"]["seeds"]))
            if len(seeds_test) >= 3 and 2.0 in pull_cache and 0.25 in pull_cache:
                def delta_fn(src, idx, _rep=rep, _set=sname):
                    dom = dom_by[src][_rep][0][_set]
                    orth = dom_by[src][_rep][1][_set]
                    r2, _ = _rho_dom_orth(dom, orth, pull_cache[2.0][src], idx)
                    r025, _ = _rho_dom_orth(dom, orth, pull_cache[0.25][src], idx)
                    return r2 - r025

                dlt = paired_delta_ci(delta_fn, seeds_test, g, n_boot, np.random.default_rng(seed))
                met = bool(dlt["mean"] > 0 and dlt["ci"][0] > 0)
            else:
                dlt = {"mean": None, "ci": [None, None], "n_seeds": len(seeds_test), "seeds": seeds_test}
                met = False
            blk["reps"][rep] = {
                "by_lambda": by_lam,
                "delta_rho_dom_l2_minus_l025": dlt["mean"], "delta_ci": dlt["ci"],
                "l2_exceeds_l025": met, "seeds_test": dlt.get("seeds"),
            }
            print("dose", sname, rep, "δ=", dlt["mean"], flush=True)
        out["sets"][sname] = blk
    tested = [s for s in SETS if out["sets"][s]["reps"]["latent"]["delta_rho_dom_l2_minus_l025"] is not None]
    k = sum(1 for s in tested if out["sets"][s]["reps"]["latent"]["l2_exceeds_l025"])
    out["verdict"] = {
        "sets_tested": tested, "latent_n_met": k, "n_sets_tested": len(tested),
        "l2_exceeds_l025": bool(len(tested) >= 4 and k >= 4),
        "pending_lams": out["pending_lams"],
        "rule": "ρ_dom(λ2)−ρ_dom(λ0.25)>0 CI excludes 0 in ≥4/5 sets when mid-λ lesion features exist",
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
        L += ["## (2) Dose — pending matched mid-λ seeds", "",
              "λ0→λ2 ρ_dom is in the placebo DANN column above. R4 Table 1 mid-λ used item4 seeds **42–46** "
              "(features already under `r2/item4/features/`); λ0/λ2 used 42–82. Matched λ0∩mid-λ = {42} only, "
              "so the preregistered ≥3-seed dose / δ(λ2−λ0.25) test is not run. No lesion mid-λ at 52–82 "
              "(phase3/phase13 are image-level; not used).", ""]
    else:
        L += ["## (2) Dose", "",
              "Pending λ (insufficient matched lesion-level seeds): {}.".format(
                  ", ".join("{:g}".format(p["lambda"]) for p in dose.get("pending_lams", [])) or "none"), "",
              "| set | rep | " + " | ".join("λ={:g}".format(x) for x in LAMS_DOSE) + " | δ(2−0.25) | 2>0.25 |",
              "|---|---|" + "|".join(["---:"] * len(LAMS_DOSE)) + "|---|---|"]
        for s in SETS:
            for rep in REPS:
                r = dose["sets"][s]["reps"][rep]
                cells = []
                for lam in LAMS_DOSE:
                    b = r["by_lambda"].get(str(lam))
                    cells.append("—" if not b else "{:+.3f}".format(b["rho_dom"]))
                L.append("| {} | {} | {} | {} | {} |".format(
                    D1.SET_LABEL[s], rep, " | ".join(cells),
                    fci(r["delta_rho_dom_l2_minus_l025"], r["delta_ci"]),
                    "met" if r["l2_exceeds_l025"] else ("—" if r["delta_rho_dom_l2_minus_l025"] is None else "not met")))
        vd = dose["verdict"]
        L += ["", "Dose test: {}/{} sets met (tested {}).".format(
            vd["latent_n_met"], vd["n_sets_tested"], vd.get("sets_tested")), ""]
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
                    help="placebo (default): λ0 pairs + λ0→λ2 now; dose/all when mid-λ item4 features exist")
    a = ap.parse_args()
    pre = R.prereg()
    n_boot, seed = int(pre["n_resamples"]), int(pre["rng_seed"])
    hmask = pad_heldout_mask()
    OUT.mkdir(parents=True, exist_ok=True)
    placebo = dose = None
    if a.stage in ("placebo", "all"):
        print("placebo (item4 lesion-level)…", flush=True)
        placebo = analyse_placebo(hmask, n_boot, seed)
        R.write_json(OUT / "placebo.json", placebo)
    if a.stage in ("dose", "all"):
        print("dose (item4 lesion-level)…", flush=True)
        if placebo is None and (OUT / "placebo.json").exists():
            import json
            placebo = json.loads((OUT / "placebo.json").read_text())
        dose = analyse_dose(hmask, n_boot, seed, placebo)
        R.write_json(OUT / "dose.json", dose)
    if placebo is None:
        import json
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
