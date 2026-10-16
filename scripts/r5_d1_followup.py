#!/usr/bin/env python3
"""Paper B R5 D1 follow-up: placebo (λ0→λ0 other seed) and dose (λ0→λk).

Decisions: decisions/paperB_r5_d1_followup.md. CPU.
Feature roots (deviation): item4 for P7a + λ∈{0,2}; phase13 for pad_heldout/Fitzpatrick across mid-λ
with seeds 42/52/62/72/82 when present.
Outputs: results/paperB/r5/d1_followup/
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
FEAT_I4 = R.FEAT_L
FEAT_P13 = R.FEAT_I
SEEDS = D1.SEEDS
PARTNER = {42: 52, 52: 62, 62: 72, 72: 82, 82: 42}
LAMS_DOSE = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SETS_FULL = D1.LESION_SETS  # all five for placebo
SETS_P13 = ("pad_heldout", "fitzpatrick17k")  # dose mid-λ coverage
SETS_P7A = ("mednode", "sd198", "scin")
REPS = ("latent", "backbone")
REP_KEY = D1.REPS
DET = D1.DET_REP
DELTA_MIN = 0.15
SPLITS_CORE = ("isic_train", "isic_test", "pad_full")


def _rho_dom_orth(dom, orth, pull, idx):
    rd, ro, rp = rankdata(dom[idx]), rankdata(orth[idx]), rankdata(pull[idx])
    return D1._r(rd, rp), D1._r(ro, rp)


def has_splits(root, tag, splits):
    d = Path(root) / tag
    return all((d / "{}.npz".format(s)).exists() for s in splits)


def resolve_root(lam, seed, sets_needed):
    """Prefer item4 if it has all needed OOD splits; else phase13 (pad/fitz only)."""
    tag = R.run_tag(lam, seed)
    need = SPLITS_CORE + tuple(sets_needed)
    if has_splits(FEAT_I4, tag, need):
        return FEAT_I4
    if set(sets_needed) <= set(SETS_P13) and has_splits(FEAT_P13, tag, SPLITS_CORE + tuple(sets_needed)):
        return FEAT_P13
    return None


def load_pack(lam, seed, sets_needed):
    root = resolve_root(lam, seed, sets_needed)
    if root is None:
        raise FileNotFoundError("no features for λ={} seed={} sets={}".format(lam, seed, sets_needed))
    tag = R.run_tag(lam, seed)
    splits = SPLITS_CORE + tuple(sets_needed)
    return {s: R.load(tag, s, feat_root=root) for s in splits}, root


def seeds_for(lam, sets_needed):
    return [s for s in SEEDS if resolve_root(lam, s, sets_needed) is not None]


def domain_scores(A0, hmask, sets):
    out = {}
    for rep_name, rep in REP_KEY.items():
        X0, X1 = A0["isic_train"][rep], A0["pad_full"][rep][~hmask]
        _clf, _sc, score = D1.domain_fit(X0, X1)
        dom, orth = {}, {}
        for s in sets:
            d, o = score(A0[s][rep])
            dom[s], orth[s] = d, o
        out[rep_name] = (dom, orth)
    return out


def pull_pair(A0, Ak, s):
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


def set_groups(name, n):
    return D1.set_groups(name, n)


def analyse_placebo(hmask, n_boot, seed):
    """Placebo on all 5 sets; item4 λ0/λ2 seeds 42–82."""
    sets = SETS_FULL
    cache0, cache2, dom_by = {}, {}, {}
    for s in SEEDS:
        cache0[s], _ = load_pack(0.0, s, sets)
        cache2[s], _ = load_pack(2.0, s, sets)
        dom_by[s] = domain_scores(cache0[s], hmask, sets)
    pull_dann = {s: {sn: pull_pair(cache0[s], cache2[s], sn) for sn in sets} for s in SEEDS}
    pull_plc = {s: {sn: pull_pair(cache0[s], cache0[PARTNER[s]], sn) for sn in sets} for s in SEEDS}
    groups = {sn: set_groups(sn, len(cache0[SEEDS[0]][sn]["labels"])) for sn in sets}
    out = {"pairs": [{"source": s, "partner": PARTNER[s]} for s in SEEDS], "sets": {},
           "feature_note": "item4 λ∈{0,2} × seeds 42–82, all 5 lesion-level sets"}
    for sname in sets:
        g = groups[sname]
        blk = {"n": int(len(g)), "n_groups": int(len(np.unique(g))), "reps": {}}
        for rep in REPS:
            pts_d, pts_o, pts_pd, pts_po = [], [], [], []
            for s in SEEDS:
                dom, orth = dom_by[s][rep][0][sname], dom_by[s][rep][1][sname]
                full = np.arange(len(dom))
                rd, ro = _rho_dom_orth(dom, orth, pull_dann[s][sname], full)
                pd_, po = _rho_dom_orth(dom, orth, pull_plc[s][sname], full)
                pts_d.append(rd); pts_o.append(ro); pts_pd.append(pd_); pts_po.append(po)

            def delta_fn(src, idx, _rep=rep, _set=sname):
                dom = dom_by[src][_rep][0][_set]
                orth = dom_by[src][_rep][1][_set]
                rd, _ = _rho_dom_orth(dom, orth, pull_dann[src][_set], idx)
                pd_, _ = _rho_dom_orth(dom, orth, pull_plc[src][_set], idx)
                return rd - pd_

            dlt = paired_delta_ci(delta_fn, SEEDS, g, n_boot, np.random.default_rng(seed))
            met = bool(dlt["mean"] >= DELTA_MIN and dlt["ci"][0] > 0)
            blk["reps"][rep] = {
                "rho_dom_dann": float(np.mean(pts_d)), "rho_orth_dann": float(np.mean(pts_o)),
                "rho_dom_placebo": float(np.mean(pts_pd)), "rho_orth_placebo": float(np.mean(pts_po)),
                "delta_rho_dom": dlt["mean"], "delta_rho_dom_ci": dlt["ci"],
                "delta_excludes_0": dlt["excludes_0"], "dann_specific_set": met,
                "n_seeds": 5,
            }
            print("placebo", sname, rep, "Δρ={:+.3f} {}".format(dlt["mean"], dlt["ci"]), "met" if met else "no",
                  flush=True)
        out["sets"][sname] = blk
    k = sum(1 for s in sets if out["sets"][s]["reps"]["latent"]["dann_specific_set"])
    kb = sum(1 for s in sets if out["sets"][s]["reps"]["backbone"]["dann_specific_set"])
    out["verdict"] = {"latent_n_met": k, "backbone_n_met": kb, "dann_specific": k >= 4,
                      "rule": "Δρ_dom ≥ 0.15 and CI > 0 in ≥ 4/5 sets (latent); paired Δ only"}
    return out


def analyse_dose(hmask, n_boot, seed):
    """Dose: pad/fitz via phase13 (or item4) with matched seeds; P7a mid-λ skipped (no matched λ0)."""
    out = {"lambdas": list(LAMS_DOSE), "sets": {},
           "feature_note": "pad_heldout/Fitzpatrick: phase13 (fallback item4); P7a: item4 λ∈{0,2} only for dose"}
    # preload λ=0 for all seeds (item4)
    cache0 = {}
    for s in SEEDS:
        cache0[s], _ = load_pack(0.0, s, SETS_FULL)
    dom_full = {s: domain_scores(cache0[s], hmask, SETS_FULL) for s in SEEDS}

    for sname in SETS_FULL:
        g = set_groups(sname, len(cache0[SEEDS[0]][sname]["labels"]))
        blk = {"reps": {}}
        for rep in REPS:
            by_lam = {}
            pull_cache = {}  # lam -> {seed: pull}
            for lam in LAMS_DOSE:
                if sname in SETS_P7A and lam not in (2.0,):
                    # mid-λ P7a: no matched λ0 for seeds 43–46; skip (report null)
                    by_lam[str(lam)] = None
                    continue
                sets_need = (sname,) if sname in SETS_P13 else SETS_FULL
                # for P7a only λ=2 here; for pad/fitz any lam
                use_sets = SETS_P13 if sname in SETS_P13 else (sname,)
                seeds = [s for s in SEEDS if resolve_root(lam, s, use_sets) and resolve_root(0.0, s, SETS_FULL)]
                if sname in SETS_P13:
                    seeds = [s for s in SEEDS if resolve_root(lam, s, SETS_P13)]
                if len(seeds) < 3:
                    by_lam[str(lam)] = {"rho_dom": None, "rho_orth": None, "rho_dom_ci": None,
                                        "n_seeds": len(seeds), "seeds": seeds, "note": "too few matched seeds"}
                    continue
                pulls, rhos = {}, []
                for s in seeds:
                    Ak, _ = load_pack(lam, s, SETS_P13 if sname in SETS_P13 else (sname,))
                    # λ0 pack may need only this set for pull; domain from full cache0
                    A0 = cache0[s]
                    if sname not in A0:
                        A0, _ = load_pack(0.0, s, (sname,))
                    pulls[s] = pull_pair(A0, Ak, sname)
                    dom = dom_full[s][rep][0][sname]
                    orth = dom_full[s][rep][1][sname]
                    rd, ro = _rho_dom_orth(dom, orth, pulls[s], np.arange(len(dom)))
                    rhos.append((rd, ro))
                pull_cache[lam] = pulls

                def arm_fn(src, idx, _lam=lam, _rep=rep, _set=sname, _seeds=tuple(seeds)):
                    if src not in pull_cache[_lam]:
                        return np.nan
                    dom = dom_full[src][_rep][0][_set]
                    orth = dom_full[src][_rep][1][_set]
                    rd, _ = _rho_dom_orth(dom, orth, pull_cache[_lam][src], idx)
                    return rd

                ci = paired_delta_ci(arm_fn, seeds, g, n_boot, np.random.default_rng(seed))
                by_lam[str(lam)] = {
                    "rho_dom": float(np.mean([r[0] for r in rhos])),
                    "rho_orth": float(np.mean([r[1] for r in rhos])),
                    "rho_dom_ci": ci["ci"], "n_seeds": len(seeds), "seeds": seeds,
                }

            # paired test λ2 vs λ0.25
            seeds_test = sorted(set(seeds_for(2.0, SETS_P13 if sname in SETS_P13 else (sname,))) &
                                set(seeds_for(0.25, SETS_P13 if sname in SETS_P13 else (sname,))))
            if sname in SETS_P7A:
                # only if both exist on item4 — typically seed 42 only; require ≥3 else skip test
                seeds_test = [s for s in SEEDS
                              if resolve_root(2.0, s, (sname,)) and resolve_root(0.25, s, (sname,))
                              and resolve_root(0.0, s, SETS_FULL)]
            if str(2.0) in by_lam and by_lam[str(2.0)] and str(0.25) in by_lam and by_lam[str(0.25)]:
                seeds_test = sorted(set(by_lam["2.0"]["seeds"]) & set(by_lam["0.25"]["seeds"]))
            else:
                seeds_test = []

            if len(seeds_test) >= 3 and 2.0 in pull_cache and 0.25 in pull_cache:
                def delta_fn(src, idx, _rep=rep, _set=sname):
                    dom = dom_full[src][_rep][0][_set]
                    orth = dom_full[src][_rep][1][_set]
                    r2, _ = _rho_dom_orth(dom, orth, pull_cache[2.0][src], idx)
                    r025, _ = _rho_dom_orth(dom, orth, pull_cache[0.25][src], idx)
                    return r2 - r025

                dlt = paired_delta_ci(delta_fn, seeds_test, g, n_boot, np.random.default_rng(seed))
                met = bool(dlt["mean"] > 0 and dlt["ci"][0] > 0)
            else:
                dlt = {"mean": None, "ci": [None, None], "excludes_0": None, "n_seeds": len(seeds_test),
                       "seeds": seeds_test}
                met = False

            blk["reps"][rep] = {
                "by_lambda": by_lam,
                "delta_rho_dom_l2_minus_l025": dlt["mean"],
                "delta_ci": dlt["ci"], "excludes_0": dlt.get("excludes_0"),
                "l2_exceeds_l025": met, "n_seeds_test": dlt.get("n_seeds"),
                "seeds_test": dlt.get("seeds"),
            }
            print("dose", sname, rep, "δ=", dlt["mean"], "met" if met else "no", flush=True)
        out["sets"][sname] = blk

    # verdict on sets that have a valid latent test with ≥3 seeds
    tested = [s for s in SETS_FULL
              if out["sets"][s]["reps"]["latent"]["delta_rho_dom_l2_minus_l025"] is not None]
    k = sum(1 for s in tested if out["sets"][s]["reps"]["latent"]["l2_exceeds_l025"])
    kb = sum(1 for s in tested if out["sets"][s]["reps"]["backbone"]["l2_exceeds_l025"])
    out["verdict"] = {
        "sets_tested": tested, "latent_n_met": k, "backbone_n_met": kb,
        "l2_exceeds_l025": bool(len(tested) >= 4 and k >= 4) if len(tested) >= 4 else bool(k == len(tested) and k >= 1),
        "rule": "ρ_dom(λ2)−ρ_dom(λ0.25)>0 CI excludes 0 in ≥4/5 sets when tested; P7a mid-λ omitted (no matched λ0)",
        "n_sets_tested": len(tested),
    }
    # stricter: decisions say ≥4 of 5 — if only pad+fitz tested, report that explicitly
    if set(tested) == set(SETS_P13):
        out["verdict"]["l2_exceeds_l025"] = bool(k >= 2 and len(tested) == 2)  # both pad+fitz; flag incomplete
        out["verdict"]["note"] = "only pad_heldout+Fitzpatrick have matched mid-λ seeds; not a full 5-set verdict"
    return out


def fci(mean, ci):
    if mean is None or ci is None or ci[0] is None:
        return "—"
    return "{:+.3f} [{:+.3f}, {:+.3f}]".format(mean, *ci)


def write_report(placebo, dose):
    L = ["# Paper B R5 D1 follow-up — placebo and dose", "",
         "Decisions: `decisions/paperB_r5_d1_followup.md`. Commit `{}`.".format(R.git_head()), "",
         "backbone = `backbone_raw_lesion` (pre-projection features of the same DANN-trained model; reversed gradient). "
         "pull = change from λ = 0 to λ = k. Domain score from the source λ = 0 model.", "",
         "Feature note: {} | Dose: {}".format(placebo.get("feature_note"), dose.get("feature_note")), "",
         "## (1) Placebo", "",
         "Pairs: {}. DANN-specific if Δρ_dom ≥ {:.2f} and CI entirely above 0 in ≥ 4/5 sets (latent).".format(
             ", ".join("{}→{}".format(s, PARTNER[s]) for s in SEEDS), DELTA_MIN), "",
         "| set | rep | ρ_dom DANN | ρ_dom placebo | Δρ_dom [CI] | DANN-specific |",
         "|---|---|---|---|---|---|"]
    rows = []
    for s in SETS_FULL:
        for rep in REPS:
            r = placebo["sets"][s]["reps"][rep]
            L.append("| {} | {} | {:+.3f} | {:+.3f} | {} | {} |".format(
                D1.SET_LABEL[s], rep, r["rho_dom_dann"], r["rho_dom_placebo"],
                fci(r["delta_rho_dom"], r["delta_rho_dom_ci"]),
                "met" if r["dann_specific_set"] else "not met"))
            rows.append({"item": "placebo", "set": s, "rep": rep, **r})
    v = placebo["verdict"]
    L += ["", "Placebo verdict: **{}/5** latent → DANN-specific {} (backbone {}/5).".format(
        v["latent_n_met"], "supported" if v["dann_specific"] else "not supported", v["backbone_n_met"]), "",
          "## (2) Dose", "",
          "Test: ρ_dom(λ0→λ2) − ρ_dom(λ0→λ0.25); no monotonicity claim. P7a mid-λ rows are — (no matched λ0).", "",
          "| set | rep | " + " | ".join("λ={:g}".format(x) for x in LAMS_DOSE) + " | δ(2−0.25) | 2>0.25 |",
          "|---|---|" + "|".join(["---:"] * len(LAMS_DOSE)) + "|---|---|"]
    for s in SETS_FULL:
        for rep in REPS:
            r = dose["sets"][s]["reps"][rep]
            cells = []
            for lam in LAMS_DOSE:
                b = r["by_lambda"].get(str(lam))
                cells.append("—" if not b or b.get("rho_dom") is None else "{:+.3f}".format(b["rho_dom"]))
            L.append("| {} | {} | {} | {} | {} |".format(
                D1.SET_LABEL[s], rep, " | ".join(cells),
                fci(r["delta_rho_dom_l2_minus_l025"], r["delta_ci"]),
                "met" if r["l2_exceeds_l025"] else ("—" if r["delta_rho_dom_l2_minus_l025"] is None else "not met")))
    vd = dose["verdict"]
    L += ["", "Dose test on sets {}: **{}/{}** latent met → {}. {}".format(
        vd.get("sets_tested"), vd["latent_n_met"], vd.get("n_sets_tested"),
        "supported" if vd["l2_exceeds_l025"] else "not supported / incomplete",
        vd.get("note", "")), ""]
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
    pre = R.prereg()
    n_boot, seed = int(pre["n_resamples"]), int(pre["rng_seed"])
    hmask = pad_heldout_mask()
    print("placebo…", flush=True)
    placebo = analyse_placebo(hmask, n_boot, seed)
    print("dose…", flush=True)
    dose = analyse_dose(hmask, n_boot, seed)
    OUT.mkdir(parents=True, exist_ok=True)
    R.write_json(OUT / "placebo.json", placebo)
    R.write_json(OUT / "dose.json", dose)
    R.write_json(OUT / "D1_FOLLOWUP.json",
                 {"commit": R.git_head(), "n_boot": n_boot, "rng_seed": seed, "placebo": placebo, "dose": dose})
    write_report(placebo, dose)
    amend_parent_report()
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
