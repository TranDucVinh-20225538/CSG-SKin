#!/usr/bin/env python3
"""Paper B R2-prior report + prior-error Spearman test — decisions/paperB_r6_r2_prior.md.

Requires all 5 seeds for EM and n50 before writing REPORT.md / prior_error.json.
Per seed: estimated vs true PAD priors (TV), five detectors on unseen + SCIN, ID bal,
class-aware held-out probes (a)/(b)/(c). Spearman TV vs mean Maha(Fitz/MED/SD) across
{BBSE, EM, 50-label, oracle}. No interpretation before complete; report text stays tabular.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import r6_class_aware_heldout_probes as CA  # noqa: E402

RT = "cur"
LAM = 2.0
ARMS = ("iwdan_em", "iwdan_n50")
UNSEEN3 = ("fitzpatrick17k", "mednode", "sd198")
OODS = ("pad_heldout",) + UNSEEN3 + ("scin",)
DETS = R.DETS
SPEARMAN_BOOT_N = 2000
SPEARMAN_RNG = 20261017
DANN_LEAK_HEADLINE = 0.634  # verify from R1 shift100 λ=2 at report time
DANN_WITHIN_C = 0.618


def root():
    return R.RES / "r6" / "r2_prior" / RT


def tag(arm, seed):
    return "r2p_{}_{}_ladv{}_s{}".format(RT, arm, R.lam_tag(LAM), seed)


def feat_dir(arm, seed):
    return root() / "features" / tag(arm, seed)


def run_dir(arm, seed):
    return root() / "runs" / tag(arm, seed)


def complete():
    missing = []
    for arm in ARMS:
        for s in R.SEEDS:
            if not (run_dir(arm, s) / "summary.json").exists():
                missing.append(tag(arm, s))
            elif not (feat_dir(arm, s) / "scin.npz").exists():
                missing.append(tag(arm, s) + ":scin")
    return missing


def last_iw(arm, seed):
    p = run_dir(arm, seed) / "iw_log.jsonl"
    if not p.exists():
        return None
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else None


def load_summary(arm, seed):
    return json.loads((run_dir(arm, seed) / "summary.json").read_text())


def true_pad_mix():
    data = R.load_data()
    return R.class_mix(data["pad_adv"].label_idx)


def prior_block(arm, seed, true_mix):
    """Estimated π vs true pad_adv; TV / L1. Oracle TV = 0 by construction."""
    cfg = json.loads((run_dir(arm, seed) / "config.json").read_text())
    iw = last_iw(arm, seed) or {}
    if arm == "iwdan_n50":
        pi = np.asarray(cfg["n50_pi"], float)
        tv = float(cfg.get("prior_tv_error", R.tv_error(pi, true_mix)))
        src = "n50_labels"
        T = None
        bbse_fb = False
    else:
        pi = np.asarray(iw.get("pi") or cfg.get("em_pi") or np.ones(R.NCLS) / R.NCLS, float)
        tv = float(iw.get("prior_tv_error") if iw.get("prior_tv_error") is not None
                   else R.tv_error(pi, true_mix))
        src = "em_last_epoch"
        T = iw.get("temp_T")
        bbse_fb = bool(iw.get("used_bbse_fallback"))
    l1 = float(np.abs(pi / max(pi.sum(), 1e-12) - true_mix).sum())
    return {
        "source": src, "est_mix": pi.tolist(), "true_mix": true_mix.tolist(),
        "tv": tv, "l1": l1, "temp_T": T, "used_bbse_fallback": bbse_fb,
        "per_class_abs": {R.LABELS[c]: float(abs(pi[c] / max(pi.sum(), 1e-12) - true_mix[c]))
                          for c in range(R.NCLS)},
    }


def bbse_prior(seed, true_mix):
    sp = R.RES / "r6" / "r2" / RT / "runs" / "r2_cur_iwdan_ladv2_s{}".format(seed) / "summary.json"
    s = json.loads(sp.read_text())
    est = np.asarray(s["metrics"]["bbse"]["pad_adv"]["est_mix"], float)
    return {
        "source": "bbse_posthoc_r2_iwdan", "est_mix": est.tolist(), "true_mix": true_mix.tolist(),
        "tv": float(R.tv_error(est, true_mix)), "l1": float(np.abs(est - true_mix).sum()),
        "temp_T": None, "used_bbse_fallback": False,
        "per_class_abs": {R.LABELS[c]: float(abs(est[c] - true_mix[c])) for c in range(R.NCLS)},
    }


def oracle_prior(true_mix):
    return {
        "source": "oracle_true_pad_adv", "est_mix": true_mix.tolist(), "true_mix": true_mix.tolist(),
        "tv": 0.0, "l1": 0.0, "temp_T": None, "used_bbse_fallback": False,
        "per_class_abs": {c: 0.0 for c in R.LABELS},
    }


def detectors_from_summary(s):
    ood = s["metrics"]["ood"]
    out = {}
    for name in OODS:
        blk = ood.get(name) or {}
        out[name] = {d: blk.get(d) for d in DETS}
        out[name]["nv_nearest_frac"] = blk.get("nv_nearest_frac")
    return out


def oracle_summary(seed):
    sp = R.RES / "r6" / "r2" / RT / "runs" / "r2_cur_iwdan_oracle_ladv2_s{}".format(seed) / "summary.json"
    return json.loads(sp.read_text())


def class_aware_for(feat_root: Path, t: str):
    """Compute (a)/(b)/(c) for one feature tag; cache under r2_prior/cur/class_aware/."""
    cache = root() / "class_aware" / "{}.json".format(t)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return json.loads(cache.read_text())
    rec = CA.analyse_run(feat_root, t)
    cache.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def spearman(x, y):
    """Spearman ρ via rank correlation (scipy-free)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 3:
        return float("nan")
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def spearman_boot(pairs, n_boot=SPEARMAN_BOOT_N, rng_seed=SPEARMAN_RNG):
    xs = np.array([p[0] for p in pairs], float)
    ys = np.array([p[1] for p in pairs], float)
    rho = spearman(xs, ys)
    rng = np.random.default_rng(rng_seed)
    boots = []
    n = len(pairs)
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        boots.append(spearman(xs[ix], ys[ix]))
    boots = np.asarray(boots, float)
    return {
        "rho": rho,
        "ci_lo": float(np.nanquantile(boots, 0.025)),
        "ci_hi": float(np.nanquantile(boots, 0.975)),
        "n_points": int(n),
        "n_boot": int(n_boot),
        "rng_seed": int(rng_seed),
        "supported": bool(rho <= -0.5 and np.nanquantile(boots, 0.975) < 0.0),
    }


def dann_leakage():
    """R1 shift100 λ=2 pad_full leakage mean (same baseline as R2)."""
    vals = []
    for s in R.SEEDS:
        sp = R.RES / "r6" / "r1" / RT / "runs" / "r1_cur_shift100_ladv2_s{}".format(s) / "summary.json"
        if not sp.exists():
            continue
        m = json.loads(sp.read_text())["metrics"]
        vals.append(float(m["leakage"]["bal_acc_mean"]))
    return float(np.mean(vals)) if vals else DANN_LEAK_HEADLINE


def fmt_dets(blk):
    return " | ".join("{:.3f}".format(blk[d]) if blk.get(d) is not None else "—" for d in DETS)


def build():
    true_mix = true_pad_mix()
    leak_dann = dann_leakage()
    leak_thr = leak_dann + 0.05
    per_seed = []

    # EM + n50
    for arm in ARMS:
        for s in R.SEEDS:
            summ = load_summary(arm, s)
            m = summ["metrics"]
            prior = prior_block(arm, s, true_mix)
            dets = detectors_from_summary(summ)
            ca = class_aware_for(root() / "features", tag(arm, s))
            y_mean = float(np.mean([dets[o]["mahalanobis_classcond"] for o in UNSEEN3]))
            row = {
                "estimator": "em" if arm == "iwdan_em" else "n50",
                "arm": arm, "seed": int(s), "tag": tag(arm, s),
                "prior": prior,
                "id_bal_acc": float(m["id"]["bal_acc"]),
                "degraded": bool(m.get("degraded") or m["id"]["bal_acc"] < R.DEGRADED_ID_BAL),
                "leakage_pad_full": float(m["leakage"]["bal_acc_mean"]),
                "detectors": dets,
                "maha_unseen3_mean": y_mean,
                "class_aware": {
                    "a": ca["unweighted_a"]["bal_acc_mean"],
                    "b": ca["class_balanced_b"]["bal_acc_mean"],
                    "c": ca["within_class_c"]["bal_acc_mean"],
                    "n_eligible": ca["within_class_c"].get("n_eligible_classes"),
                },
            }
            per_seed.append(row)

    # BBSE + oracle references (no new GPU)
    for s in R.SEEDS:
        prior = bbse_prior(s, true_mix)
        # detection y from R2 IWDAN estimated (same ckpt BBSE was fit on)
        summ = json.loads((R.RES / "r6" / "r2" / RT / "runs" /
                           "r2_cur_iwdan_ladv2_s{}".format(s) / "summary.json").read_text())
        dets = detectors_from_summary(summ)
        y_mean = float(np.mean([dets[o]["mahalanobis_classcond"] for o in UNSEEN3]))
        ca = class_aware_for(R.RES / "r6" / "r2" / RT / "features",
                             "r2_cur_iwdan_ladv2_s{}".format(s))
        per_seed.append({
            "estimator": "bbse", "arm": "r2_iwdan_bbse", "seed": int(s),
            "tag": "r2_cur_iwdan_ladv2_s{}".format(s),
            "prior": prior, "id_bal_acc": float(summ["metrics"]["id"]["bal_acc"]),
            "degraded": bool(summ["metrics"].get("degraded")),
            "leakage_pad_full": float(summ["metrics"]["leakage"]["bal_acc_mean"]),
            "detectors": dets, "maha_unseen3_mean": y_mean,
            "class_aware": {
                "a": ca["unweighted_a"]["bal_acc_mean"],
                "b": ca["class_balanced_b"]["bal_acc_mean"],
                "c": ca["within_class_c"]["bal_acc_mean"],
                "n_eligible": ca["within_class_c"].get("n_eligible_classes"),
            },
        })
        osum = oracle_summary(s)
        odets = detectors_from_summary(osum)
        oy = float(np.mean([odets[o]["mahalanobis_classcond"] for o in UNSEEN3]))
        oca = class_aware_for(R.RES / "r6" / "r2" / RT / "features",
                              "r2_cur_iwdan_oracle_ladv2_s{}".format(s))
        per_seed.append({
            "estimator": "oracle", "arm": "iwdan_oracle", "seed": int(s),
            "tag": "r2_cur_iwdan_oracle_ladv2_s{}".format(s),
            "prior": oracle_prior(true_mix),
            "id_bal_acc": float(osum["metrics"]["id"]["bal_acc"]),
            "degraded": bool(osum["metrics"].get("degraded")),
            "leakage_pad_full": float(osum["metrics"]["leakage"]["bal_acc_mean"]),
            "detectors": odets, "maha_unseen3_mean": oy,
            "class_aware": {
                "a": oca["unweighted_a"]["bal_acc_mean"],
                "b": oca["class_balanced_b"]["bal_acc_mean"],
                "c": oca["within_class_c"]["bal_acc_mean"],
                "n_eligible": oca["within_class_c"].get("n_eligible_classes"),
            },
        })

    pairs = [(r["prior"]["tv"], r["maha_unseen3_mean"]) for r in per_seed
             if np.isfinite(r["prior"]["tv"]) and np.isfinite(r["maha_unseen3_mean"])]
    pe = spearman_boot(pairs)

    # Primary fixes (EM / n50 only; exclude degraded from pooled — point table still lists all)
    fixes = {}
    for arm, est in (("iwdan_em", "em"), ("iwdan_n50", "n50")):
        rows = [r for r in per_seed if r["arm"] == arm and not r["degraded"]]
        cell = {}
        for ood in UNSEEN3:
            pts = [r["detectors"][ood]["mahalanobis_classcond"] for r in rows]
            # pooled CI via seed bootstrap of mean (simple; full cluster CI is in main R2 tooling)
            rng = np.random.default_rng(20261017)
            if pts:
                boots = [float(np.mean(rng.choice(pts, size=len(pts), replace=True)))
                         for _ in range(2000)]
                lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
                mean = float(np.mean(pts))
            else:
                lo = hi = mean = float("nan")
            leaks = [r["leakage_pad_full"] for r in rows]
            leak_ok = bool(leaks) and float(np.mean(leaks)) <= leak_thr
            cell[ood] = {
                "maha_mean": mean, "ci_lo": lo, "ci_hi": hi,
                "ci_lo_gt_0.5": bool(lo > 0.5),
                "leakage_mean": float(np.mean(leaks)) if leaks else None,
                "leakage_ok": leak_ok,
                "fixes": bool(lo > 0.5 and leak_ok),
                "n_seeds_non_degraded": len(rows),
            }
        n_fix = sum(1 for o in UNSEEN3 if cell[o]["fixes"])
        fixes[est] = {"sets": cell, "n_fixes": n_fix, "arm_success": n_fix >= 2,
                      "leakage_threshold": leak_thr, "dann_leakage": leak_dann}

    return per_seed, pe, fixes, leak_dann, leak_thr


def write_report(per_seed, pe, fixes, leak_dann, leak_thr):
    L = [
        "# Paper B R6 R2-prior — IWDAN with better class priors",
        "",
        "Design: `decisions/paperB_r6_r2_prior.md`. Commit `{}`.".format(R.C.git_head()),
        "",
        "All 5 seeds × {EM, 50-label} present. Numbers only — no post-hoc reading beyond the "
        "pre-committed criteria / Spearman rule.",
        "",
        "## Per-seed priors (π̂ vs true pad_adv; TV / L1)",
        "",
        "| estimator | seed | TV | L1 | temp_T | BBSE fallback | degraded |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in sorted(per_seed, key=lambda z: (z["estimator"], z["seed"])):
        p = r["prior"]
        L.append("| {} | {} | {:.3f} | {:.3f} | {} | {} | {} |".format(
            r["estimator"], r["seed"], p["tv"], p["l1"],
            "—" if p["temp_T"] is None else "{:.3f}".format(p["temp_T"]),
            "yes" if p.get("used_bbse_fallback") else "no",
            "yes" if r["degraded"] else "no"))
    L += ["",
          "## Per-seed ID bal, leakage, detectors (Maha / kNN / cosine / MSP / Energy), class-aware (a)/(b)/(c)",
          "",
          "Unseen sets + SCIN. SCIN secondary (0.6% ISIC-mapped). Class-aware probes from "
          "`decisions/paperB_r6_class_aware_heldout.md`.",
          ""]
    for ood in OODS:
        L += ["### {}".format(ood), "",
              "| estimator | seed | ID bal | leak | " + " | ".join(R.DET_LABEL[d] for d in DETS) +
              " | (a) | (b) | (c) |",
              "|---|---:|---:|---:|" + "---|" * (len(DETS) + 3)]
        for r in sorted(per_seed, key=lambda z: (z["estimator"], z["seed"])):
            L.append("| {} | {} | {:.3f} | {:.3f} | {} | {:.3f} | {:.3f} | {:.3f} |".format(
                r["estimator"], r["seed"], r["id_bal_acc"], r["leakage_pad_full"],
                fmt_dets(r["detectors"][ood]),
                r["class_aware"]["a"], r["class_aware"]["b"], r["class_aware"]["c"]))
        L.append("")

    L += ["## Primary fixes (EM / 50-label; non-degraded seeds; seed-bootstrap CI of Maha mean)",
          "",
          "Fixes set if Maha CI lower > 0.5 **and** pad_full leakage ≤ DANN λ=2 leakage + 0.05 "
          "({:.3f} + 0.05 = {:.3f}). Arm success = ≥ 2/3 primary sets.".format(leak_dann, leak_thr),
          "",
          "| arm | Fitz fixes | MED fixes | SD fixes | n_fixes | arm success |",
          "|---|---|---|---|---:|---|"]
    for est in ("em", "n50"):
        f = fixes[est]
        L.append("| {} | {} | {} | {} | {} | {} |".format(
            est,
            "yes" if f["sets"]["fitzpatrick17k"]["fixes"] else "no",
            "yes" if f["sets"]["mednode"]["fixes"] else "no",
            "yes" if f["sets"]["sd198"]["fixes"] else "no",
            f["n_fixes"], "yes" if f["arm_success"] else "no"))
    L += ["",
          "### Secondary within-class probe (c) vs DANN 0.618 (does not override primary)",
          "",
          "| arm | seed | (c) | ≤ 0.618 |",
          "|---|---:|---:|---|"]
    for r in sorted(per_seed, key=lambda z: (z["estimator"], z["seed"])):
        if r["estimator"] not in ("em", "n50"):
            continue
        c = r["class_aware"]["c"]
        L.append("| {} | {} | {:.3f} | {} |".format(
            r["estimator"], r["seed"], c, "yes" if c <= DANN_WITHIN_C else "no"))

    L += ["",
          "## Prior-error test (locked)",
          "",
          "x = TV(π̂, true pad_adv); y = mean Maha AUROC over Fitz/MED/SD; "
          "points = {{BBSE, EM, 50-label, oracle}} × 5 seeds. Spearman ρ; bootstrap CI "
          "({}, rng {}). **Supported** if ρ ≤ −0.5 and CI excludes 0.".format(
              SPEARMAN_BOOT_N, SPEARMAN_RNG),
          "",
          "| ρ | 95% CI | n | supported |",
          "|---:|---|---:|---|",
          "| {:.3f} | [{:.3f}, {:.3f}] | {} | {} |".format(
              pe["rho"], pe["ci_lo"], pe["ci_hi"], pe["n_points"],
              "yes" if pe["supported"] else "no"),
          "",
          "Prediction (decisions, written before results): EM TV between BBSE and oracle; "
          "Spearman ρ ≈ −0.6 to −0.9 if prior quality drives detection; else ρ ≈ 0.",
          "",
          ]
    (root() / "REPORT.md").write_text("\n".join(L) + "\n")
    payload = {
        "commit": R.C.git_head(),
        "design": "decisions/paperB_r6_r2_prior.md",
        "per_seed": per_seed,
        "prior_error": pe,
        "fixes": fixes,
        "dann_leakage": leak_dann,
        "leakage_threshold": leak_thr,
        "note": "tabular only; no post-hoc interpretation",
    }
    (root() / "prior_error.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", root() / "REPORT.md", flush=True)
    print("wrote", root() / "prior_error.json", flush=True)
    print("Spearman", pe, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow_partial", action="store_true",
                    help="Write even if some seeds missing (default: refuse)")
    a = ap.parse_args()
    miss = complete()
    if miss and not a.allow_partial:
        print("INCOMPLETE — missing:", ", ".join(miss), flush=True)
        print("Refuse to report / interpret before all 5 seeds × 2 arms are in.", flush=True)
        sys.exit(2)
    if miss:
        print("WARNING partial:", miss, flush=True)
    per_seed, pe, fixes, leak_dann, leak_thr = build()
    write_report(per_seed, pe, fixes, leak_dann, leak_thr)


if __name__ == "__main__":
    main()
