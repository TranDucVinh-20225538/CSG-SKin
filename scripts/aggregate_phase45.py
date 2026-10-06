#!/usr/bin/env python3
"""Aggregate Phase 4.5a/b per-run JSONs → PHASE4_5_REPORT.md.

No double-dissociation claim. 4a and 4b with equal prominence.
If 4b paired-diff CI includes 0, say so and do not carry a 4b-win claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase4_5")
MASTER = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB")
CONFIGS = ("4a", "4b")
METHODS = ("baseline", "effb3", "runB_orth1")
SEEDS = (42, 52, 62)
DETS = ("mahalanobis_classcond", "MSP", "Energy_T1", "cosine_max", "knn_k50")
SPACE = {"baseline": "backbone_raw", "effb3": "backbone_raw", "runB_orth1": "z_lesion"}


def pull(rows, config, method, space, det, which="pooled_holdout"):
    vals, boots = [], []
    for r in rows:
        if r["config"] != config or r["method"] != method:
            continue
        if space not in r["spaces"]:
            continue
        try:
            e = r["spaces"][space][det][which]
        except KeyError:
            continue
        vals.append((r["seed"], e["AUROC"], e.get("FPR95"), e["bootstrap_auroc"]))
    if not vals:
        return None
    vals = sorted(vals, key=lambda x: x[0])
    aurocs = [v[1] for v in vals]
    return {
        "seeds": [v[0] for v in vals],
        "values": aurocs,
        "mean": float(np.mean(aurocs)),
        "std": float(np.std(aurocs, ddof=1) if len(aurocs) > 1 else 0.0),
        "per_seed": [
            {"seed": v[0], "AUROC": v[1], "FPR95": v[2], "boot": v[3]} for v in vals
        ],
        "n": len(aurocs),
    }


def paired(a, b):
    """a,b are pull() dicts aligned by seed."""
    if not a or not b:
        return None
    ma = {p["seed"]: p["AUROC"] for p in a["per_seed"]}
    mb = {p["seed"]: p["AUROC"] for p in b["per_seed"]}
    seeds = sorted(set(ma) & set(mb))
    diffs = np.array([ma[s] - mb[s] for s in seeds], dtype=np.float64)
    if len(diffs) == 0:
        return None
    std = float(np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0)
    d = float(diffs.mean() / std) if std > 1e-12 else 0.0
    # bootstrap the 3 seed-level diffs (with replacement); n=3 is noisy — report it as such
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(10000):
        boots.append(float(rng.choice(diffs, size=len(diffs), replace=True).mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "seeds": seeds,
        "diffs": [float(x) for x in diffs],
        "mean_diff": float(diffs.mean()),
        "std_diff": std,
        "cohens_d": d,
        "seed_bootstrap_ci95": [float(lo), float(hi)],
        "ci_includes_zero": bool(lo <= 0 <= hi),
        "n": len(diffs),
        "note": "CI is over n=3 seeds with replacement; wide by construction. Per-seed bootstrap CIs over test samples are in per_seed.boot.",
    }


def fmt(b, digits=3):
    if not b:
        return "n/a"
    return "{:.{d}f} ± {:.{d}f}  [{}]".format(
        b["mean"], b["std"], ", ".join("{:.{d}f}".format(v, d=digits) for v in b["values"]), d=digits
    )


def boot_span(b):
    if not b:
        return ""
    return "; ".join(
        "s{} [{:.3f}, {:.3f}]".format(p["seed"], p["boot"]["lo"], p["boot"]["hi"]) for p in b["per_seed"]
    )


def write_report(agg, missing):
    lines = []
    a = lines.append
    a("# Paper B — Phase 4.5 report")
    a("")
    a("The clean 2×2 double dissociation is **dead**. It is not rescued. 4a and 4b are reported with equal prominence.")
    a("Detectors, splits and hyperparameters were not adjusted toward a predicted value.")
    a("")
    if missing:
        a("**Incomplete:** missing {} per-run files. Numbers use whatever is on disk.".format(len(missing)))
        a("")

    a("## Framing (locked)")
    a("")
    a("- Domain OOD: `z_context` > 0.9999 vs `z_lesion` 0.41 — a large, robust separation between branches.")
    a("- Semantic OOD: both branches land in 0.6–0.8; ordering depends on which class is held out.")
    a("- The split is **graded**, not categorical: some diagnostic classes carry low-level appearance signatures.")
    a("- **4b is not a near-OOD claim** until 4.5c reports. Both configurations are descriptive.")
    a("")

    a("## 4.5a — Variance (blocking)")
    a("")
    a("Mahalanobis class-conditional AUROC. Mean ± std over seeds, per-seed values, bootstrap 95% CI over test samples (1000 resamples of ID and OOD).")
    a("")
    a("### 4a hold {DF, VASC} — n_id=4968, n_ood=492")
    a("")
    a("| Model | space | mean ± std [s42, s52, s62] | per-seed bootstrap CI |")
    a("|---|---|---|---|")
    for method, space in [("baseline", "backbone_raw"), ("effb3", "backbone_raw"), ("runB_orth1", "z_lesion"), ("runB_orth1", "z_context")]:
        b = agg["maha"]["4a"].get("{}/{}".format(method, space))
        a("| {} | {} | {} | {} |".format(method, space, fmt(b), boot_span(b)))
    a("")
    pa = agg["paired"]["4a_lesion_minus_baseline"]
    pc = agg["paired"]["4a_lesion_minus_context"]
    a("Paired seed-wise `z_lesion − baseline`: {}.".format(
        "n/a" if not pa else "Δ={:.3f}, Cohen's d={:.2f}, seed-bootstrap CI [{:.3f}, {:.3f}], includes 0: **{}**".format(
            pa["mean_diff"], pa["cohens_d"], pa["seed_bootstrap_ci95"][0], pa["seed_bootstrap_ci95"][1], pa["ci_includes_zero"]
        )
    ))
    a("Paired `z_lesion − z_context`: {}.".format(
        "n/a" if not pc else "Δ={:.3f}, d={:.2f}, CI [{:.3f}, {:.3f}], includes 0: **{}**".format(
            pc["mean_diff"], pc["cohens_d"], pc["seed_bootstrap_ci95"][0], pc["seed_bootstrap_ci95"][1], pc["ci_includes_zero"]
        )
    ))
    a("")
    a("### 4b hold {SCC} — n_id=4941, n_ood=628")
    a("")
    a("| Model | space | mean ± std [s42, s52, s62] | per-seed bootstrap CI |")
    a("|---|---|---|---|")
    for method, space in [("baseline", "backbone_raw"), ("effb3", "backbone_raw"), ("runB_orth1", "z_lesion"), ("runB_orth1", "z_context")]:
        b = agg["maha"]["4b"].get("{}/{}".format(method, space))
        a("| {} | {} | {} | {} |".format(method, space, fmt(b), boot_span(b)))
    a("")
    pb = agg["paired"]["4b_lesion_minus_baseline"]
    pbc = agg["paired"]["4b_lesion_minus_context"]
    a("Paired seed-wise `z_lesion − baseline`: {}.".format(
        "n/a" if not pb else "Δ={:.3f}, Cohen's d={:.2f}, seed-bootstrap CI [{:.3f}, {:.3f}], includes 0: **{}**".format(
            pb["mean_diff"], pb["cohens_d"], pb["seed_bootstrap_ci95"][0], pb["seed_bootstrap_ci95"][1], pb["ci_includes_zero"]
        )
    ))
    if pb and pb["ci_includes_zero"]:
        a("")
        a("**4b gap CI includes zero. Stop: 4b is not used to support a win of `z_lesion` over baseline.**")
    a("Paired `z_lesion − z_context`: {}.".format(
        "n/a" if not pbc else "Δ={:.3f}, d={:.2f}, CI [{:.3f}, {:.3f}], includes 0: **{}**".format(
            pbc["mean_diff"], pbc["cohens_d"], pbc["seed_bootstrap_ci95"][0], pbc["seed_bootstrap_ci95"][1], pbc["ci_includes_zero"]
        )
    ))
    a("")

    a("## 4.5b — 4a decomposed by class")
    a("")
    a("Hypothesis: `z_context` 0.64 on 4a is VASC (colour) not DF. Prediction: high on VASC, ≈ 0.50 on DF.")
    a("")
    a("| Model / space | DF-only Maha | VASC-only Maha | pooled |")
    a("|---|---|---|---|")
    for method, space in [("runB_orth1", "z_context"), ("runB_orth1", "z_lesion"), ("baseline", "backbone_raw"), ("effb3", "backbone_raw")]:
        dfb = agg["decomp"].get("4a/{}/{}/DF".format(method, space))
        vb = agg["decomp"].get("4a/{}/{}/VASC".format(method, space))
        pb_ = agg["maha"]["4a"].get("{}/{}".format(method, space))
        a("| {} {} | {} | {} | {} |".format(method, space, fmt(dfb), fmt(vb), fmt(pb_)))
    a("")
    a("Other detectors (MSP, Energy, cosine, kNN): `phase4_5/phase45_aggregate.json`.")
    a("")
    pred = agg.get("vasc_hypothesis")
    a("**VASC/DF hypothesis:** {}.".format(pred if pred else "pending"))
    a("")

    a("## 4.5c — Near/far (queued behind Phase 3)")
    a("")
    if agg.get("hold3"):
        a("Hold-{DF,VASC,SCC} summaries are on disk. See `phase4_5/hold3/`.")
        a("If AUROC advantage vs baseline does not correlate with the near/far index, 4a and 4b simply differ — that is the result, not a near-OOD story.")
    else:
        a("Not started until Phase 3 (`60512`) releases GPUs. 9 runs. 4b must not be used as a near-OOD claim in the meantime.")
    a("")
    a("## Files")
    a("")
    a("- `results/paperB/phase4_5/per_run/`")
    a("- `results/paperB/phase4_5/phase45_aggregate.json`")
    a("")
    text = "\n".join(lines) + "\n"
    (OUT / "PHASE4_5_REPORT.md").write_text(text)
    (MASTER / "PHASE4_5_REPORT.md").write_text(text)
    return text


def vasc_reading(agg):
    ctx_df = agg["decomp"].get("4a/runB_orth1/z_context/DF")
    ctx_v = agg["decomp"].get("4a/runB_orth1/z_context/VASC")
    if not ctx_df or not ctx_v:
        return None
    dfm, vm = ctx_df["mean"], ctx_v["mean"]
    if vm >= 0.70 and abs(dfm - 0.50) <= 0.08:
        return "confirmed: z_context high on VASC ({:.2f}) and ≈0.50 on DF ({:.2f}) — some diagnostic classes carry low-level appearance signatures".format(vm, dfm)
    if vm > dfm + 0.08:
        return "partial: VASC ({:.2f}) > DF ({:.2f}), but DF is not ≈0.50".format(vm, dfm)
    return "not confirmed: DF {:.2f}, VASC {:.2f}".format(dfm, vm)


def main():
    rows, missing = [], []
    for c in CONFIGS:
        for m in METHODS:
            for s in SEEDS:
                p = OUT / "per_run" / "{}_{}_s{}.json".format(c, m, s)
                if not p.exists():
                    missing.append(str(p))
                    continue
                rows.append(json.loads(p.read_text()))
    agg = {"maha": {"4a": {}, "4b": {}}, "paired": {}, "decomp": {}, "missing": missing}
    for c in CONFIGS:
        for method, space in [("baseline", "backbone_raw"), ("effb3", "backbone_raw"), ("runB_orth1", "z_lesion"), ("runB_orth1", "z_context")]:
            agg["maha"][c]["{}/{}".format(method, space)] = pull(rows, c, method, space, "mahalanobis_classcond")
        if c != "4a":
            continue
        for method, space in [("runB_orth1", "z_lesion"), ("runB_orth1", "z_context"), ("baseline", "backbone_raw"), ("effb3", "backbone_raw")]:
            for cls in ("DF", "VASC"):
                agg["decomp"]["{}/{}/{}/{}".format(c, method, space, cls)] = pull(
                    rows, c, method, space, "mahalanobis_classcond", which=cls
                )
    agg["paired"]["4a_lesion_minus_baseline"] = paired(
        agg["maha"]["4a"].get("runB_orth1/z_lesion"), agg["maha"]["4a"].get("baseline/backbone_raw")
    )
    agg["paired"]["4a_lesion_minus_context"] = paired(
        agg["maha"]["4a"].get("runB_orth1/z_lesion"), agg["maha"]["4a"].get("runB_orth1/z_context")
    )
    agg["paired"]["4b_lesion_minus_baseline"] = paired(
        agg["maha"]["4b"].get("runB_orth1/z_lesion"), agg["maha"]["4b"].get("baseline/backbone_raw")
    )
    agg["paired"]["4b_lesion_minus_context"] = paired(
        agg["maha"]["4b"].get("runB_orth1/z_lesion"), agg["maha"]["4b"].get("runB_orth1/z_context")
    )
    agg["vasc_hypothesis"] = vasc_reading(agg)
    hold3 = list((OUT / "hold3").glob("*/summary.json")) if (OUT / "hold3").exists() else []
    agg["hold3"] = [str(p) for p in hold3]
    (OUT / "phase45_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    write_report(agg, missing)
    print(json.dumps({"n_rows": len(rows), "missing": len(missing), "vasc": agg["vasc_hypothesis"], "paired_4b_includes0": (agg["paired"]["4b_lesion_minus_baseline"] or {}).get("ci_includes_zero")}, indent=2))


if __name__ == "__main__":
    main()
