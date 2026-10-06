#!/usr/bin/env python3
"""Aggregate Phase 2.5 per-seed JSONs + 2.5c axis file → PHASE2_5_REPORT.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.datasets.constants import INDEX_TO_LABEL

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase2_5")
MASTER = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB")
METHODS = ("runB_orth1", "baseline_soft", "effb3_control")
SEEDS = (42, 52, 62, 72, 82)
DETS = ("mahalanobis_classcond", "MSP", "Energy_T1", "cosine_max", "knn_k50", "mahalanobis_agnostic")
HEADLINE = {
    "runB_orth1": "z_lesion (CSG runB_orth1)",
    "baseline_soft": "ResNet-50 backbone_raw",
    "effb3_control": "EffNet-B3 16-d z",
}


def ms(vals):
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
        "n": len(vals),
        "values": vals,
    }


def pull(rows, method, ood, det, which):
    vals = []
    for r in rows:
        if r["method"] != method:
            continue
        v = r["ood_sets"][ood][det][which]
        if isinstance(v, dict):
            v = v["AUROC"]
        vals.append(v)
    return ms(vals)


def fmt(block, digits=3):
    if not block:
        return "n/a"
    return "{:.{d}f} ± {:.{d}f}".format(block["mean"], block["std"], d=digits)


def nearest_story(x, a=0.50, b=0.41):
    if x is None:
        return "n/a"
    return "≈0.50 (domain-specific)" if abs(x - a) <= abs(x - b) else "≈0.41 (structural)"


def class_rows_agg(per_seed_rows, method):
    """Median-of-medians across seeds for per-class Maha."""
    seeds = [r for r in per_seed_rows if r["method"] == method]
    if not seeds:
        return []
    out = []
    n_cls = len(seeds[0]["per_class_maha"])
    for i in range(n_cls):
        name = seeds[0]["per_class_maha"][i]["class"]
        in_pad = seeds[0]["per_class_maha"][i]["in_pad"]
        id_med = [r["per_class_maha"][i]["id"]["median"] for r in seeds if r["per_class_maha"][i]["id"]["n"]]
        id_iqr = [r["per_class_maha"][i]["id"]["iqr"] for r in seeds if r["per_class_maha"][i]["id"]["n"]]
        ood_med = [r["per_class_maha"][i]["ood"]["median"] for r in seeds if r["per_class_maha"][i]["ood"]["n"]]
        ood_iqr = [r["per_class_maha"][i]["ood"]["iqr"] for r in seeds if r["per_class_maha"][i]["ood"]["n"]]
        n_id = seeds[0]["per_class_maha"][i]["id"]["n"]
        n_ood = seeds[0]["per_class_maha"][i]["ood"]["n"]
        out.append(
            {
                "class": name,
                "in_pad": in_pad,
                "n_id": n_id,
                "n_ood": n_ood,
                "id_median": ms(id_med),
                "id_iqr": ms(id_iqr),
                "ood_median": ms(ood_med),
                "ood_iqr": ms(ood_iqr),
            }
        )
    return out


def decide(agg):
    csg = agg["methods"]["runB_orth1"]
    fitz = csg["fitz_maha"]["mean"] if csg["fitz_maha"] else None
    pad_u = csg["pad_maha_unrestricted"]["mean"] if csg["pad_maha_unrestricted"] else None
    pad6 = csg["pad_maha_6class"]["mean"] if csg["pad_maha_6class"] else None
    padw = csg["pad_maha_reweighted"]["mean"] if csg.get("pad_maha_reweighted") else None
    a = nearest_story(fitz)
    moved = None if (pad_u is None or pad6 is None) else pad6 - pad_u
    if moved is None:
        b = "n/a"
    elif moved >= 0.04:
        b = "class composition contributes (6-class restriction moved AUROC toward 0.50)"
    elif abs(moved) < 0.02:
        b = "class composition is not the driver (restriction moved AUROC by <0.02)"
    else:
        b = "class composition is a partial contributor (Δ={:.3f})".format(moved)

    if fitz is None or pad6 is None:
        overall = "incomplete"
    elif abs(fitz - 0.50) < 0.04 and abs(pad6 - 0.50) < 0.04:
        overall = "PAD-domain mapping plus class composition; neither is the sole driver"
    elif abs(fitz - 0.41) < 0.04 and abs(pad6 - 0.50) < 0.04:
        overall = "class-composition confound (Fitz stays inverted ~0.41 would have refuted this; it did not stay)"
    elif abs(fitz - 0.41) < 0.04 and abs(pad6 - 0.41) < 0.04:
        overall = "structural: inversion is not PAD-image memorisation and is not class composition"
    elif abs(fitz - 0.50) < 0.04 and abs(pad6 - 0.41) < 0.04:
        overall = "learned PAD-domain mapping; class composition does not explain the PAD inversion"
    else:
        overall = "neither cleanly"
    return {
        "2.5a_fitz_maha": fitz,
        "2.5a_reading": a,
        "2.5b_pad_unrestricted": pad_u,
        "2.5b_pad_6class": pad6,
        "2.5b_pad_reweighted": padw,
        "2.5b_delta_6class": moved,
        "2.5b_reading": b,
        "overall": overall,
    }


def write_report(agg, axis, missing):
    csg = agg["methods"]["runB_orth1"]
    d = agg["decision"]
    lines = []
    a = lambda s="": lines.append(s)

    a("# Paper B — Phase 2.5 report")
    a()
    a("Inference only. Detectors, splits and hyperparameters were not adjusted toward a predicted value.")
    a("Phase 2 prediction (held-out PAD restores z_lesion Maha ≈ 0.50) **FAILED** and is left as a finding.")
    a()
    a("## Which explanation the data support")
    a()
    a("**{}**".format(d["overall"]))
    a()
    a("- 2.5a Fitzpatrick17k z_lesion Maha: **{}** — {}.".format(fmt(csg["fitz_maha"]), d["2.5a_reading"]))
    a("- 2.5b PAD unrestricted: **{}**; 6-class restricted: **{}**; reweighted: **{}**.".format(
        fmt(csg["pad_maha_unrestricted"]), fmt(csg["pad_maha_6class"]), fmt(csg.get("pad_maha_reweighted"))
    ))
    a("- 2.5b reading: {}.".format(d["2.5b_reading"]))
    a()
    if missing:
        a("**Incomplete:** missing {} / 15 per-seed files. Numbers below use whatever is on disk.".format(len(missing)))
        a()

    a("## 2.5a — Unseen-domain test (Fitzpatrick17k, n=3887)")
    a()
    a("ID = ISIC test. Statistics fit on ISIC train only. OOD = Fitzpatrick17k (never in any training branch, any seed, any phase). 5 `runB_orth1` seeds.")
    a()
    a("| Detector | z_lesion / 16-d z / backbone AUROC |")
    a("|---|---|")
    a("| Mahalanobis class-cond | {} |".format(fmt(csg["fitz_maha"])))
    a("| MSP | {} |".format(fmt(csg["fitz_msp"])))
    a("| Energy_T1 | {} |".format(fmt(csg["fitz_energy"])))
    a("| cosine_max | {} |".format(fmt(csg["fitz_cosine"])))
    a("| knn_k50 | {} |".format(fmt(csg["fitz_knn"])))
    a()
    a("Pre-registered reading: AUROC ≈ 0.50 → domain-specific mapping that generalises across PAD but not beyond it; AUROC ≈ 0.41 → structural / class-composition. Observed: **{}**.".format(
        fmt(csg["fitz_maha"])
    ))
    a()
    a("Same Fitzpatrick suite on the two controls:")
    a()
    a("| Model | Maha | MSP | Energy | cosine | kNN |")
    a("|---|---|---|---|---|---|")
    for m in METHODS:
        blk = agg["methods"][m]
        a("| {} | {} | {} | {} | {} | {} |".format(
            HEADLINE[m],
            fmt(blk["fitz_maha"]),
            fmt(blk["fitz_msp"]),
            fmt(blk["fitz_energy"]),
            fmt(blk["fitz_cosine"]),
            fmt(blk["fitz_knn"]),
        ))
    a()

    a("## 2.5b — Class-composition control")
    a()
    a("ISIC test has 8 classes; PAD has 6 (no DF, no VASC) and is concentrated in BCC / AK / NEV. Class-conditional Mahalanobis takes distance to the nearest class centroid. Rare ISIC-only classes can drive AUROC below 0.5 with no adversarial explanation.")
    a()
    a("| Model | Unrestricted (Phase 1 headline protocol) | ID restricted to PAD's 6 classes | ID reweighted to PAD class mix |")
    a("|---|---|---|---|")
    for m in METHODS:
        blk = agg["methods"][m]
        a("| {} | {} | {} | {} |".format(
            HEADLINE[m],
            fmt(blk["pad_maha_unrestricted"]),
            fmt(blk["pad_maha_6class"]),
            fmt(blk["pad_maha_reweighted"]),
        ))
    a()
    a("Phase 1 headlines under the 6-class restriction: if a number moves, the published inversion was partly a class-mix artifact. If it does not, the inversion survives the confound.")
    a()

    a("### Per-class Mahalanobis (median, IQR, n) — CSG `z_lesion`, mean across seeds")
    a()
    a("| Class | in PAD | n ID | n PAD | ID median (IQR) | PAD median (IQR) |")
    a("|---|---|---:|---:|---|---|")
    for row in agg["per_class"]["runB_orth1"]:
        idm = fmt(row["id_median"], 1) if row["id_median"] else "—"
        idi = fmt(row["id_iqr"], 1) if row["id_iqr"] else "—"
        om = fmt(row["ood_median"], 1) if row["ood_median"] else "—"
        oi = fmt(row["ood_iqr"], 1) if row["ood_iqr"] else "—"
        a("| {} | {} | {} | {} | {} ({}) | {} ({}) |".format(
            row["class"], "yes" if row["in_pad"] else "**no**", row["n_id"], row["n_ood"], idm, idi, om, oi
        ))
    a()
    a("Drivers: classes whose ID median Maha exceeds the PAD median are the ones that make ID look more OOD than PAD.")
    a()
    a("Same table for baseline and EffB3: `phase2_5/per_class_maha.json`.")
    a()

    a("## 2.5c — Fitzpatrick17k axis: distributions, not means")
    a()
    if axis:
        a("Shared k=1 axis of `z_context` (ISIC-train PCA, PAD-positive).")
        a()
        a("| Dataset | n | mean | std | median | IQR |")
        a("|---|---:|---:|---:|---:|---:|")
        for key, lab in [("isic", "ISIC test"), ("pad", "PAD"), ("fitzpatrick17k", "Fitzpatrick17k")]:
            x = axis[key]
            a("| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
                lab, x["n"], x["mean"], x["std"], x["median"], x["iqr"]
            ))
        a()
        a("| Pair | overlap coefficient | Cohen's d |")
        a("|---|---:|---:|")
        for pair, lab in [("ISIC_Fitz", "ISIC ↔ Fitz"), ("Fitz_PAD", "Fitz ↔ PAD"), ("ISIC_PAD", "ISIC ↔ PAD")]:
            p = axis["pairwise"][pair]
            a("| {} | {:.3f} | {:.2f} |".format(lab, p["overlap_coef"], p["cohens_d"]))
        a()
        a("- Fraction of Fitzpatrick17k outside the ISIC–PAD **mean** interval [{:.2f}, {:.2f}]: **{:.3f}**.".format(
            axis["mean_interval"][0], axis["mean_interval"][1], axis["frac_fitz_outside_mean_interval"]
        ))
        a("- Fraction outside the ISIC–PAD **minmax** interval: **{:.3f}**.".format(
            axis["frac_fitz_outside_minmax_ISIC_PAD"]
        ))
        a()
        a("**Supported claim:** {}.".format(axis["supported_claim"]))
        a()
        a("Density plot: `results/paperB/figures/domain_axis_distributions.pdf`.")
        a()
        a("Means alone (ISIC 0.07, Fitz 36, PAD 45) overstated tightness. Fitz std is {:.2f} against an ISIC–PAD gap of {:.1f}. {}".format(
            axis["fitzpatrick17k"]["std"],
            abs(axis["pad"]["mean"] - axis["isic"]["mean"]),
            axis["claim_check"] + ".",
        ))
    else:
        a("axis_distributions.json not on disk.")
    a()

    a("## Reframing locked from Phase 1.6a")
    a()
    a("A frozen ImageNet ResNet-50 that has never seen this data reaches **linear-head domain AUROC 0.998**. A domain monitor is cheap. The ISIC/PAD distinction is linearly decodable from generic visual features.")
    a()
    a("`z_context` is **not** a better detector. Factorization does not make domain detectable — it already is, from any backbone. Factorization concentrates domain into a single interpretable axis: k=1 gives AUROC 1.00 at 84.5% variance for `z_context`, versus 0.55–0.70 at k=1 for every other representation tested.")
    a()
    a("No mechanism claim is written beyond what 2.5a/b distinguish. Phase 3 (`λ_adv` sweep) is the controlled intervention.")
    a()
    a("## Files")
    a()
    a("- `results/paperB/phase2_5/per_seed/`")
    a("- `results/paperB/phase2_5/phase25_aggregate.json`")
    a("- `results/paperB/phase2_5/axis_distributions.json`")
    a("- `results/paperB/figures/domain_axis_distributions.pdf`")
    a()

    text = "\n".join(lines) + "\n"
    (OUT / "PHASE2_5_REPORT.md").write_text(text)
    (MASTER / "PHASE2_5_REPORT.md").write_text(text)
    return text


def main():
    seed_dir = OUT / "per_seed"
    rows = []
    missing = []
    for method in METHODS:
        for s in SEEDS:
            p = seed_dir / "{}_s{}.json".format(method, s)
            if not p.exists():
                missing.append(str(p))
                continue
            rows.append(json.loads(p.read_text()))
    agg = {"methods": {}, "pad_classes": [INDEX_TO_LABEL[c] for c in (0, 1, 2, 3, 4, 7)], "missing": missing}
    for method in METHODS:
        agg["methods"][method] = {
            "fitz_maha": pull(rows, method, "fitzpatrick17k", "mahalanobis_classcond", "unrestricted"),
            "fitz_msp": pull(rows, method, "fitzpatrick17k", "MSP", "unrestricted"),
            "fitz_energy": pull(rows, method, "fitzpatrick17k", "Energy_T1", "unrestricted"),
            "fitz_cosine": pull(rows, method, "fitzpatrick17k", "cosine_max", "unrestricted"),
            "fitz_knn": pull(rows, method, "fitzpatrick17k", "knn_k50", "unrestricted"),
            "pad_maha_unrestricted": pull(rows, method, "pad", "mahalanobis_classcond", "unrestricted"),
            "pad_maha_6class": pull(rows, method, "pad", "mahalanobis_classcond", "id_6class_restricted"),
            "pad_maha_reweighted": pull(rows, method, "pad", "mahalanobis_classcond", "id_reweighted_to_pad"),
        }
    agg["per_class"] = {m: class_rows_agg(rows, m) for m in METHODS}
    (OUT / "per_class_maha.json").write_text(json.dumps(agg["per_class"], indent=2) + "\n")
    agg["decision"] = decide(agg) if rows else {"overall": "no per-seed files yet"}
    axis_path = OUT / "axis_distributions.json"
    axis = json.loads(axis_path.read_text()) if axis_path.exists() else None
    agg["axis_supported_claim"] = axis["supported_claim"] if axis else None
    (OUT / "phase25_aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    write_report(agg, axis, missing)
    print(json.dumps({"n_rows": len(rows), "missing": len(missing), "decision": agg["decision"]}, indent=2))
    print("wrote", OUT / "PHASE2_5_REPORT.md")


if __name__ == "__main__":
    main()
