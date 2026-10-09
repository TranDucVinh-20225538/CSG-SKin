#!/usr/bin/env python3
"""R3 A3: write the values behind markdown-only manuscript numbers to JSON beside their reports.

- phase13/participation_ratio.json  (aggregated from phase13/per_run/*.json, the inputs of PHASE13_REPORT.md 13.1)
- phase1_6/concentration.json       (phase1_6/supervised_head.json + ood_dual_branch_per_seed.csv)
- r2/item4/image_level_overlap.json (r2/isic_image_split_with_lesion.csv)

No recomputation from features. Each value is checked against the number printed in its markdown report;
any difference is written to results/paperB/r3/a3/REPORT.md as a finding."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C  # noqa: E402

RES = C.PAPERB / "results" / "paperB"
OUT = RES / "r3" / "a3"


def ms(xs):
    a = np.asarray(xs, float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size),
            "values": [float(x) for x in a]}


def phase13():
    per_run = [json.loads(p.read_text()) for p in sorted((RES / "phase13" / "per_run").glob("runB_orth1_ladv*_s*.json"))]
    cols = {
        "pr_z_lesion_norm": lambda r: r["spectrum"]["z_lesion_norm"]["pr"],
        "pr_z_lesion_preBN": lambda r: r["spectrum"]["z_lesion"]["pr"],
        "pr_z_context": lambda r: r["spectrum"]["z_context"]["pr"],
        "pr_backbone_raw_lesion": lambda r: r["spectrum"]["backbone_raw_lesion"]["pr"],
        "maha_eps1e-3_pad_full": lambda r: r["maha_robustness_z_lesion_norm_pad_full"]["headline_auroc"],
        "maha_ledoit_wolf_pad_full": lambda r: r["maha_robustness_z_lesion_norm_pad_full"]["ledoit_wolf"]["auroc"],
        "condition_number_eps1e-3": lambda r: r["maha_robustness_z_lesion_norm_pad_full"]["headline_condition_number"],
    }
    by = defaultdict(list)
    for r in per_run:
        by[float(r["lambda_adv"])].append(r)
    table = {}
    for lam in sorted(by):
        rs = by[lam]
        table["{:g}".format(lam)] = {"seeds": [int(r["seed"]) for r in rs],
                                     **{k: ms([f(r) for r in rs]) for k, f in cols.items()}}
    js = {"source": "results/paperB/phase13/per_run/runB_orth1_ladv*_s*.json (aggregated as in scripts/analyze_phase13.py)",
          "report": "results/paperB/phase13/PHASE13_REPORT.md section 13.1",
          "definition": "participation ratio (sum eig)^2 / sum eig^2 of the ISIC-train covariance",
          "by_lambda": table}
    # Compare to the printed table.
    txt = (RES / "phase13" / "PHASE13_REPORT.md").read_text()
    sec = txt[txt.index("## 13.1"):txt.index("## 13.2")]
    diffs = []
    for line in sec.splitlines():
        m = re.match(r"\|\s*([0-9.]+)\s*\|\s*(\d+)\s*\|(.*)\|\s*$", line)
        if not m:
            continue
        lam = "{:g}".format(float(m.group(1)))
        cells = [c.strip() for c in m.group(3).split("|")]
        for k, c in zip(cols, cells):
            mean, sd = (float(x) for x in c.split("±"))
            v = table[lam][k]
            if round(v["mean"], 3) != round(mean, 3) or round(v["sd"], 3) != round(sd, 3):
                diffs.append("phase13 λ={} {}: report {} vs JSON {:.3f} ± {:.3f}".format(lam, k, c, v["mean"], v["sd"]))
    (RES / "phase13" / "participation_ratio.json").write_text(json.dumps(js, indent=1) + "\n")
    return js, diffs


def phase1_6():
    sh = json.loads((RES / "phase1_6" / "supervised_head.json").read_text())
    k1 = {r["representation"]: {"pca_k1_auroc": r["pca_k1"]["AUROC"], "pca_k1_var_explained": r["pca_k1"]["var_explained"],
                                 "linear_domain_head_auroc_mean": r["linear_domain_head"]["AUROC_mean"]} for r in sh["results"]}
    df = pd.read_csv(RES / "ood_dual_branch_per_seed.csv")
    zc = df[(df.method == "runB_orth1") & (df.feature_space == "z_context") & (df.detector == "mahalanobis_classcond")
            & (df.metric == "AUROC")].sort_values("seed")
    pairs = []
    for _, r in zc.iterrows():
        n = int(r.n_id) * int(r.n_ood)
        pairs.append({"seed": int(r.seed), "auroc": float(r.value), "n_pairs": n,
                      "discordant_pairs": float((1.0 - r.value) * n)})
    others = [v["pca_k1_auroc"] for k, v in k1.items() if k != "z_context"]
    js = {"sources": ["results/paperB/phase1_6/supervised_head.json", "results/paperB/ood_dual_branch_per_seed.csv"],
          "report": "results/paperB/phase1_6/PHASE1_6_REPORT.md",
          "z_context_pc1_var_explained": k1["z_context"]["pca_k1_var_explained"],
          "z_context_pc1_auroc": k1["z_context"]["pca_k1_auroc"],
          "pca_k1_by_representation": k1,
          "pca_k1_auroc_other_representations_range": [min(others), max(others)],
          "z_context_maha_classcond_pad_full_by_seed": pairs,
          "z_context_maha_classcond_pad_full_mean": ms([p["auroc"] for p in pairs]),
          "note": "pairs = n_id * n_ood = 5067 * 2298 (pad_full); discordant = (1 - AUROC) * pairs (ties count half)"}
    diffs = []
    if round(100 * js["z_context_pc1_var_explained"], 1) != 84.5:
        diffs.append("phase1_6 PC1 variance {:.2f}% vs report 84.5%".format(100 * js["z_context_pc1_var_explained"]))
    s42 = pairs[0]
    if s42["seed"] != 42 or round(s42["discordant_pairs"]) != 2 or round(s42["n_pairs"] / 1e6, 1) != 11.6:
        diffs.append("phase1_6 seed-42 discordant pairs {:.2f} of {} vs report 2 / 11.6M".format(s42["discordant_pairs"], s42["n_pairs"]))
    lo, hi = js["pca_k1_auroc_other_representations_range"]
    if (round(lo, 2), round(hi, 2)) != (0.55, 0.70):
        diffs.append("phase1_6 k=1 range for other representations {:.3f}–{:.3f} vs report 0.55–0.70".format(lo, hi))
    (RES / "phase1_6" / "concentration.json").write_text(json.dumps(js, indent=1) + "\n")
    return js, diffs


def overlap():
    disjoint, _g, te = C.isic_test_groups()
    n = int(len(disjoint))
    n_shared = int((~disjoint).sum())
    js = {"source": "results/paperB/r2/isic_image_split_with_lesion.csv (image-level split, ISIC lesion_id)",
          "n_test": n, "n_test_sharing_lesion_with_train_or_val": n_shared, "frac_sharing": n_shared / n,
          "pct_sharing": 100.0 * n_shared / n, "n_lesion_disjoint": int(disjoint.sum()),
          "n_lesion_disjoint_null_lesion_id": int(te[disjoint].lesion_id.isna().sum()),
          "null_lesion_policy": "null lesion_id never matches; such images count as disjoint"}
    diffs = []
    if (n_shared, n, round(js["pct_sharing"], 1), js["n_lesion_disjoint"]) != (3041, 5067, 60.0, 2026):
        diffs.append("overlap {}/{} ({:.1f}%), disjoint {} vs manuscript 3041/5067 (60.0%), 2026".format(
            n_shared, n, js["pct_sharing"], js["n_lesion_disjoint"]))
    (C.R2 / "item4" / "image_level_overlap.json").write_text(json.dumps(js, indent=1) + "\n")
    return js, diffs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p13, d13 = phase13()
    p16, d16 = phase1_6()
    ov, dov = overlap()
    diffs = d13 + d16 + dov
    head = C.git_head()
    verdict = ("All values written to JSON match their published reports; no recomputation needed."
               if not diffs else "{} value(s) differ from the published report; see Findings.".format(len(diffs)))
    t = p13["by_lambda"]
    L = ["# R3 A3 — JSON for markdown-only manuscript numbers", "",
         "Commit: `{}`.".format(head), "", "**Verdict:** {}".format(verdict), "",
         "| Manuscript number | JSON file | key | value |", "|---|---|---|---|",
         "| 4.879 | phase13/participation_ratio.json | by_lambda.0.pr_z_lesion_norm.mean | {:.4f} |".format(t["0"]["pr_z_lesion_norm"]["mean"]),
         "| 4.500 | phase13/participation_ratio.json | by_lambda.0.25.pr_z_lesion_norm.mean | {:.4f} |".format(t["0.25"]["pr_z_lesion_norm"]["mean"]),
         "| 84.5% | phase1_6/concentration.json | z_context_pc1_var_explained | {:.5f} |".format(p16["z_context_pc1_var_explained"]),
         "| 2 discordant pairs | phase1_6/concentration.json | z_context_maha_classcond_pad_full_by_seed[seed 42].discordant_pairs | {:.3f} |".format(
             p16["z_context_maha_classcond_pad_full_by_seed"][0]["discordant_pairs"]),
         "| 11.6M | phase1_6/concentration.json | z_context_maha_classcond_pad_full_by_seed[seed 42].n_pairs | {} |".format(
             p16["z_context_maha_classcond_pad_full_by_seed"][0]["n_pairs"]),
         "| 0.55–0.70 | phase1_6/concentration.json | pca_k1_auroc_other_representations_range | {:.3f}–{:.3f} |".format(
             *p16["pca_k1_auroc_other_representations_range"]),
         "| 3,041 | r2/item4/image_level_overlap.json | n_test_sharing_lesion_with_train_or_val | {} |".format(ov["n_test_sharing_lesion_with_train_or_val"]),
         "| 5,067 | r2/item4/image_level_overlap.json | n_test | {} |".format(ov["n_test"]),
         "| 60.0% | r2/item4/image_level_overlap.json | pct_sharing | {:.2f} |".format(ov["pct_sharing"]),
         "| 2,026 | r2/item4/image_level_overlap.json | n_lesion_disjoint | {} |".format(ov["n_lesion_disjoint"]), "",
         "## Participation ratio table (phase 13, ISIC-train covariance), mean ± s.d.", "",
         "| λ | n | z_lesion^norm | z_lesion pre-BN | z_context | backbone (1536-d) |", "|---:|---:|---|---|---|---|"]
    for lam, v in t.items():
        L.append("| {} | {} | {} | {} | {} | {} |".format(lam, v["pr_z_lesion_norm"]["n"], *[
            C.fmt(v[k]["mean"], v[k]["sd"]) for k in ("pr_z_lesion_norm", "pr_z_lesion_preBN", "pr_z_context", "pr_backbone_raw_lesion")]))
    L += ["", "## Findings", ""]
    L += ["- " + d for d in diffs] if diffs else ["- None: every value equals its published report at the printed precision."]
    L += ["- Wording, not a value: the manuscript's 'participation ratio 4.879 → 4.500 while leakage falls 0.914 → 0.553' is "
          "λ = 0 → λ = 0.25 (PHASE13_REPORT.md 13.1). At λ = 2 the ratio is {:.3f} ± {:.3f}. The sentence sits after a "
          "λ = 2 statement and does not name the λ pair.".format(t["2"]["pr_z_lesion_norm"]["mean"], t["2"]["pr_z_lesion_norm"]["sd"])]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print(verdict)
    for d in diffs:
        print(" ", d)


if __name__ == "__main__":
    main()
