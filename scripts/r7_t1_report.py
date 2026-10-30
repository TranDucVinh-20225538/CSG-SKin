#!/usr/bin/env python3
"""Paper B R7 T1 report: dual DINOv2 + ERM reference (decisions/paperB_r7_t1.md).

Writes results/paperB/r7/t1/{summary.csv,BOOTSTRAP.json,REPORT.md}. CPU only.
Computes reading-rule case A/B/C BEFORE any commentary.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = R.R7 / "t1"
LAMS = (0.0, 2.0)
OODS = ("pad_heldout", "fitzpatrick17k") + R.EXTRA_SETS
EFFB3_ERM_ID = (0.558, 0.022)  # mean ± sd quoted in the work order
LABELLED_UNSEEN = ("fitzpatrick17k", "mednode", "sd198")


def load_summaries():
    dual, erm = [], []
    for f in sorted((OUT / "runs").glob("*/summary.json")):
        s = json.loads(f.read_text())
        if s.get("penalty", "none") != "none":
            continue
        dual.append((s, f.parent))
    for f in sorted((OUT / "erm").glob("*/summary.json")):
        s = json.loads(f.read_text())
        erm.append((s, f.parent))
    return dual, erm


def rule_a_inverts(boot_cell):
    """Rule A: pooled CI upper < 0.5 and ≥3 seeds with CI upper < 0.5."""
    if not boot_cell:
        return False
    return boot_cell.get("rule") == "A"


def classify_case(gap, boot, leak_l2):
    """Reading rules from decisions/paperB_r7_t1.md (locked). Returns case letter + reason."""
    near = gap <= 0.03
    pad_inv = rule_a_inverts(boot.get("dual|lam2", {}).get("pad_heldout", {}).get("mahalanobis_classcond"))
    n_unseen = sum(
        1 for o in LABELLED_UNSEEN
        if rule_a_inverts(boot.get("dual|lam2", {}).get(o, {}).get("mahalanobis_classcond"))
    )
    no_inv_any = (not pad_inv) and all(
        not rule_a_inverts(boot.get("dual|lam2", {}).get(o, {}).get("mahalanobis_classcond"))
        for o in ("pad_heldout",) + LABELLED_UNSEEN
    )
    leak_hi = leak_l2 is not None and leak_l2 > 0.60
    if near and pad_inv and n_unseen >= 2:
        return "A", "near ERM (gap={:.3f}) and Rule A on pad_heldout plus {}/3 labelled unseen".format(gap, n_unseen)
    if near and no_inv_any:
        return "B", "near ERM (gap={:.3f}) and Rule A fails on pad_heldout and every unseen set".format(gap)
    return "C", ("gap={:.3f} (near={}), pad Rule A={}, labelled-unseen Rule A count={}, "
                 "leakage λ=2={}".format(gap, near, pad_inv, n_unseen,
                                         "—" if leak_l2 is None else "{:.3f}".format(leak_l2)))


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dual, erm = load_summaries()
    if not dual:
        print("no T1 dual runs yet under", OUT / "runs")
        return
    rows = []
    cells = defaultdict(dict)
    for s, d in dual:
        arm = "dual " + s["backbone"]
        rows.append({"arm": arm, "tag": s["tag"], "lambda_adv": float(s["lambda_adv"]), "seed": s["seed"],
                     **R.table1_values(s), "knn_pad_heldout": s["ood"]["pad_heldout"]["knn_k50"],
                     "nv_nearest_pad_heldout": s["ood"]["pad_heldout"]["majority_nearest_frac"],
                     **{"maha_" + e: s["ood"][e]["mahalanobis_classcond"] for e in R.EXTRA_SETS if e in s["ood"]},
                     "gate": s.get("gate", {}).get("status"), "gpu_hours": s.get("gpu_hours"),
                     "degraded": bool(s["id_balanced_acc"] < 0.25)})
        cells["{}|lam{:g}".format(arm, float(s["lambda_adv"]))][int(s["seed"])] = str(d)
    for s, d in erm:
        arm = "ERM " + s["backbone"]
        rows.append({"arm": arm, "tag": s["tag"], "lambda_adv": float(s["lambda_adv"]), "seed": s["seed"],
                     **R.table1_values(s), "knn_pad_heldout": s["ood"]["pad_heldout"]["knn_k50"],
                     "nv_nearest_pad_heldout": s["ood"]["pad_heldout"]["majority_nearest_frac"],
                     **{"maha_" + e: s["ood"][e]["mahalanobis_classcond"] for e in R.EXTRA_SETS if e in s["ood"]},
                     "gate": s.get("gate", {}).get("status"), "gpu_hours": s.get("gpu_hours"),
                     "degraded": bool(s["id_balanced_acc"] < 0.25)})
        cells["{}|lam{:g}".format(arm, float(s["lambda_adv"]))][int(s["seed"])] = str(d)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    B = R.isic_bootstrap(cells, OODS)
    (OUT / "BOOTSTRAP.json").write_text(json.dumps(B, indent=2) + "\n")

    dual0 = df[(df.arm.str.startswith("dual")) & (df.lambda_adv == 0.0)]
    erm0 = df[(df.arm.str.startswith("ERM")) & (df.lambda_adv == 0.0)]
    dual2 = df[(df.arm.str.startswith("dual")) & (df.lambda_adv == 2.0)]
    id_dual0 = float(dual0.id_bal_acc.mean()) if len(dual0) else float("nan")
    id_erm = float(erm0.id_bal_acc.mean()) if len(erm0) else float("nan")
    gap = id_erm - id_dual0 if np.isfinite(id_erm) and np.isfinite(id_dual0) else float("nan")
    leak_l2 = float(dual2.leakage.mean()) if len(dual2) else None
    # remap boot keys for classify (arm names vary)
    boot_map = {}
    for k, v in B.items():
        if k.startswith("dual ") and "|lam2" in k:
            boot_map["dual|lam2"] = v
        elif k.startswith("dual ") and "|lam0" in k:
            boot_map["dual|lam0"] = v
    case, reason = classify_case(gap, boot_map, leak_l2)

    L = ["# Paper B R7 T1: dual DINOv2 + ERM reference", "",
         "Design: `decisions/paperB_r7_t1.md`. Commit `{}`.".format(R.git_head()), "",
         "## Reading-rule case (computed first)", "",
         "**Case {}.** {}.".format(case, reason), "",
         "Competence gap = ERM-DINOv2 mean ID bal − dual λ=0 mean ID bal = "
         "{:.3f} − {:.3f} = {:.3f} (near-ERM iff ≤ 0.03).".format(id_erm, id_dual0, gap),
         "EfficientNet-B3 ERM reference (work order): {:.3f} ± {:.3f}; dual λ=0 vs that gap = {:.3f}.".format(
             EFFB3_ERM_ID[0], EFFB3_ERM_ID[1], EFFB3_ERM_ID[0] - id_dual0),
         "Leakage at λ=2 (mean): {}.".format("—" if leak_l2 is None else "{:.3f}".format(leak_l2)), "",
         "| arm | λ | n | " + " | ".join(R.TABLE1_HEAD) + " | pad rule | Fitz rule |",
         "|---|---:|---:|" + "---:|" * len(R.TABLE1_HEAD) + "---|---|"]
    tex = []
    for k in sorted(cells):
        arm, lam = k.split("|lam")
        g = df[(df.arm == arm) & (df.lambda_adv == float(lam))]
        b = B.get(k, {})
        ru = [b.get(o, {}).get("mahalanobis_classcond", {}).get("rule", "—") for o in ("pad_heldout", "fitzpatrick17k")]
        L.append("| {} | {} | {} | {} | {} | {} |".format(
            arm, lam, len(g), " | ".join(R.ms(g[c]) for c in R.TABLE1_COLS), *ru))
        tex.append("{} $\\lambda={}$ & {} \\\\".format(arm, lam, " & ".join(R.tex_ms(g[c]) for c in R.TABLE1_COLS)))
    L += ["", "### Table 2 (unseen Mahalanobis)", "",
          "| arm | λ | " + " | ".join(R.EXTRA_SETS) + " |",
          "|---|---:|" + "|".join(["---:"] * len(R.EXTRA_SETS)) + "|"]
    for k in sorted(cells):
        arm, lam = k.split("|lam")
        g = df[(df.arm == arm) & (df.lambda_adv == float(lam))]
        L.append("| {} | {} | {} |".format(arm, lam, " | ".join(
            R.ms(g["maha_" + e]) if "maha_" + e in g else "—" for e in R.EXTRA_SETS)))
    L += ["", "```latex"] + tex + ["```", "",
          "GPU hours (sum of run summaries): {:.2f}.".format(float(df.gpu_hours.fillna(0).sum())), "",
          "## Plain paragraph", "",
          "Case {}: {}. Dual DINOv2 λ=0 ID balanced accuracy {}; ERM-DINOv2 {}; "
          "competence gap {:.3f}; EffB3-ERM reference 0.558 ± 0.022.".format(
              case, reason, R.ms(dual0.id_bal_acc) if len(dual0) else "—",
              R.ms(erm0.id_bal_acc) if len(erm0) else "—", gap), ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    (OUT / "CASE.json").write_text(json.dumps({
        "case": case, "reason": reason, "gap": gap, "id_dual_l0": id_dual0, "id_erm": id_erm,
        "leakage_l2": leak_l2}, indent=2) + "\n")
    print("case", case, "gap", gap, "wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
