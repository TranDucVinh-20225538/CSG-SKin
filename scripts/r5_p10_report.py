#!/usr/bin/env python3
"""Paper B R5 P10 report: Camelyon17 two-hospital adversary toolbox ladder and conditional stage 2.

CPU only; reads results/paperB/r5/p10/{ladder,stage2}/*/summary.json and the R2 item-5 λ = 0 runs
(results/paperB/r2/item5/runs/camelyon2_ladv0_s{42,43,44}). Writes p10/{summary.csv, REPORT.md}.
Uncertainty: per-seed values and SD over seeds (as R2 item 5); no bootstrap (B held-out = 3 slides, so a slide-level
cluster bootstrap is degenerate and a patch-level one is anti-conservative).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
P10 = PAPERB / "results" / "paperB" / "r5" / "p10"
ITEM5 = PAPERB / "results" / "paperB" / "r2" / "item5" / "runs"
ORDER = ("L1_bank", "L2_ganin", "L3_strong", "L4_lam3", "L5_lam10")
GATE_PROBE, GATE_ID = 0.852, 0.85
DETS = ("mahalanobis_classcond_sharedcov", "knn_k50", "cosine_max", "MSP", "Energy_T1")


def flat(s, stage):
    o = s["ood"]
    adv = s.get("adversary", {})
    hist = s.get("history", [])
    return {"tag": s["tag"], "stage": stage, "rung": s.get("rung", "item5_lambda0"), "lambda_adv": s["lambda_adv"],
            "seed": s["seed"], "probe": s["leakage_site_probe_a_vs_bhold"]["balanced_acc_mean"],
            "probe_slide_disjoint": s["leakage_site_probe_aclean_vs_bhold"]["balanced_acc_mean"],
            "id_acc": s["id_test"]["acc"], "id_bal_acc": s["id_test"]["balanced_acc"],
            "id_acc_slide_disjoint": s["id_test_slide_disjoint"]["balanced_acc"],
            **{"bhold_" + d: o["b_hold"]["detectors"][d] for d in DETS},
            "c_maha": o["c"]["detectors"]["mahalanobis_classcond_sharedcov"],
            "bhold_acc": o["b_hold"]["accuracy"]["acc"],
            "adv_ce_min_epoch": min((r.get("loss_adv", r.get("adv_ce")) for r in hist), default=None),
            "adv_ce_last_epoch": (hist[-1].get("loss_adv", hist[-1].get("adv_ce")) if hist else None),
            "inner_ce_last": hist[-1].get("inner_ce") if hist else None,
            "gate_met": bool(s["leakage_site_probe_a_vs_bhold"]["balanced_acc_mean"] <= GATE_PROBE
                             and s["id_test"]["acc"] > GATE_ID),
            "gpu_hours": s.get("gpu_hours")}


def ms(x):
    a = np.asarray([v for v in x if v is not None], float)
    if a.size == 0:
        return "—"
    return "{:.3f}".format(a[0]) if a.size == 1 else "{:.3f} ± {:.3f}".format(a.mean(), a.std(ddof=1))


def main():
    rows = []
    for st in ("ladder", "stage2"):
        for f in sorted((P10 / st).glob("*/summary.json")):
            rows.append(flat(json.loads(f.read_text()), st))
    ref = [flat(json.loads((ITEM5 / "camelyon2_ladv0_s{}".format(s) / "summary.json").read_text()), "item5_ref")
           for s in (42, 43, 44) if (ITEM5 / "camelyon2_ladv0_s{}".format(s) / "summary.json").exists()]
    skipped = [json.loads(f.read_text()) for st in ("ladder", "stage2") for f in sorted((P10 / st).glob("*/skipped.json"))]
    if not rows:
        print("no P10 runs yet")
        return
    df = pd.DataFrame(rows + ref)
    df.to_csv(P10 / "summary.csv", index=False)
    used = float(np.nansum([r["gpu_hours"] or 0 for r in rows]))
    L = ["# Paper B R5 P10: Camelyon17 two-hospital adversary toolbox", "",
         "Gate (pre-committed in R2 item 5): site probe A test vs B held-out ≤ {:.3f} (λ = 0: 0.952 − 0.10) and A-test "
         "accuracy > {:.2f}. GPU-h used {:.2f} of 12.".format(GATE_PROBE, GATE_ID, used), "",
         "## Ladder (seed 42)", "",
         "| rung | λ | site probe | probe slide-disjoint | ID acc | min epoch adv CE | last adv CE | last inner CE | "
         "Maha B_heldout | kNN B_heldout | Maha C | gate met | GPU-h |", "|---|---:|" + "---:|" * 10 + "---:|"]
    lad = sorted([r for r in rows if r["stage"] == "ladder"], key=lambda r: ORDER.index(r["rung"]))
    f3 = lambda v: "—" if v is None else "{:.3f}".format(v)  # noqa: E731
    for r in lad:
        L.append("| {} | {:g} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.2f} |".format(
            r["rung"], r["lambda_adv"], *[f3(r[k]) for k in ("probe", "probe_slide_disjoint", "id_acc", "adv_ce_min_epoch",
                                                              "adv_ce_last_epoch", "inner_ce_last",
                                                              "bhold_mahalanobis_classcond_sharedcov", "bhold_knn_k50",
                                                              "c_maha")],
            r["gate_met"], r["gpu_hours"] or 0.0))
    for k in skipped:
        L.append("")
        L.append("Skipped `{}`: gate met by {}, GPU-h used {:.2f}.".format(k["tag"], k.get("gate_met_by", []),
                                                                           k["p10_gpu_h_used"]))
    L.append("")
    s2 = [r for r in rows if r["stage"] == "stage2"]
    hit = next((r for r in lad if r["gate_met"]), None)
    if hit is not None:
        cell = [r for r in lad if r["rung"] == hit["rung"]] + [r for r in s2 if r["rung"] == hit["rung"]]
        L += ["## Detectors rerun at the gate-meeting rung `{}` (λ = {:g}) vs λ = 0 (R2 item 5)".format(
            hit["rung"], hit["lambda_adv"]), "",
              "| cell | n | site probe | ID acc | Maha B_heldout | kNN | Cosine | MSP | Energy | Maha C | B_heldout acc |",
              "|---|---:|" + "---:|" * 9]
        tex = []
        for name, g in (("λ = 0 (item 5)", ref), ("{} λ = {:g}".format(hit["rung"], hit["lambda_adv"]), cell)):
            if not g:
                continue
            gd = pd.DataFrame(g)
            vals = [ms(gd[c]) for c in ("probe", "id_acc", *["bhold_" + d for d in DETS], "c_maha", "bhold_acc")]
            L.append("| {} | {} | {} |".format(name, len(g), " | ".join(vals)))
            tex.append("Camelyon17 {} & {} \\\\".format(name.replace("λ", "$\\lambda$"),
                                                       " & ".join(v.replace(" ± ", "\\,$\\pm$\\,") for v in vals)))
        L += ["", "```latex"] + tex + ["```", ""]
    L += ["## Results in plain words", ""]
    for r in lad:
        L.append("Rung {} (λ = {:g}): site probe {:.3f} (λ = 0: 0.952), ID accuracy {:.3f}, lowest epoch adversary CE {}; "
                 "B held-out Mahalanobis AUROC {:.3f}; gate {}.".format(
                     r["rung"], r["lambda_adv"], r["probe"], r["id_acc"], f3(r["adv_ce_min_epoch"]),
                     r["bhold_mahalanobis_classcond_sharedcov"], "met" if r["gate_met"] else "not met"))
        L.append("")
    if hit is None and lad:
        L.append("No rung met the gate within the cap; the Camelyon17 limitation stands as written (the adversary is "
                 "driven to chance while a linear probe still separates the hospitals).")
        L.append("")
    (P10 / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", P10 / "REPORT.md")


if __name__ == "__main__":
    main()
