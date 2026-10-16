#!/usr/bin/env python3
"""Paper B R5 P10 report (decisions/paperB_r5_p10.md).

Gate: held-out hospital probe drop ≥ 0.10 vs λ = 0; else “invariance not reached”.
Reports B_heldout accuracy next to OOD AUROC. Cap 40 GPU-h.
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
GATE_DROP = 0.10
CAP = 40.0
DETS = ("mahalanobis_classcond_sharedcov", "knn_k50", "cosine_max", "MSP", "Energy_T1")


def lambda0_probe_mean():
    vals = []
    for s in (42, 43, 44):
        p = ITEM5 / "camelyon2_ladv0_s{}".format(s) / "summary.json"
        if p.exists():
            vals.append(json.loads(p.read_text())["leakage_site_probe_a_vs_bhold"]["balanced_acc_mean"])
    return float(np.mean(vals)) if vals else None


def flat(s, stage, probe0):
    o = s["ood"]
    hist = s.get("history", [])
    probe = s["leakage_site_probe_a_vs_bhold"]["balanced_acc_mean"]
    drop = s.get("probe_drop_vs_lambda0")
    if drop is None and probe0 is not None:
        drop = float(probe0 - probe)
    gate = bool(s.get("gate_met", drop is not None and drop >= GATE_DROP))
    return {"tag": s["tag"], "stage": stage, "rung": s.get("rung", "item5_lambda0"), "lambda_adv": s["lambda_adv"],
            "seed": s["seed"], "probe": probe, "probe_drop": drop,
            "probe_slide_disjoint": s["leakage_site_probe_aclean_vs_bhold"]["balanced_acc_mean"],
            "id_acc": s["id_test"]["acc"], "id_bal_acc": s["id_test"]["balanced_acc"],
            "id_acc_slide_disjoint": s["id_test_slide_disjoint"]["balanced_acc"],
            **{"bhold_" + d: o["b_hold"]["detectors"][d] for d in DETS},
            "c_maha": o["c"]["detectors"]["mahalanobis_classcond_sharedcov"],
            "bhold_acc": o["b_hold"]["accuracy"]["acc"],
            "bhold_bal_acc": o["b_hold"]["accuracy"].get("balanced_acc"),
            "adv_ce_min_epoch": min((r.get("loss_adv", r.get("adv_ce")) for r in hist), default=None),
            "adv_ce_last_epoch": (hist[-1].get("loss_adv", hist[-1].get("adv_ce")) if hist else None),
            "inner_ce_last": hist[-1].get("inner_ce") if hist else None,
            "gate_met": gate,
            "invariance_not_reached": not gate,
            "gpu_hours": s.get("gpu_hours")}


def ms(x):
    a = np.asarray([v for v in x if v is not None], float)
    if a.size == 0:
        return "—"
    return "{:.3f}".format(a[0]) if a.size == 1 else "{:.3f} ± {:.3f}".format(a.mean(), a.std(ddof=1))


def main():
    probe0 = lambda0_probe_mean()
    rows = []
    for st in ("ladder", "stage2"):
        for f in sorted((P10 / st).glob("*/summary.json")):
            rows.append(flat(json.loads(f.read_text()), st, probe0))
    ref = [flat(json.loads((ITEM5 / "camelyon2_ladv0_s{}".format(s) / "summary.json").read_text()), "item5_ref",
                probe0)
           for s in (42, 43, 44) if (ITEM5 / "camelyon2_ladv0_s{}".format(s) / "summary.json").exists()]
    skipped = [json.loads(f.read_text()) for st in ("ladder", "stage2") for f in sorted((P10 / st).glob("*/skipped.json"))]
    if not rows and not ref:
        print("no P10 runs yet")
        return
    df = pd.DataFrame(rows + ref)
    P10.mkdir(parents=True, exist_ok=True)
    df.to_csv(P10 / "summary.csv", index=False)
    used = float(np.nansum([r["gpu_hours"] or 0 for r in rows]))
    L = ["# Paper B R5 P10: Camelyon17 two-hospital adversary toolbox", "",
         "Design: `decisions/paperB_r5_p10.md`. Gate: held-out hospital probe drop ≥ {:.2f} vs λ = 0 mean "
         "({:.3f}); otherwise **invariance not reached** (inconclusive). GPU-h used {:.2f} of {:.0f}. "
         "Semantic OOD: NCT-CRC-HE-100K (CC-BY-4.0). B_heldout accuracy is reported next to OOD AUROC "
         "(competence vs detection).".format(GATE_DROP, probe0 if probe0 is not None else float("nan"), used, CAP),
         "",
         "## Ladder (seed 42)", "",
         "| rung | λ | site probe | Δprobe vs λ0 | ID acc | B_heldout acc | Maha B_heldout | kNN | "
         "Maha C | gate | invariance | GPU-h |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
    lad = sorted([r for r in rows if r["stage"] == "ladder"], key=lambda r: ORDER.index(r["rung"]))
    f3 = lambda v: "—" if v is None else "{:.3f}".format(v)  # noqa: E731
    for r in lad:
        L.append("| {} | {:g} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.2f} |".format(
            r["rung"], r["lambda_adv"],
            f3(r["probe"]), f3(r["probe_drop"]), f3(r["id_acc"]), f3(r["bhold_acc"]),
            f3(r["bhold_mahalanobis_classcond_sharedcov"]), f3(r["bhold_knn_k50"]), f3(r["c_maha"]),
            "met" if r["gate_met"] else "no",
            "not reached" if r["invariance_not_reached"] else "—",
            r["gpu_hours"] or 0.0))
    for k in skipped:
        L += ["", "Skipped `{}`: gate met by {}, GPU-h used {:.2f} / {}.".format(
            k["tag"], k.get("gate_met_by", []), k["p10_gpu_h_used"], k.get("cap", CAP))]
    L.append("")
    hit = next((r for r in lad if r["gate_met"]), None)
    s2 = [r for r in rows if r["stage"] == "stage2"]
    if hit is None and lad:
        L += ["## Gate", "",
              "**Invariance not reached** on the ladder (no rung with probe drop ≥ {:.2f}). "
              "Detection collapse/inversion on Camelyon is inconclusive from P10.".format(GATE_DROP), ""]
    elif hit is not None:
        cell = [r for r in lad if r["rung"] == hit["rung"]] + [r for r in s2 if r["rung"] == hit["rung"]]
        L += ["## Detectors at gate-meeting rung `{}` (λ = {:g}) vs λ = 0".format(hit["rung"], hit["lambda_adv"]),
              "",
              "| cell | n | site probe | ID acc | B_heldout acc | Maha B_heldout | kNN | Cosine | MSP | Energy | Maha C |",
              "|---|---:|" + "---:|" * 9]
        for name, g in (("λ = 0 (item 5)", ref), ("{} λ = {:g}".format(hit["rung"], hit["lambda_adv"]), cell)):
            if not g:
                continue
            gd = pd.DataFrame(g)
            vals = [ms(gd[c]) for c in ("probe", "id_acc", "bhold_acc", *["bhold_" + d for d in DETS], "c_maha")]
            L.append("| {} | {} | {} |".format(name, len(g), " | ".join(vals)))
        L.append("")
    L += ["## Predicted outcome (committed before runs)", "",
          "If the hospital probe falls by ≥ 0.10 vs λ = 0: detection on the held-out hospital inverts, and "
          "NCT-CRC-HE at least collapses. If the gate fails: invariance not reached.", ""]
    (P10 / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
