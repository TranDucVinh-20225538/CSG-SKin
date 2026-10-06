#!/usr/bin/env python3
"""Aggregate Phase 15.1 single-encoder DANN. Gate first, then OOD numbers."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
from phase15_common import adversary_gate

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase15/single_dann")
LN2 = math.log(2.0)
OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase15/PHASE15_1_REPORT.md")
FLOOR = 0.5


def ms(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    if a.size == 1:
        return float(a[0]), 0.0, 1
    return float(a.mean()), float(a.std(ddof=1)), int(a.size)


def fmt(m, s):
    if m != m:
        return "—"
    return "{:.3f} ± {:.3f}".format(m, s)


def load_row(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    log_path = summary_path.parent / "adversary_log.jsonl"
    logged = []
    if log_path.exists():
        logged = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    gate = adversary_gate(
        logged,
        LN2,
        summary["lambda_adv"],
        leakage_bal=summary["leakage"]["pad_full"]["bal_acc_mean"],
        floor=0.5,
    )
    summary = dict(summary)
    summary["gate"] = gate
    return summary


def main():
    rows = []
    for p in sorted(ROOT.glob("single_dann_ladv*_s*/summary.json")):
        rows.append(load_row(p))
    by = defaultdict(list)
    for r in rows:
        by[float(r["lambda_adv"])].append(r)

    lines = [
        "# Phase 15.1 — single-encoder DANN on ISIC↔PAD",
        "",
        "Tier: **main text** (fixed before numbers). Leakage is 2-class balanced accuracy; floor **0.5**.",
        "OOD statistics fit on ISIC train only. Transfer is pad_heldout 6-class.",
        "",
        "n_summaries = {} / 25.".format(len(rows)),
        "",
        "Gate rescored from `adversary_log.jsonl` (summary.json on disk may still carry the old label).",
        "Valid when best-epoch adversary CE is below ln 2 − 0.02; at chance at convergence after that drop is success.",
        "",
    ]
    statuses = [r.get("gate", {}).get("status") for r in rows]
    n_valid = sum(s in ("valid", "valid_control") for s in statuses)
    n_inc = statuses.count("inconclusive")
    leak0 = [r["leakage"]["pad_full"]["bal_acc_mean"] for r in by.get(0.0, [])]
    leaks = {lam: [r["leakage"]["pad_full"]["bal_acc_mean"] for r in recs] for lam, recs in by.items()}
    moved = False
    if leak0:
        m0 = float(np.mean(leak0))
        for lam, vs in leaks.items():
            if lam <= 0:
                continue
            if vs and float(np.mean(vs)) <= m0 - 0.10:
                moved = True
    port = (
        "sound"
        if moved and n_inc == 0 and n_valid == len(rows)
        else ("broken_or_null" if n_inc == len(rows) and rows else "pending_or_mixed")
    )
    lines.append("## Gate")
    lines.append("")
    lines.append("- rescored: {} valid/control, {} inconclusive, {} total.".format(n_valid, n_inc, len(rows)))
    lines.append("- leakage moved toward 0.5 by ≥0.10 from λ=0: **{}**.".format(moved))
    lines.append("- port verdict: **{}**.".format(port))
    lines.append("")
    if port == "broken_or_null":
        lines.append("The port did not induce invariance on data where the dual-encoder does. Camelyon17 is a resolved implementation bug, not an open scientific question. 15.4 is not launched.")
        lines.append("")
    elif port == "sound":
        lines.append("The port reproduces a leakage drop. Camelyon17 may be retried (15.4) as a dataset-specific question.")
        lines.append("")

    lines.append("## Four-column table plus detectors / ECE / transfer")
    lines.append("")
    lines.append("| λ | n | leak bal | ID bal | Maha | kNN | MSP | cosine | Energy | OOD ECE | OOD conf | xfer bal |")
    lines.append("|---:|---:|---|---|---|---|---|---|---|---|---|---|")
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        recs = by.get(lam, [])
        if not recs:
            lines.append("| {:g} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |".format(lam))
            continue
        leak = ms(r["leakage"]["pad_full"]["bal_acc_mean"] for r in recs)
        idb = ms(r["id_balanced_acc"] for r in recs)
        maha = ms(r["ood"]["pad_full"]["mahalanobis_classcond"] for r in recs)
        knn = ms(r["ood"]["pad_full"]["knn_k50"] for r in recs)
        msp = ms(r["ood"]["pad_full"]["MSP"] for r in recs)
        cos = ms(r["ood"]["pad_full"]["cosine_max"] for r in recs)
        en = ms(r["ood"]["pad_full"]["Energy_T1"] for r in recs)
        ece = ms(r["ood_ece_pad_heldout"] for r in recs)
        conf = ms(r["ood_mean_confidence_pad_heldout"] for r in recs)
        xfer = ms(r["xfer_pad_heldout_6class"]["balanced_accuracy"] for r in recs)
        lines.append(
            "| {:g} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                lam, len(recs), fmt(*leak[:2]), fmt(*idb[:2]), fmt(*maha[:2]), fmt(*knn[:2]),
                fmt(*msp[:2]), fmt(*cos[:2]), fmt(*en[:2]), fmt(*ece[:2]), fmt(*conf[:2]), fmt(*xfer[:2]),
            )
        )
    lines.append("")
    lines.append("Per-run `gate` and `adversary_log.jsonl` sit next to each summary. Do not drop a λ because it is ugly.")
    lines.append("")
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT, "n", len(rows), "port", port)


if __name__ == "__main__":
    main()
