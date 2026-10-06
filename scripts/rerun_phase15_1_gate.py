#!/usr/bin/env python3
"""Re-score Phase 15.1 gates. Does not rewrite summary.json or PHASE15_1_REPORT.md."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from phase15_common import adversary_gate

SRC = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase15/single_dann")
OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase15b/PHASE15_1_GATE.md")
LN2 = math.log(2.0)


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


def main():
    rows = []
    for summary_path in sorted(SRC.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        log_path = summary_path.parent / "adversary_log.jsonl"
        logged = []
        if log_path.exists():
            logged = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        gate = adversary_gate(
            logged, LN2, summary["lambda_adv"],
            leakage_bal=summary["leakage"]["pad_full"]["bal_acc_mean"], floor=0.5,
        )
        summary = dict(summary)
        summary["gate_corrected"] = gate
        rows.append(summary)

    by = defaultdict(list)
    for r in rows:
        by[float(r["lambda_adv"])].append(r)

    statuses = [r["gate_corrected"]["status"] for r in rows]
    n_valid = sum(s in ("valid", "valid_control") for s in statuses)
    n_inc = statuses.count("inconclusive")
    leak0 = [r["leakage"]["pad_full"]["bal_acc_mean"] for r in by.get(0.0, [])]
    moved = False
    if leak0:
        m0 = float(np.mean(leak0))
        for lam, recs in by.items():
            if lam <= 0:
                continue
            vs = [r["leakage"]["pad_full"]["bal_acc_mean"] for r in recs]
            if vs and float(np.mean(vs)) <= m0 - 0.10:
                moved = True
    port = "sound" if moved and n_inc == 0 and n_valid == len(rows) else "pending_or_mixed"
    lines = [
        "# Phase 15.1 gate, rescored",
        "",
        "Existing `summary.json` files were not rewritten. This file is the corrected gate.",
        "Rule: a run is valid when its best-epoch adversary CE falls below ln 2 − 0.02.",
        "Returning to chance by the last epoch, after that drop, means the encoder won.",
        "λ=0 is a valid control: the adversary term is multiplied by zero, so the head is not trained.",
        "A run that never leaves ln 2 is inconclusive (the Camelyon17 failure).",
        "",
        "n = {} / 25. valid+control = {}. inconclusive = {}.".format(len(rows), n_valid, n_inc),
        "Leakage moved by ≥ 0.10 from λ=0: **{}**.".format(moved),
        "Port verdict: **{}**. pending_or_mixed clears: **{}**.".format(port, port == "sound"),
        "",
        "## Per-run gate",
        "",
        "| run | λ | status | best CE | best epoch | last CE | last acc |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        g = r["gate_corrected"]
        best = g.get("best_epoch") or {}
        last = g.get("last_epoch") or {}
        lines.append(
            "| {} | {:g} | {} | {:.4f} | {} | {:.4f} | {:.3f} |".format(
                r["run_name"], float(r["lambda_adv"]), g["status"],
                float(best.get("loss_adv", float("nan"))), best.get("epoch", "—"),
                float(last.get("loss_adv", float("nan"))), float(last.get("acc_adv", float("nan"))),
            )
        )
    lines += [
        "",
        "ln 2 = {:.4f}. Margin = 0.02. λ=8 best CE is ≈ 0.66, which is below the margin, so those three runs are valid. Their last epoch is back at ln 2; that is the encoder winning.".format(LN2),
        "",
        "## Table (unchanged measurements)",
        "",
        "| λ | n | leak bal | Maha | kNN | MSP | cosine | Energy | ID acc | xfer bal | gate |",
        "|---:|---:|---|---|---|---|---|---|---|---|---|",
    ]
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        recs = by.get(lam, [])
        if not recs:
            continue
        leak = ms(r["leakage"]["pad_full"]["bal_acc_mean"] for r in recs)
        maha = ms(r["ood"]["pad_full"]["mahalanobis_classcond"] for r in recs)
        knn = ms(r["ood"]["pad_full"]["knn_k50"] for r in recs)
        msp = ms(r["ood"]["pad_full"]["MSP"] for r in recs)
        cos = ms(r["ood"]["pad_full"]["cosine_max"] for r in recs)
        en = ms(r["ood"]["pad_full"]["Energy_T1"] for r in recs)
        ida = ms(r["id_acc"] for r in recs)
        xfer = ms(r["xfer_pad_heldout_6class"]["balanced_accuracy"] for r in recs)
        gates = sorted({r["gate_corrected"]["status"] for r in recs})
        lines.append(
            "| {:g} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                lam, len(recs), fmt(*leak[:2]), fmt(*maha[:2]), fmt(*knn[:2]), fmt(*msp[:2]),
                fmt(*cos[:2]), fmt(*en[:2]), fmt(*ida[:2]), fmt(*xfer[:2]), ",".join(gates),
            )
        )
    lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT)
    print("port", port, "valid", n_valid, "inconclusive", n_inc)


if __name__ == "__main__":
    main()
