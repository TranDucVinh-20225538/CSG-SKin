#!/usr/bin/env python3
"""Recompute adversary gate on Item 5 ResNet runs from adversary_log.jsonl (epoch aggregates)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
sys.path.insert(0, str(ROOT / "scripts"))
from phase15_common import LN2, adversary_gate

RUNS = ROOT / "results/paperB/reviewer_r1/resnet50_single_dann"


def epoch_rows(log_path: Path):
    rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["epoch"]].append(r)
    return [
        {
            "epoch": ep + 1,
            "acc_adv": sum(x["adv_acc"] for x in xs) / len(xs),
            "loss_adv": sum(x["adv_ce"] for x in xs) / len(xs),
        }
        for ep, xs in sorted(by.items())
    ]


def main():
    out_lines = ["# ResNet-50 Item 5 — adversary gate (rescored from logs)\n\n"]
    for p in sorted(RUNS.glob("resnet50_single_dann_ladv*/summary.json")):
        run_dir = p.parent
        log = run_dir / "adversary_log.jsonl"
        if not log.exists():
            continue
        d = json.loads(p.read_text())
        ep = epoch_rows(log)
        lam = float(d["lambda_adv"])
        leak = d["leakage_pad_heldout"]["bal_acc_mean"]
        gate = adversary_gate(ep, LN2, lam, leakage_bal=leak, floor=0.5)
        d["gate"] = gate
        d["gate_rescore_note"] = "From adversary_log.jsonl epoch means; fixes step-only logging bug."
        p.write_text(json.dumps(d, indent=2) + "\n")
        out_lines.append(
            f"- `{run_dir.name}`: **{gate['status']}** — {gate['reason']}\n"
        )
    report = ROOT / "results/paperB/reviewer_r1/ITEM5_RESNET50_GATE.md"
    report.write_text("".join(out_lines))
    print("updated summaries; wrote", report)


if __name__ == "__main__":
    main()
