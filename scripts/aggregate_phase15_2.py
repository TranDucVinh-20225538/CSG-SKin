#!/usr/bin/env python3
"""Aggregate 15.2: OOD AUROC vs achieved leakage, one curve per objective."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase15/objectives")
OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase15/PHASE15_2_REPORT.md")
FLOOR = 0.5


def ms(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def main():
    rows = []
    for p in sorted(ROOT.glob("*_w*_s*/summary.json")):
        rows.append(json.loads(p.read_text()))
    lines = [
        "# Phase 15.2 — non-adversarial objectives",
        "",
        "Tier: **main text** if 15.1 port is sound. Plot against **achieved leakage**, not nominal weight.",
        "Leakage floor **0.5**. CORAL/MMD/ERM never use PAD labels. IRM/GroupDRO use pad_adv labels; pad_heldout unseen.",
        "",
        "n_summaries = {} / 39.".format(len(rows)),
        "",
        "| objective | weight | n | leak bal | Maha | ID bal | xfer bal | pad labels in L_cls |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    by = defaultdict(list)
    for r in rows:
        by[(r["objective"], float(r["weight"]))].append(r)
    for key in sorted(by, key=lambda k: (k[0], k[1])):
        recs = by[key]
        leak = ms(r["leakage"]["bal_acc_mean"] for r in recs)
        maha = ms(r["ood_pad_full"]["mahalanobis_classcond"] for r in recs)
        idb = ms(r["id_balanced_acc"] for r in recs)
        xfer = ms(r["xfer_pad_heldout_6class"]["balanced_accuracy"] for r in recs)
        lines.append(
            "| {} | {:g} | {} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {} |".format(
                key[0], key[1], len(recs), leak[0], leak[1], maha[0], maha[1], idb[0], idb[1], xfer[0], xfer[1],
                recs[0]["uses_pad_adv_labels"],
            )
        )
    lines.append("")
    lines.append("If all objectives collapse together vs leakage, the mechanism is invariance. If GRL (15.1) separates, it is adversarial dynamics.")
    lines.append("")
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT, "n", len(rows))


if __name__ == "__main__":
    main()
