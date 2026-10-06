#!/usr/bin/env python3
"""Aggregate Phase 2 per-seed summaries → CSV + PHASE2_REPORT.md stub numbers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase2_pad_holdout")
SEEDS = (42, 52, 62, 72, 82)


def main():
    rows = []
    missing = []
    for s in SEEDS:
        p = ROOT / "runB_orth1_padhold_s{}".format(s) / "summary.json"
        if not p.exists():
            missing.append(str(p))
            continue
        obj = json.loads(p.read_text())
        rec = {
            "seed": s,
            "id_acc": obj["id_acc"],
            "id_balanced_acc": obj["id_balanced_acc"],
            "best_checkpoint": obj.get("best_checkpoint"),
        }
        for oname, mets in obj["ood"].items():
            rec["{}_z_lesion".format(oname)] = mets["z_lesion_maha_auroc"]
            rec["{}_z_context".format(oname)] = mets["z_context_maha_auroc"]
        rows.append(rec)
    if not rows:
        raise SystemExit("no phase2 summaries yet. missing:\n" + "\n".join(missing))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "phase2_per_seed.csv", index=False)
    agg = {}
    for c in df.columns:
        if c in ("seed", "best_checkpoint"):
            continue
        agg[c] = {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=1) if len(df) > 1 else 0.0), "n": int(len(df))}
    (ROOT / "phase2_aggregate.json").write_text(json.dumps({"n": len(df), "missing": missing, "aggregate": agg}, indent=2) + "\n")
    print(df.to_string(index=False))
    print("aggregate", json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
