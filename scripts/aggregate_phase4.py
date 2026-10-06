#!/usr/bin/env python3
"""Aggregate Phase 4 per-run summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase4_semantic_ood")
CONFIGS = ("4a", "4b")
METHODS = ("baseline", "effb3", "runB_orth1")
SEEDS = (42, 52, 62)


def flatten_ood(ood):
    out = {}
    for space, mets in ood.items():
        out["{}_auroc".format(space)] = mets.get("AUROC")
        out["{}_dim".format(space)] = mets.get("feat_dim")
    return out


def main():
    rows = []
    missing = []
    for cfg in CONFIGS:
        for method in METHODS:
            for s in SEEDS:
                p = ROOT / "{}_{}_s{}".format(cfg, method, s) / "summary.json"
                if not p.exists():
                    missing.append(str(p))
                    continue
                obj = json.loads(p.read_text())
                rec = {
                    "config": cfg,
                    "method": method,
                    "seed": s,
                    "n_classes": obj["n_classes"],
                    "id_acc": obj["id_acc"],
                    "id_balanced_acc": obj["id_balanced_acc"],
                    "id_ece": obj["id_ece"],
                    "n_id": obj["n_id"],
                    "n_ood": obj["n_ood"],
                    "hold_out": ",".join(obj["hold_out"]),
                }
                rec.update(flatten_ood(obj["ood"]))
                rows.append(rec)
    if not rows:
        raise SystemExit("no phase4 summaries yet")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "phase4_per_seed.csv", index=False)
    grp = df.groupby(["config", "method"], as_index=False).agg(
        n=("seed", "count"),
        id_bal_mean=("id_balanced_acc", "mean"),
        id_bal_std=("id_balanced_acc", "std"),
    )
    # add auroc means if columns exist
    for col in df.columns:
        if col.endswith("_auroc"):
            g2 = df.groupby(["config", "method"])[col].agg(["mean", "std", "count"]).reset_index()
            g2.columns = ["config", "method", col + "_mean", col + "_std", col + "_n"]
            grp = grp.merge(g2, on=["config", "method"], how="left")
    grp.to_csv(ROOT / "phase4_aggregate.csv", index=False)
    (ROOT / "phase4_aggregate.json").write_text(
        json.dumps({"n_rows": len(df), "missing": missing, "aggregate": grp.to_dict(orient="records")}, indent=2) + "\n"
    )
    print(df.to_string(index=False))
    print(grp.to_string(index=False))
    if missing:
        print("missing {} / {}".format(len(missing), 18))


if __name__ == "__main__":
    main()
