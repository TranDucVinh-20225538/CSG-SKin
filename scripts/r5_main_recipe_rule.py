#!/usr/bin/env python3
"""Paper B R5 P8 main-recipe rule (decisions/paperB_r5_p8.md, "Main-recipe rule", 2026-10-15 23:48).

A4 vs the old Table 1 recipe at λ = 0, lesion-level validation split, last.ckpt for both, seeds 42–82 matched by seed.
D = mean_s (A4_s − old_s), 95 % paired t-interval (df = 4). A4 main iff D ≥ 0.03 and CI lower bound > 0.
Writes results/paperB/r5/p8/MAIN_RECIPE_RULE.{json,md}; exits 1 (no outcome) if any of the ten values is missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/r5")
P8 = ROOT / "p8"
OLD = ROOT / "ckpt_audit" / "compete"
SEEDS = (42, 52, 62, 72, 82)
D_MIN = 0.03
OUT = P8 / "MAIN_RECIPE_RULE.json"


def main():
    rows, missing = [], []
    for s in SEEDS:
        fo = OLD / "runB_orth1_ladv0_s{}.json".format(s)
        fa = P8 / "main" / "runs" / "main_ladv0_s{}".format(s) / "summary.json"
        a = json.loads(fa.read_text()) if fa.exists() else None
        if not fo.exists():
            missing.append(str(fo))
        if a is None or a.get("checkpoint_policy") != "last":
            missing.append(str(fa) + " (missing or not last.ckpt policy)")
        if missing:
            continue
        o = json.loads(fo.read_text())
        rows.append({"seed": s, "A4_val_bal_acc_last": a["metrics"]["val"]["bal_acc"], "A4_epoch": a["primary_epoch"],
                     "old_val_bal_acc_last": o["last_val_bal_acc"], "old_epoch": o["last_epoch"]})
    if missing:
        print("missing inputs, no outcome written:\n  " + "\n  ".join(missing), flush=True)
        sys.exit(1)
    d = np.array([r["A4_val_bal_acc_last"] - r["old_val_bal_acc_last"] for r in rows])
    for r, x in zip(rows, d):
        r["diff"] = float(x)
    n = len(d)
    D, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    lo, hi = D - half, D + half
    a4 = bool(D >= D_MIN and lo > 0)
    rec = {"rule": "decisions/paperB_r5_p8.md, Main-recipe rule (2026-10-15 23:48)", "split": "lesion-level ISIC val",
           "checkpoint": "last.ckpt (both recipes)", "seeds": list(SEEDS), "per_seed": rows, "D": D, "sd_diff": sd,
           "ci": [lo, hi], "t_df": n - 1, "D_min": D_MIN, "D_ge_D_min": bool(D >= D_MIN), "ci_excludes_0": bool(lo > 0),
           "outcome": "A4" if a4 else "old", "main_recipe": "p8" if a4 else "current", "p8_ext": "run" if a4 else "skip"}
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    L = ["# Main-recipe rule (decisions/paperB_r5_p8.md, 2026-10-15 23:48)", "",
         "λ = 0, lesion-level ISIC validation split, last.ckpt for both recipes, seeds matched.", "",
         "| seed | A4 val bal acc | old val bal acc | A4 − old |", "|---:|---:|---:|---:|"]
    L += ["| {} | {:.4f} | {:.4f} | {:+.4f} |".format(r["seed"], r["A4_val_bal_acc_last"], r["old_val_bal_acc_last"], r["diff"])
          for r in rows]
    L += ["", "D = {:+.4f}, 95 % paired t-interval [{:+.4f}, {:+.4f}] (df = {}).".format(D, lo, hi, n - 1),
          "D ≥ {}: {}; CI excludes 0: {}.".format(D_MIN, rec["D_ge_D_min"], rec["ci_excludes_0"]),
          "Outcome: **{}** main recipe (RECIPE=auto → {}); P8 ext: {}.".format(rec["outcome"], rec["main_recipe"], rec["p8_ext"]),
          "", "LaTeX row (D & CI lo & CI hi & outcome):", "", "```",
          "{:+.3f} & {:+.3f} & {:+.3f} & {} \\\\".format(D, lo, hi, rec["outcome"]), "```", ""]
    (P8 / "MAIN_RECIPE_RULE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
