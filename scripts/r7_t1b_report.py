#!/usr/bin/env python3
"""Paper B R7 T1b report (decisions/paperB_r7_t1b.md)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = R.R7 / "t1b"
OODS = ("pad_heldout", "fitzpatrick17k") + R.EXTRA_SETS


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for f in sorted((OUT / "runs").glob("*/summary.json")):
        runs.append((json.loads(f.read_text()), f.parent))
    if not runs:
        # also accept stage-less
        for f in sorted(OUT.glob("runs/*/summary.json")):
            runs.append((json.loads(f.read_text()), f.parent))
    if not runs:
        (OUT / "REPORT.md").write_text(
            "# Paper B R7 T1b\n\nNo runs yet.\n\n"
            "If T1b cannot run: W1 is not addressed by R7; the title and scope must match what the paper shows (dual encoder).\n")
        print("no T1b runs")
        return
    rows, cells = [], defaultdict(dict)
    for s, d in runs:
        rows.append({"tag": s["tag"], "lambda_adv": float(s["lambda_adv"]), "seed": s["seed"],
                     **R.table1_values(s), "gpu_hours": s.get("gpu_hours"),
                     "degraded": bool(s["id_balanced_acc"] < 0.25)})
        cells["single|lam{:g}".format(float(s["lambda_adv"]))][int(s["seed"])] = str(d)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    B = R.isic_bootstrap(cells, OODS)
    (OUT / "BOOTSTRAP.json").write_text(json.dumps(B, indent=2) + "\n")
    leak = float(df.leakage.mean())
    inv = leak <= 0.60
    pad = B.get("single|lam2", {}).get("pad_heldout", {}).get("mahalanobis_classcond", {})
    # With 3 seeds Rule A requires all 3 CI uppers < 0.5
    inverts = bool(pad) and pad.get("rule") == "A" and pad.get("k_below", 0) >= 3 and pad.get("n_seeds", 0) == 3
    if inv and inverts:
        outcome = "invariance reached and inverts on pad_heldout: W1 answered for standard DANN on this backbone"
    elif inv and not inverts:
        outcome = ("invariance reached and does not invert: standard DANN at comparable invariance does not invert; "
                   "title/scope must say the effect is shown for the dual encoder")
    else:
        outcome = "invariance not reached (V1={:.3f}>0.60): test did not take place; W1 stays open".format(leak)
    L = ["# Paper B R7 T1b: single-encoder DANN DINOv2-S/14", "",
         "Design: `decisions/paperB_r7_t1b.md`. Commit `{}`.".format(R.git_head()), "",
         "## Reading-rule outcome (computed first)", "", outcome, "",
         "Held-out leakage V1 mean: {:.3f}. Invariance (≤0.60): {}. pad_heldout Rule A: {} (k_below={}/{}).".format(
             leak, inv, pad.get("rule"), pad.get("k_below"), pad.get("n_seeds")), "",
         "| λ | n | " + " | ".join(R.TABLE1_HEAD) + " |",
         "|---:|---:|" + "|".join(["---:"] * len(R.TABLE1_HEAD)) + "|"]
    for lam in sorted(df.lambda_adv.unique()):
        g = df[df.lambda_adv == lam]
        L.append("| {:g} | {} | {} |".format(lam, len(g), " | ".join(R.ms(g[c]) for c in R.TABLE1_COLS)))
    L += ["", "GPU hours: {:.2f}.".format(float(df.gpu_hours.fillna(0).sum())), ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    (OUT / "OUTCOME.json").write_text(json.dumps({"outcome": outcome, "leakage": leak, "invariance": inv,
                                                   "inverts": inverts, "pad_rule": pad}, indent=2) + "\n")
    print(outcome)


if __name__ == "__main__":
    main()
