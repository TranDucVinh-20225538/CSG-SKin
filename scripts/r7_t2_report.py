#!/usr/bin/env python3
"""Paper B R7 T2 report: pad_adv size sensitivity (decisions/paperB_r7_t2.md)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402
import pandas as pd  # noqa: E402

OUT = R.R7 / "t2"
OODS = ("pad_heldout", "fitzpatrick17k") + R.EXTRA_SETS
# Table 1 100% reference seeds 42,52,62 at λ=0/2 from t1last
T1LAST = R.R5 / "t1last" / "per_run"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for f in sorted((OUT / "runs").glob("*/summary.json")):
        runs.append((json.loads(f.read_text()), f.parent))
    if not runs:
        print("no T2 runs")
        return
    rows, cells = [], defaultdict(dict)
    for s, d in runs:
        frac = float(s.get("pad_adv_frac", 1.0))
        arm = "pad_adv {:.0%}".format(frac)
        rows.append({"arm": arm, "frac": frac, "tag": s["tag"], "lambda_adv": float(s["lambda_adv"]),
                     "seed": s["seed"], **R.table1_values(s), "gpu_hours": s.get("gpu_hours"),
                     "pad_class_counts": json.dumps(s.get("pad_adv_subsample", {}).get("class_counts", {}))})
        cells["{}|lam{:g}".format(arm, float(s["lambda_adv"]))][int(s["seed"])] = str(d)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    B = R.isic_bootstrap(cells, OODS)
    (OUT / "BOOTSTRAP.json").write_text(json.dumps(B, indent=2) + "\n")
    L = ["# Paper B R7 T2: pad_adv size", "",
         "Design: `decisions/paperB_r7_t2.md`. Commit `{}`.".format(R.git_head()), "",
         "pad_adv subsample feeds **both** adversarial and context branches; pad_heldout unchanged. "
         "With 3 seeds, Rule A means all 3 seed CI uppers < 0.5.", "",
         "| arm | λ | n | " + " | ".join(R.TABLE1_HEAD) + " | pad Rule A |",
         "|---|---:|---:|" + "|".join(["---:"] * len(R.TABLE1_HEAD)) + "|---|"]
    for k in sorted(cells):
        arm, lam = k.split("|lam")
        g = df[(df.arm == arm) & (df.lambda_adv == float(lam))]
        rule = B.get(k, {}).get("pad_heldout", {}).get("mahalanobis_classcond", {}).get("rule", "—")
        L.append("| {} | {} | {} | {} | {} |".format(
            arm, lam, len(g), " | ".join(R.ms(g[c]) for c in R.TABLE1_COLS), rule))
    L += ["", "Class counts per subsample (from first run of each frac):", ""]
    seen = set()
    for s, _ in runs:
        frac = float(s.get("pad_adv_frac", 1.0))
        if frac in seen:
            continue
        seen.add(frac)
        meta = s.get("pad_adv_subsample", {})
        L.append("- {:.0%}: n_images={}, n_patients={}, counts={}".format(
            frac, meta.get("n_images"), meta.get("n_patients"), meta.get("class_counts")))
    L += ["", "GPU hours: {:.2f}.".format(float(df.gpu_hours.fillna(0).sum())), ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
