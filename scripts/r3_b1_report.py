#!/usr/bin/env python3
"""R3 B1: lesion-level λ ∈ {0, 2} at n = 5 (seeds 42, 52, 62 from R2 Item 4 + 72, 82), the image-level seed set.

Full Table 1 columns; ID balanced accuracy drop as the per-seed paired difference λ=2 − λ=0 with a 95% t-interval."""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C  # noqa: E402
from r2_item4_report import LES_FEAT, LES_RUNS, row  # noqa: E402
from r2_item4_sweep_report import COLS_X, xdom  # noqa: E402

OUT = C.PAPERB / "results" / "paperB" / "r3" / "b1"
SEEDS = (42, 52, 62, 72, 82)
LAMS = (0.0, 2.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tab, missing = {}, []
    for lam in LAMS:
        tab[lam] = {}
        for s in SEEDS:
            tag = "runB_orth1_ladv{}_s{}".format(C.lam_tag(lam), s)
            if not (LES_RUNS / tag / "summary.json").exists() or not (LES_FEAT / tag / "pad_heldout.npz").exists():
                missing.append(tag)
                continue
            r = row(LES_RUNS, LES_FEAT, tag)
            r["xdom"] = xdom(tag)
            tab[lam][s] = r
    paired = [s for s in SEEDS if s in tab[0.0] and s in tab[2.0]]
    d = np.array([tab[2.0][s]["id_bal"] - tab[0.0][s]["id_bal"] for s in paired])
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    lo, hi = mean - half, mean + half
    verdict = ("Paired 95% interval excludes 0 -> the abstract keeps the ID balanced-accuracy drop as a result."
               if hi < 0 or lo > 0 else
               "Paired 95% interval includes 0 -> report the drop as a trend with the interval; do not call it a degradation.")
    head = C.git_head()
    L = ["# R3 B1 — lesion-level λ ∈ {0, 2} at n = 5", "",
         "Commit: `{}`. Seeds 42, 52, 62 (R2 Item 4) + 72, 82 (this item) = the image-level endpoint seed set. Same code and "
         "recipe as Item 4 (`scripts/train_r2_item4_lesion_split.py`, 40 epochs).".format(head), "",
         "**Verdict:** {}".format(verdict), ""]
    if missing:
        L += ["**Missing runs:** {}".format(", ".join(missing)), ""]
    L += ["## ID balanced accuracy, paired by seed (λ = 2 − λ = 0)", "",
          "| seed | λ = 0 | λ = 2 | difference |", "|---:|---|---|---|"]
    for s, x in zip(paired, d):
        L.append("| {} | {:.3f} | {:.3f} | {:+.3f} |".format(s, tab[0.0][s]["id_bal"], tab[2.0][s]["id_bal"], x))
    L += ["", "Mean difference {:+.4f}, s.d. {:.4f}, n = {}; 95% t-interval [{:+.4f}, {:+.4f}] (t, df = {}).".format(
        mean, sd, n, lo, hi, n - 1), "",
          "## Table 1 columns, lesion-level split, mean ± s.d.", "",
          "| λ | n | " + " | ".join(lab for _, lab in COLS_X) + " |", "|---:|---:|" + "---|" * len(COLS_X)]
    for lam in LAMS:
        rs = list(tab[lam].values())
        L.append("| {:g} | {} | ".format(lam, len(rs)) + " | ".join(C.fmt(*C.mean_sd([r[k] for r in rs])[:2]) for k, _ in COLS_X) + " |")
    L += ["", "## Per seed", "", "| λ | seed | " + " | ".join(lab for _, lab in COLS_X) + " |", "|---:|---:|" + "---|" * len(COLS_X)]
    for lam in LAMS:
        for s, r in tab[lam].items():
            L.append("| {:g} | {} | ".format(lam, s) + " | ".join("{:.3f}".format(r[k]) for k, _ in COLS_X) + " |")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    (OUT / "b1_results.json").write_text(json.dumps({
        "commit": head, "seeds": list(SEEDS), "missing": missing, "verdict": verdict,
        "id_bal_paired_diff": {"seeds": paired, "diffs": d.tolist(), "mean": mean, "sd": sd, "n": n, "t_ci95": [lo, hi]},
        "summary": {"{:g}".format(l): {k: dict(zip(("mean", "sd", "n"), C.mean_sd([r[k] for r in tab[l].values()])[:2]
                                                    + (len(tab[l]),))) for k, _ in COLS_X} for l in LAMS},
        "table": {"{:g}".format(l): {str(s): r for s, r in tab[l].items()} for l in LAMS}}, indent=1) + "\n")
    print(verdict, mean, (lo, hi), "missing", missing)


if __name__ == "__main__":
    main()
