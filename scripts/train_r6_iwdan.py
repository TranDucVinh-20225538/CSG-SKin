#!/usr/bin/env python3
"""Paper B R6 R2: label-shift-aware alignment on the dual encoder (decisions/paperB_r6_r2.md).

CDAN (Long et al. 2018, no entropy conditioning), IWDAN and IWCDAN (Tachet des Combes et al. 2020): target class weights
estimated without target labels (source confusion matrix + target prediction histogram, QP, once per epoch, moving
average 0.5; source classification and domain losses weighted). Oracle IWDAN: w fixed from true pad_adv proportions
(PAD labels set weights only; never in a loss) — labelled oracle, not deployable.
Estimated: λ ∈ {0.5, 2} × 3 methods × 5 seeds = 30. Oracle: λ ∈ {0.5, 2} × 5 seeds = 10. Total 40.

Run order (deviation 2026-10-17, order only): λ=2 IWDAN → IWCDAN → oracle → CDAN; then all λ=0.5
(IWDAN, IWCDAN, CDAN, oracle). Same 40 cells; criteria unchanged.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

# (lam, method) — method "iwdan_oracle" is the oracle branch
ORDER = (
    (2.0, "iwdan"),
    (2.0, "iwcdan"),
    (2.0, "iwdan_oracle"),
    (2.0, "cdan"),
    (0.5, "iwdan"),
    (0.5, "iwcdan"),
    (0.5, "cdan"),
    (0.5, "iwdan_oracle"),
)


def _spec(r, rt, lam, m, s):
    oracle = m == "iwdan_oracle"
    extra = {"method": m}
    if oracle:
        extra.update(oracle=True, label="oracle, not deployable")
    return {
        "item": "r6_r2",
        "out_root": str(R.RES / "r6" / "r2" / rt),
        "ckpt_root": str(R.CKPT / "r6_r2" / rt),
        "tag": "r2_{}_{}_ladv{}_s{}".format(rt, m, R.lam_tag(lam), s),
        "recipe": r,
        "lam": float(lam),
        "seed": int(s),
        "mode": R.default_mode(r),
        "align": m,
        "eval": "full",
        "iw_cls": True,
        "extra": extra,
    }


def jobs(recipe_name):
    r = R.resolve_recipe(recipe_name)
    rt = R.recipe_tag(r)
    out = []
    for lam, m in ORDER:
        for s in R.SEEDS:
            out.append(_spec(r, rt, lam, m, s))
    return out


def main():
    p = R.common_args(argparse.ArgumentParser())
    p.add_argument("--recipe", default="current", help="current | p8 | auto")
    a = p.parse_args()
    js = jobs(a.recipe)
    if a.list:
        return R.print_jobs(js)
    R.run(js[a.task_id], a)


if __name__ == "__main__":
    main()
