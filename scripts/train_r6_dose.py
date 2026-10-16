#!/usr/bin/env python3
"""Paper B R6 R1: causal dose test (decisions/paperB_r6_r1.md).

Classifier loss on the full ISIC train set (stream 1, natural order). Only the ISIC images feeding the adversarial and
context-domain branches (stream 2) are resampled, to a class mix at shift 100% (ISIC train), 50% or 0% (the class mix of
pad_adv, the PAD images the adversary sees). PAD stream unchanged. PAD labels are used only to set the proportions.
31 jobs: λ = 0 once per seed (reused checkpoint of the same recipe, evaluation only), λ ∈ {0.5, 2} x shift {100, 0}
x 5 seeds, λ ∈ {0.5, 2} x shift 50 x 3 seeds (user decisions 2026-10-15).
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

SHIFTS = {100: 1.0, 50: 0.5, 0: 0.0}
# Remaining-work order: λ=2 at 0% and 100% (5 seeds), then shift 50% and λ=0.5, then λ=0 eval.
ORDER = ((2.0, 0), (2.0, 100), (2.0, 50), (0.5, 0), (0.5, 100), (0.5, 50), (0.0, 100))


def jobs(recipe_name):
    r = R.resolve_recipe(recipe_name)
    rt = R.recipe_tag(r)
    out = []
    for lam, sh in ORDER:
        for s in (R.SEEDS[:3] if sh == 50 else R.SEEDS):
            reuse = str(R.lambda0_ckpt_dir(r, s)) if lam == 0.0 else None
            out.append({"item": "r6_r1", "out_root": str(R.RES / "r6" / "r1" / rt), "ckpt_root": str(R.CKPT / "r6_r1" / rt),
                        "tag": "r1_{}_shift{}_ladv{}_s{}".format(rt, sh, R.lam_tag(lam), s), "recipe": r, "lam": lam,
                        "seed": s, "mode": "three", "align": "dann", "adv_shift": SHIFTS[sh], "eval": "full",
                        "reuse_ckpt": reuse, "extra": {"shift_pct": sh, "lambda0_reused": reuse is not None}})
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
