#!/usr/bin/env python3
"""Paper B R5 P7(b): PAD fully held out of training (decisions/paperB_r5_p7b.md).

Adversarial (and context-domain) domains = ISIC sites BCN (0) vs HAM (1), both drawn from the lesion-level ISIC train
split; site from the ISIC 2019 lesion_id prefix (the master CSV has no site column). Classifier on the full ISIC train
split. No PAD image enters training, BatchNorm statistics or checkpoint selection. OOD = all of PAD (2,298 images),
plus pad_heldout, Fitzpatrick17k and the P7a sets. λ ∈ {0, 0.5, 2} x 5 seeds = 15 runs. λ = 0 is trained inside this design
(user decision 2026-10-15): the same-recipe λ = 0 checkpoint saw PAD as the second domain (BatchNorm statistics, context
branch), so --reuse_l0 (evaluate that checkpoint instead) is not PAD-free.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

LAMS = (2.0, 0.0, 0.5)


def jobs(recipe_name, reuse_l0=False):
    r = R.resolve_recipe(recipe_name)
    rt = R.recipe_tag(r)
    out = []
    for lam in LAMS:
        for s in R.SEEDS:
            reuse = str(R.lambda0_ckpt_dir(r, s)) if (lam == 0.0 and reuse_l0) else None
            out.append({"item": "p7b", "out_root": str(R.RES / "r5" / "p7b" / rt), "ckpt_root": str(R.CKPT / "r5_p7b" / rt),
                        "tag": "p7b_{}_ladv{}_s{}".format(rt, R.lam_tag(lam), s), "recipe": r, "lam": lam, "seed": s,
                        "mode": "p7b", "align": "dann", "eval": "full", "reuse_ckpt": reuse,
                        "extra": {"lambda0_reused": reuse is not None}})
    return out


def main():
    p = R.common_args(argparse.ArgumentParser())
    p.add_argument("--recipe", default="current", help="current | p8 | auto")
    p.add_argument("--reuse_l0", action="store_true",
                   help="evaluate the same-recipe λ = 0 checkpoint instead of training λ = 0 (not PAD-free)")
    a = p.parse_args()
    js = jobs(a.recipe, reuse_l0=a.reuse_l0)
    if a.list:
        return R.print_jobs(js)
    R.run(js[a.task_id], a)


if __name__ == "__main__":
    main()
