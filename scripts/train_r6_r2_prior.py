#!/usr/bin/env python3
"""Paper B R2-prior: IWDAN with EM / 50-label priors — decisions/paperB_r6_r2_prior.md.

Arms λ=2 × 5 seeds: (i) iwdan_em — Saerens EM on unlabeled pad_adv after ISIC-val temperature scaling;
(ii) iwdan_n50 — fixed mix from 50 labelled pad_adv images (few-label; not unsupervised DA).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

LAM = 2.0
ARMS = ("iwdan_em", "iwdan_n50")
N50 = 50
N50_BASE_SEED = 20261017


def n50_indices(n_pad, seed):
    rng = np.random.default_rng(N50_BASE_SEED + int(seed))
    return rng.choice(n_pad, size=min(N50, n_pad), replace=False).astype(int)


def _spec(r, rt, arm, s):
    return {
        "item": "r6_r2_prior",
        "out_root": str(R.RES / "r6" / "r2_prior" / rt),
        "ckpt_root": str(R.CKPT / "r6_r2_prior" / rt),
        "tag": "r2p_{}_{}_ladv{}_s{}".format(rt, arm, R.lam_tag(LAM), s),
        "recipe": r,
        "lam": float(LAM),
        "seed": int(s),
        "mode": R.default_mode(r),
        "align": arm,
        "eval": "full",
        "iw_cls": True,
        "extra": {
            "method": arm,
            "setting": "few-label target" if arm == "iwdan_n50" else "unsupervised EM prior (Saerens 2002)",
            "temperature_scaled_before_em": arm == "iwdan_em",
        },
    }


def jobs(recipe_name):
    r = R.resolve_recipe(recipe_name)
    rt = R.recipe_tag(r)
    out = []
    for arm in ARMS:
        for s in R.SEEDS:
            out.append(_spec(r, rt, arm, s))
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
