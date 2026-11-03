#!/usr/bin/env python3
"""Paper B R10 G3: IWDAN n50 at λ=0.5 × seeds 42–82 (decisions/paperB_r10.md).

Same 50-image draw per seed as the λ=2 n50 arm (train_r6_r2_prior.n50_indices).
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import train_r6_r2_prior as P  # noqa: E402

LAM = 0.5


def jobs(recipe_name="current"):
    r = R.resolve_recipe(recipe_name)
    rt = R.recipe_tag(r)
    out = []
    for s in R.SEEDS:
        spec = P._spec(r, rt, "iwdan_n50", s)
        spec["lam"] = float(LAM)
        spec["tag"] = "r2p_{}_iwdan_n50_ladv{}_s{}".format(rt, R.lam_tag(LAM), s)
        extra = dict(spec.get("extra") or {})
        extra.update({"r10_item": "G3", "lambda_note": "R10 G3: n50 priors at λ=0.5; same draw as λ=2"})
        spec["extra"] = extra
        out.append(spec)
    return out


def main():
    p = R.common_args(argparse.ArgumentParser())
    p.add_argument("--recipe", default="current")
    a = p.parse_args()
    js = jobs(a.recipe)
    if a.list:
        return R.print_jobs(js)
    R.run(js[a.task_id], a)


if __name__ == "__main__":
    main()
