#!/usr/bin/env python3
"""Paper B R6 R1-hi: high-λ R1 at shift 0% (decisions/paperB_r6_r1_hi.md).

RECIPE=current, three-stream, last.ckpt. Main wave: λ ∈ {4, 8} × 5 seeds.
Fallback wave (submitted only if λ=8 gate empty): λ=4 with adv_lr_multiplier ×3.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

SHIFT_PCT = 0
LAMS_MAIN = (4.0, 8.0)


def _spec(recipe, rt, lam, seed, adv_lr_multiplier=30.0, wave="main"):
    tag = "r1hi_{}_shift{}_ladv{}_s{}".format(rt, SHIFT_PCT, R.lam_tag(lam), seed)
    if abs(float(adv_lr_multiplier) - 30.0) > 1e-9:
        tag = "r1hi_{}_shift{}_ladv{}_advlr3_s{}".format(rt, SHIFT_PCT, R.lam_tag(lam), seed)
    return {
        "item": "r6_r1_hi",
        "out_root": str(R.RES / "r6" / "r1_hi" / rt),
        "ckpt_root": str(R.CKPT / "r6_r1_hi" / rt),
        "tag": tag,
        "recipe": recipe,
        "lam": float(lam),
        "seed": int(seed),
        "mode": "three",
        "align": "dann",
        "adv_shift": 0.0,
        "adv_lr_multiplier": float(adv_lr_multiplier),
        "eval": "full",
        "reuse_ckpt": None,
        "extra": {
            "shift_pct": SHIFT_PCT,
            "wave": wave,
            "adv_lr_multiplier": float(adv_lr_multiplier),
            "decisions": "paperB_r6_r1_hi.md",
        },
    }


def jobs_main(recipe_name="current"):
    r = R.resolve_recipe(recipe_name)
    rt = R.recipe_tag(r)
    assert rt == "cur", "R1-hi fixed to RECIPE=current (got {})".format(rt)
    out = []
    for lam in LAMS_MAIN:
        for s in R.SEEDS:
            out.append(_spec(r, rt, lam, s, 30.0, "main"))
    return out


def jobs_advlr3(recipe_name="current"):
    r = R.resolve_recipe(recipe_name)
    rt = R.recipe_tag(r)
    return [_spec(r, rt, 4.0, s, 90.0, "advlr3") for s in R.SEEDS]


def main():
    p = R.common_args(argparse.ArgumentParser())
    p.add_argument("--recipe", default="current", help="must be current for R1-hi")
    p.add_argument("--wave", choices=("main", "advlr3"), default="main")
    a = p.parse_args()
    if a.recipe not in ("current", "cur"):
        raise SystemExit("R1-hi requires RECIPE=current (got {!r})".format(a.recipe))
    js = jobs_main("current") if a.wave == "main" else jobs_advlr3("current")
    if a.list:
        return R.print_jobs(js)
    R.run(js[a.task_id], a)


if __name__ == "__main__":
    main()
