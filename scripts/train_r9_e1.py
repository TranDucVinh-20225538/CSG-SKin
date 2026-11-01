#!/usr/bin/env python3
"""Paper B R9 E1: six-class control (decisions/paperB_r9_e1.md).

Three-stream current recipe as R1, with ISIC adversary mix π_6cls = π_ISIC with DF and VASC
set to 0 and renormalised. Classifier stream unchanged (8 classes). PAD labels unused.
λ ∈ {0.5, 2} × seeds 42/52/62/72/82 = 10 runs.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

# Work-order reference mix (rounded); runtime mix is computed from the lesion ISIC train split.
PI_6CLS_REF = [0.182, 0.518, 0.134, 0.035, 0.106, 0.0, 0.0, 0.025]
LAMS = (2.0, 0.5)  # λ=2 first (decides reading)


def pi_6cls_from_isic(isic_mix):
    m = np.asarray(isic_mix, float).copy()
    m[5] = 0.0  # DF
    m[6] = 0.0  # VASC
    return (m / m.sum()).tolist()


def jobs():
    r = R.resolve_recipe("current")
    # Mix is filled at run time from live ISIC train; placeholder here for --list / tags.
    out = []
    for lam in LAMS:
        for s in R.SEEDS:
            out.append({
                "item": "r9_e1",
                "out_root": str(R.RES / "r9" / "e1"),
                "ckpt_root": str(R.CKPT / "r9_e1"),
                "tag": "e1_6cls_ladv{}_s{}".format(R.lam_tag(lam), s),
                "recipe": r,
                "lam": lam,
                "seed": s,
                "mode": "three",
                "align": "dann",
                "adv_mix": PI_6CLS_REF,  # overwritten in run_e1 from live ISIC mix
                "eval": "full",
                "reuse_ckpt": None,
                "extra": {"mix": "pi_6cls", "pi_6cls_ref": PI_6CLS_REF},
            })
    return out


def run_e1(spec, args):
    data = R.load_data()
    isic = R.class_mix(data["train"].label_idx)
    pad = R.class_mix(data["pad_adv"].label_idx)
    pi = pi_6cls_from_isic(isic)
    tv_isic = float(0.5 * np.abs(np.asarray(pi) - isic).sum())
    tv_pad = float(0.5 * np.abs(np.asarray(pi) - pad).sum())
    spec = dict(spec)
    spec["adv_mix"] = pi
    spec["adv_shift"] = None  # do not rebuild via dose_mix
    extra = dict(spec.get("extra") or {})
    extra.update({
        "pi_6cls": pi,
        "pi_6cls_ref": PI_6CLS_REF,
        "tv_to_pi_isic": tv_isic,
        "tv_to_pi_pad": tv_pad,
        "isic_train_mix": isic.tolist(),
        "pad_labels_used": False,
        "note": "PAD labels not used at all in E1; π_6cls from ISIC train only",
    })
    spec["extra"] = extra
    R.run(spec, args)


def main():
    p = R.common_args(argparse.ArgumentParser())
    a = p.parse_args()
    js = jobs()
    if a.list:
        return R.print_jobs(js)
    run_e1(js[a.task_id], a)


if __name__ == "__main__":
    main()
