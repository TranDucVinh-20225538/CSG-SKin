#!/usr/bin/env python3
"""Packing test: one vs two training runs per GPU (decisions/paperB_r5_pack_test.md).

One P8 grid configuration (A1, λ = 0), shortened to --epochs, written under results/paperB/r5/pack_test/<arm>/.
  --deterministic  Lightning deterministic mode (equality check: solo vs packed, same seed)
Wall time per run goes to timing.json next to summary.json. Called by slurm/r5_pack_test.sbatch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

OUT = R.RES / "r5" / "pack_test"
CID = "A1"


def main():
    p = R.common_args(argparse.ArgumentParser())
    p.add_argument("--arm", required=True, help="e.g. det_solo, det_packed, speed_solo, speed_packed")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--deterministic", action="store_true")
    a = p.parse_args()
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    r = R.grid_recipe(CID, a.epochs)
    spec = {"item": "pack_test", "out_root": str(OUT / a.arm), "ckpt_root": str(R.CKPT / "r5_pack_test" / a.arm),
            "tag": "{}_ladv0_s{}".format(r["name"], a.seed), "recipe": r, "lam": 0.0, "seed": a.seed,
            "mode": R.default_mode(r), "align": "dann", "eval": "val"}
    t0 = time.time()
    R.run(spec, a)
    d = OUT / a.arm / "runs" / spec["tag"]
    (d / "timing.json").write_text(json.dumps({"wall_s": time.time() - t0, "epochs": a.epochs,
                                               "deterministic": a.deterministic, "host": os.uname().nodename}) + "\n")


if __name__ == "__main__":
    main()
