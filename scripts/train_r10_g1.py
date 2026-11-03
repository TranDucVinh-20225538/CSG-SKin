#!/usr/bin/env python3
"""Paper B R10 G1: R1 shift 50% seeds 72 and 82 (decisions/paperB_r10.md)."""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import train_r6_dose as dose  # noqa: E402


def main():
    p = R.common_args(argparse.ArgumentParser())
    p.add_argument("--recipe", default="current")
    a = p.parse_args()
    js = dose.r10_g1_jobs(a.recipe)
    if a.list:
        return R.print_jobs(js)
    R.run(js[a.task_id], a)


if __name__ == "__main__":
    main()
