#!/usr/bin/env python3
"""Paper B R11b Y4: X4 BCN→HAM at stronger λ ∈ {4, 8}.

Writes only results/paperB/r11b/y4/. Reuses R11 X4 splits (read-only).
Never modifies results/paperB/r11/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAIN = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
WT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b")
sys.path.insert(0, str(WT / "scripts"))
sys.path.insert(0, str(MAIN / "scripts"))

import r11_x4_train as X4  # noqa: E402
import r6_common as R  # noqa: E402

OUT = WT / "results" / "paperB" / "r11b" / "y4"
CKPT = MAIN / "checkpoints" / "r11b_y4"
X4.SPLIT_PATH = MAIN / "results" / "paperB" / "r11" / "x4" / "split_seed42.json"
SCREEN_SEEDS = (42, 52)
SEL_PATH = OUT / "SELECTION.json"


def tag_for(lam, seed, shift):
    return "y4_ladv{}_shift{}_s{}".format(
        X4.lam_tag(lam), int(round(100 * float(shift))), int(seed))


def screen_jobs():
    out = []
    for lam in (4.0, 8.0):
        for seed in SCREEN_SEEDS:
            out.append({"lam": lam, "seed": seed, "shift": 1.0, "phase": "screen"})
    return out


def main_jobs(lam):
    out = []
    for seed in R.SEEDS:
        out.append({"lam": float(lam), "seed": seed, "shift": 1.0, "phase": "main"})
    for seed in R.SEEDS:
        out.append({"lam": float(lam), "seed": seed, "shift": 0.0, "phase": "main"})
    return out


def train_one(lam, seed, shift, phase="screen", skip_done=True, batch_size=32, num_workers=4):
    X4.OUT = OUT / phase
    X4.CKPT = CKPT / phase
    X4.OUT.mkdir(parents=True, exist_ok=True)
    X4.tag_for = tag_for
    X4.train_one(lam, seed, shift, skip_done=skip_done, batch_size=batch_size, num_workers=num_workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", type=int)
    ap.add_argument("--phase", choices=("screen", "main"), default="screen")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--selected_lam", type=float)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    a = ap.parse_args()

    if a.phase == "screen":
        js = screen_jobs()
    else:
        if a.selected_lam is None:
            a.selected_lam = float(json.loads(SEL_PATH.read_text())["lam"])
        js = main_jobs(a.selected_lam)

    if a.list:
        for i, j in enumerate(js):
            print(i, tag_for(j["lam"], j["seed"], j["shift"]), j)
        print("n_jobs", len(js))
        return
    if a.task_id is None:
        raise SystemExit("need --task_id")
    j = js[a.task_id]
    train_one(j["lam"], j["seed"], j["shift"], phase=j["phase"], skip_done=a.skip_done,
              batch_size=a.batch_size, num_workers=a.num_workers)


if __name__ == "__main__":
    main()
