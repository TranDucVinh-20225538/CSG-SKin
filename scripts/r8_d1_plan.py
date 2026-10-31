#!/usr/bin/env python3
"""Paper B R8 D1 task plan (decisions/paperB_r8_d1.md). Prints: backbone config lambda_adv penalty penalty_weight.

27 tasks (0..26): EffB3 0..11, DINOv2 12..26. Seed always 42.
"""
from __future__ import annotations

import sys

# (backbone, config, lambda_adv, penalty, penalty_weight)
TASKS = []
# EffB3 base gw_lr10_d2
TASKS.append(("effb3", "gw_lr10_d2", 16.0, "none", 0.0))           # E1 λ=16 Ganin
for lam in (4.0, 8.0, 16.0):
    TASKS.append(("effb3", "fix_lr10_d2", lam, "none", 0.0))         # E2–E4 fixed
for k in (3, 5):
    TASKS.append(("effb3", "gw_lr10_d3_k{}".format(k), 8.0, "none", 0.0))  # E5–E6
for pen in ("mkmmd", "coral"):
    for w in (3.0, 10.0, 30.0):
        TASKS.append(("effb3", "baseline", 0.0, pen, w))            # E7–E12
# DINOv2 T1b base: ganin uses gw_lr30_d2 (adv×30, depth 2); backbone_lr_mult set by launcher
for lam in (4.0, 8.0, 16.0):
    TASKS.append(("dinov2_vits14", "gw_lr30_d2", lam, "none", 0.0))  # D1–D3 Ganin
for lam in (4.0, 8.0, 16.0):
    TASKS.append(("dinov2_vits14", "fix_lr30_d2", lam, "none", 0.0))  # D4–D6 fixed
for k in (1, 3, 5):
    TASKS.append(("dinov2_vits14", "gw_lr30_d3_k{}".format(k), 8.0, "none", 0.0))  # D7–D9
for pen in ("mkmmd", "coral"):
    for w in (3.0, 10.0, 30.0):
        TASKS.append(("dinov2_vits14", "baseline", 0.0, pen, w))    # D10–D15

assert len(TASKS) == 27, len(TASKS)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    if t < 0:
        for i, row in enumerate(TASKS):
            print(i, *row)
        return
    bb, cfg, lam, pen, w = TASKS[t]
    print(bb, cfg, lam, pen, w)


if __name__ == "__main__":
    main()
