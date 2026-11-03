# Paper B R10 C0 — degraded-run audit

Rule: ID test balanced accuracy < 0.25: report the run, exclude from every mean / CI / probe / regression.

## Class-aware held-out probes

Excluded n=1: r1_shift0_l2 s52
R1 shift0% λ=2 cell means after exclusion (n=4): (a) 0.630, (b) 0.522, (c) 0.555 (work-order target without s52: 0.630 / 0.522 / 0.555).
Excluded seeds in that cell: [52].

| file | cell | run | status |
|---|---|---|---|
| `r6/class_aware_heldout/` | r1_shift0_l2 | r1_cur_shift0_ladv2_s52 | fixed |

## R9 E1 shift references

OUTCOME force_degraded / excluded: `{"force_degraded": [{"arm": "s0", "lambda": 2.0, "seed": 52}], "excluded_degraded": {"shift 0%|2": [52]}}`.
| file | cell | run | status |
|---|---|---|---|
| `r9/e1/OUTCOME.json` | shift 0% λ=2 ref | seed 52 | fixed (forced exclude) |

## All degraded summaries under results/paperB/r{{5–9}}

| file | tag | seed | ID bal | degraded flag in summary |
|---|---|---:|---:|---|
| `r5/p11/runs/dual_effb3_ladv0_s42_coral30/summary.json` | dual_effb3_ladv0_s42_coral30 | 42 | 0.183 | None |
| `r5/p9/v2/single_effb3_gw_lr10_d1_ladv2_s42/summary.json` | single_effb3_gw_lr10_d1_ladv2_s42 | 42 | 0.236 | None |
| `r5/p9/v2/single_effb3_gw_lr10_d1_ladv4_s42/summary.json` | single_effb3_gw_lr10_d1_ladv4_s42 | 42 | 0.125 | None |
| `r5/p9/v2/single_effb3_gw_lr30_d1_ladv8_s42/summary.json` | single_effb3_gw_lr30_d1_ladv8_s42 | 42 | 0.125 | None |
| `r6/r1/cur/runs/r1_cur_shift0_ladv2_s52/summary.json` | r1_cur_shift0_ladv2_s52 | 52 | 0.139 | True |
| `r7b/t1c/runs/r7b_t1c_dual_dinov2_vits14_ladv0_s42_bb1/summary.json` | r7b_t1c_dual_dinov2_vits14_ladv0_s42_bb1 | 42 | 0.225 | None |
| `r8/d1/runs/r8_d1_single_dinov2_vits14_baseline_ladv0_s42_bb0p1_coral10/summary.json` | r8_d1_single_dinov2_vits14_baseline_ladv0_s42_bb0p1_coral10 | 42 | 0.125 | None |
| `r8/d1/runs/r8_d1_single_dinov2_vits14_baseline_ladv0_s42_bb0p1_coral30/summary.json` | r8_d1_single_dinov2_vits14_baseline_ladv0_s42_bb0p1_coral30 | 42 | 0.125 | None |

## Notes

- R1 report (`r6_report.py --item r1`) already excludes degraded from primary Rule A.
- R10 regenerates class-aware probes with exclusion; audit lists remaining degraded summaries on disk.
- Paper number stores are rebuilt only when their source reports are regenerated (G1 follow-ups).

