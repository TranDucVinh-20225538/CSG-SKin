# Paper B R7b T1c: higher backbone LR

Design: `decisions/paperB_r7b_t1c.md`. Commit `91cdd4a`.

## Reading-rule case (computed first)

T1c stopped after selection: dual λ=0 chose backbone LR × 0.1. **T1 is the result.** No new Case A/B/C from T1c main runs.

See `SELECTION.md` / `SELECTION.json`. T1 stays in the paper.


# Paper B R7b T1c selection

Dual chosen × 0.1 (val bal 0.4444). ERM chosen × 0.1 (val bal 0.5157).

Status: **stop_after_selection**.

| arm | × | val bal | ID bal (test, not used) |
|---|---:|---:|---:|
| dual λ=0 | 0.1 | 0.4444 | 0.4449 |
| dual λ=0 | 0.3 | 0.4025 | 0.4157 |
| dual λ=0 | 1 | 0.2358 | 0.2251 |
| ERM | 0.1 | 0.5157 | 0.5877 |
| ERM | 0.3 | 0.5042 | 0.5085 |
| ERM | 1 | 0.4180 | 0.4249 |

