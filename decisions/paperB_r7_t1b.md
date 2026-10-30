# Paper B R7 / T1b — single-encoder DANN on DINOv2-S/14

Committed before any T1b job runs. Fixed now; not changed after results are seen. Deviations are appended at the end. Run only after T1.

## Question

Addresses W1: a standard one-encoder DANN does not invert with EfficientNet-B3 / ResNet-50 (App. A; P9 never reached invariance, held-out leakage 0.70).

## Design

- Single encoder DINOv2-S/14 + GRL domain head on the pooled feature (standard DANN), λ = 2, seeds 42, 52, 62.
- Same backbone LR × 0.1 as T1.
- λ = 0 reference = the T1 ERM runs (same seeds). 3 runs at λ = 2.
- Baseline single-encoder recipe (`config=baseline`); **do not run the P9 ladder** (no budget).
- Outputs under `results/paperB/r7/t1b/` with tag prefix `r7_t1b`.
- Report held-out leakage V1.

## Data / protocol

Same as Table 1 (see `decisions/paperB_r7_t1.md`). Degraded flagged, not dropped.

## Reading rules for T1b (locked before the first run)

With 3 seeds, "inverts" requires all 3 seeds to have CI upper < 0.5 on that set (Rule A).

Invariance reached = held-out leakage V1 ≤ 0.60 at λ = 2.

Invariance reached and inverts on pad_heldout: W1 is answered for a standard DANN on this backbone.

Invariance reached and does not invert: a standard DANN at comparable invariance does not invert here; the result goes into the main text, and the title and scope must say the effect is shown for the dual encoder.

Invariance not reached: the test did not take place for the single encoder (as P9); W1 stays open.

If T1b cannot run (GPU time, failure), write in this decisions file: "W1 is not addressed by R7; the title and scope must match what the paper shows (dual encoder)." Report it in REPORT.md the same way.

## Outputs

`results/paperB/r7/t1b/runs/<tag>/`, `results/paperB/r7/t1b/{summary.csv,BOOTSTRAP.json,REPORT.md}`.

## Deviations

### Amendment 2026-10-30 (before the first T1b run)

Work-order update: T1b uses **5 seeds** (42, 52, 62, **72, 82**) at λ = 2; λ = 0 reference = the T1 ERM runs at the same five seeds. Original pending 3-seed array cancelled before any T1b task started; resubmitted as 5-seed array. Reading rules above are **not** edited. Rule A remains the absolute count (≥3 seeds with CI upper < 0.5, and pooled CI upper < 0.5); with 5 seeds that is not “all 5”.
