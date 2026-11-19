# REPORT — Paper B R14 ISIC 2019 → MILK10k dermoscopic

Generated: 2026-11-19T04:45:05.254484+00:00
Commit: `0c49904`

## Stage-1 stop (honest)

The stage-1 STOP-degenerate rule triggered (λ0 prior100 s42 first attempt ID bal 0.185 < 0.25; also λ2 prior0 s42 first attempt 0.152). R14 continued only under the 2026-11-18 deviation after a CPU diagnosis and a determinism redo. Thresholds unchanged. Stage 1 did **not** pass.

## Determinism

Seeds are not bitwise reproducible (GPU non-determinism).

First-attempt degenerate count over the whole 20-run grid: **2 of 20** (`r14_ladv0_prior100_s42`, `r14_ladv2_prior0_s42`). Both diverged (`loss_ctx`/`loss_cls` rose across epochs; see DIAG_cpu_20261118.md). No other first attempt in the grid had ID bal < 0.25.

### Rerun rule (ID only)

| tag | first ID | rerun ID | status |
|---|---:|---:|---|
| r14_ladv0_prior100_s42 | 0.185 | 0.475 | first: degenerate, superseded; rerun kept |
| r14_ladv2_prior0_s42 | 0.152 | 0.480 | first: degenerate, superseded; rerun kept |

Do not rerun any run for any reason other than ID bal < 0.25.

## Reading (locked order)

### (a) V1 at λ = 0

V1 at λ = 0 (seed mean) = **0.914** > 0.60 (n=5). Not a small-gap case.

### (b) Invariance at λ = 2 prior100

V1 seed mean = **0.604** > 0.60 → **Case C**, settles nothing. No reading of AUROC / Rule A for the verdict.

### (c) V1 at λ = 2 prior0 (label-shift-necessary clause)

V1 prior0 seed mean = **0.738** (n=5). — **not read (Case C at prior100)**

### (d) Rule A

**Not read (Case C at prior100).** Descriptive table only:

## Verdict

Not invariant (V1=0.604 > 0.60 at λ=2 prior100). Case C, settles nothing.

```json
{
  "small_gap": false,
  "invariant_prior100": false,
  "competent": true,
  "case": "C",
  "text": "Not invariant (V1=0.604 > 0.60 at \u03bb=2 prior100). Case C, settles nothing.",
  "id_bal_vs_erm": {
    "cell": 0.47357630238103815,
    "erm": 0.558,
    "delta": -0.0844236976189619
  }
}
```

## Cells (Mahalanobis / V1)

| cell | ID bal | V1 | n | Fitz Rule A | MED Rule A | SD Rule A | note |
|---|---:|---:|---:|---|---|---|---|
| ladv0_prior100 | 0.472±0.014 | 0.914±0.022 | 5 | not | not | not | for (a) + competence only |
| ladv2_prior100 | 0.474±0.025 | 0.604±0.048 | 5 | met | not | met | decides invariance / Case C |
| ladv2_prior50 | 0.494±0.016 | 0.578±0.020 | 5 | met | not | met | not read (Case C at prior100) |
| ladv2_prior0 | 0.482±0.011 | 0.738±0.042 | 5 | not | not | not | not read (Case C at prior100) |

## Per-run rows

| tag | status | ID bal | V1 | Fitz Maha |
|---|---|---:|---:|---:|
| r14_ladv0_prior100_s42 | ok | 0.475 | 0.939 | 0.855 |
| r14_ladv0_prior100_s52 | ok | 0.464 | 0.931 | 0.843 |
| r14_ladv0_prior100_s62 | ok | 0.453 | 0.888 | 0.816 |
| r14_ladv0_prior100_s72 | ok | 0.482 | 0.897 | 0.857 |
| r14_ladv0_prior100_s82 | ok | 0.487 | 0.913 | 0.833 |
| r14_ladv2_prior0_s42 | ok | 0.480 | 0.772 | 0.725 |
| r14_ladv2_prior0_s52 | ok | 0.495 | 0.742 | 0.742 |
| r14_ladv2_prior0_s62 | ok | 0.484 | 0.783 | 0.714 |
| r14_ladv2_prior0_s72 | ok | 0.465 | 0.682 | 0.735 |
| r14_ladv2_prior0_s82 | ok | 0.486 | 0.712 | 0.714 |
| r14_ladv2_prior100_s42 | ok | 0.458 | 0.617 | 0.365 |
| r14_ladv2_prior100_s52 | ok | 0.453 | 0.615 | 0.393 |
| r14_ladv2_prior100_s62 | ok | 0.470 | 0.524 | 0.215 |
| r14_ladv2_prior100_s72 | ok | 0.471 | 0.654 | 0.347 |
| r14_ladv2_prior100_s82 | ok | 0.516 | 0.610 | 0.389 |
| r14_ladv2_prior50_s42 | ok | 0.479 | 0.604 | 0.463 |
| r14_ladv2_prior50_s52 | ok | 0.497 | 0.576 | 0.340 |
| r14_ladv2_prior50_s62 | ok | 0.499 | 0.588 | 0.372 |
| r14_ladv2_prior50_s72 | ok | 0.477 | 0.551 | 0.393 |
| r14_ladv2_prior50_s82 | ok | 0.517 | 0.572 | 0.327 |

## Job / compute meta

```json
{
  "commit": "0c49904",
  "n_runs_scored": 20,
  "generated_utc": "2026-11-19T04:45:05.254452+00:00"
}
```

milk_heldout, PAD-UFES-20 and SCIN reported, not counted. prior50 = dose step (shown above; not read under Case C).
No wording beyond reading rules in decisions/paperB_r14_pair3.md.

