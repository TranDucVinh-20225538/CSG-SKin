# Paper B R9 E1: six-class control

Design: `decisions/paperB_r9_e1.md`. Commit `dba39c7`.

## Reading-rule case (λ = 2 decides)

**reported as found; it settles neither reading.**

Detail: recovery (labelled) = {'fitzpatrick17k': 0.063, 'mednode': 0.058, 'sd198': 0.037}; Rule A (six-class) = {'fitzpatrick17k': None, 'mednode': None, 'sd198': None}.

Degraded excluded from means (as R1): R1 shift 0% λ=2 seed 52 (ID bal < 0.25; forced in report because the run summary omitted the flag).

π_6cls (run config): [0.182, 0.518, 0.134, 0.035, 0.106, 0.0, 0.0, 0.025]; TV→ISIC=0.019, TV→PAD=0.579 (ref 0.019 / 0.579).

## Table (Mahalanobis mean ± sd; paired Δ six−ref with seed-bootstrap 95% CI; recovery; Rule A on six-class)

| λ | set | six-class | shift 100% | shift 0% | Δ vs 100% | Δ vs 0% | recovery | Rule A |
|---:|---|---:|---:|---:|---|---|---:|---|
| 2 | pad_heldout (not counted) | 0.372 ± 0.034 | 0.368 ± 0.017 | 0.458 ± 0.003 | +0.005 [-0.029, +0.037] | -0.080 [-0.114, -0.057] | 0.050 | — |
| 2 | fitzpatrick17k | 0.362 ± 0.032 | 0.340 ± 0.065 | 0.691 ± 0.031 | +0.022 [-0.041, +0.086] | -0.337 [-0.378, -0.296] | 0.063 | — |
| 2 | mednode | 0.312 ± 0.050 | 0.298 ± 0.032 | 0.541 ± 0.027 | +0.014 [-0.022, +0.047] | -0.231 [-0.290, -0.175] | 0.058 | — |
| 2 | sd198 | 0.339 ± 0.039 | 0.325 ± 0.069 | 0.694 ± 0.031 | +0.014 [-0.055, +0.089] | -0.358 [-0.402, -0.313] | 0.037 | — |
| 2 | scin (not counted) | 0.496 ± 0.045 | 0.519 ± 0.073 | 0.732 ± 0.035 | -0.023 [-0.071, +0.028] | -0.236 [-0.288, -0.183] | -0.108 | — |
| 0.5 | pad_heldout (not counted) | 0.374 ± 0.037 | 0.361 ± 0.029 | 0.498 ± 0.025 | +0.014 [-0.015, +0.044] | -0.124 [-0.159, -0.086] | 0.101 | — |
| 0.5 | fitzpatrick17k | 0.394 ± 0.043 | 0.385 ± 0.023 | 0.672 ± 0.046 | +0.009 [-0.010, +0.026] | -0.278 [-0.291, -0.265] | 0.031 | — |
| 0.5 | mednode | 0.332 ± 0.036 | 0.310 ± 0.055 | 0.525 ± 0.050 | +0.022 [-0.016, +0.059] | -0.192 [-0.217, -0.172] | 0.103 | — |
| 0.5 | sd198 | 0.384 ± 0.051 | 0.365 ± 0.029 | 0.664 ± 0.058 | +0.019 [+0.000, +0.038] | -0.280 [-0.301, -0.257] | 0.063 | — |
| 0.5 | scin (not counted) | 0.557 ± 0.034 | 0.534 ± 0.015 | 0.718 ± 0.036 | +0.022 [-0.008, +0.051] | -0.162 [-0.176, -0.148] | 0.121 | — |

## ID / leakage (six-class)

- λ=2.0: ID bal 0.467 ± 0.010, leakage V1 0.679 ± 0.064, NV-nearest pad 0.618 ± 0.035, degraded seeds [].
- λ=0.5: ID bal 0.470 ± 0.010, leakage V1 0.604 ± 0.065, NV-nearest pad 0.609 ± 0.056, degraded seeds [].

