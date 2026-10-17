# Paper B R6 R2 — interim: oracle λ = 2 vs IWDAN λ = 2 vs R1 DANN

Commit `94279f5`. Order deviation 2026-10-17 (oracle λ=2 advanced; criteria unchanged).

**Baselines (locked):** DANN = R1 shift 100% λ=2 `last.ckpt` (pad ≈ 0.368, Fitz ≈ 0.340, leakage ≈ 0.634); λ0 = R1 shift 100% λ=0 `last.ckpt`, **AUROC per unseen set**. Same recipe `current`, seeds `(42,52,62,72,82)`, lesion-level / Phase 2 split.

**Gap recovered (locked):** `(method−DANN)/(λ0−DANN)` Mahalanobis, paired by seed; always co-report absolute Δ. If mean `|λ0−DANN| < 0.05` → flag **small denom — prefer absolute Δ**.

Pre-committed fix reading **not** closed here — wait for CDAN and λ = 0.5.

## Per-seed Mahalanobis (point)

| seed | DANN pad | IWDAN pad | oracle pad | DANN Fitz | IWDAN Fitz | oracle Fitz |
|---:|---|---|---|---|---|---|
| 42 | 0.367 | 0.503 | 0.689 | 0.359 | 0.421 | 0.675 |
| 52 | 0.380 | 0.496 | 0.662 | 0.379 | 0.478 | 0.777 |
| 62 | 0.369 | 0.524 | 0.607 | 0.411 | 0.506 | 0.674 |
| 72 | 0.383 | 0.527 | 0.625 | 0.304 | 0.635 | 0.743 |
| 82 | 0.340 | 0.517 | 0.650 | 0.245 | 0.499 | 0.714 |

## Gap recovered and paired Δ vs DANN (per unseen set)

Absolute Δ is primary when the denom is small; ratio is still printed for every set.

| method | set | Δ vs DANN [cluster CI] | gap recovered (method−DANN)/(λ0−DANN) [seed boot CI · Δabs · denom] |
|---|---|---|---|
| IWDAN | pad_heldout | +0.145 ± 0.023 [0.129, 0.162] (n=5) | +0.319 ± 0.048 [0.278, 0.353] (n=5) · Δabs=+0.145 · mean(λ0−DANN)=+0.457 |
| IWDAN | fitzpatrick17k | +0.168 ± 0.117 [0.161, 0.175] (n=5) | +0.426 ± 0.240 [0.255, 0.615] (n=5) · Δabs=+0.168 · mean(λ0−DANN)=+0.372 |
| IWDAN | mednode | +0.052 ± 0.031 [0.028, 0.074] (n=5) | +0.195 ± 0.122 [0.087, 0.281] (n=5) · Δabs=+0.052 · mean(λ0−DANN)=+0.261 |
| IWDAN | sd198 | +0.163 ± 0.117 [0.157, 0.168] (n=5) | +0.397 ± 0.238 [0.220, 0.573] (n=5) · Δabs=+0.163 · mean(λ0−DANN)=+0.383 |
| IWDAN_ORACLE | pad_heldout | +0.279 ± 0.038 [0.256, 0.303] (n=5) | +0.611 ± 0.090 [0.546, 0.689] (n=5) · Δabs=+0.279 · mean(λ0−DANN)=+0.457 |
| IWDAN_ORACLE | fitzpatrick17k | +0.377 ± 0.086 [0.366, 0.388] (n=5) | +1.010 ± 0.138 [0.913, 1.118] (n=5) · Δabs=+0.377 · mean(λ0−DANN)=+0.372 |
| IWDAN_ORACLE | mednode | +0.265 ± 0.065 [0.234, 0.294] (n=5) | +1.012 ± 0.224 [0.838, 1.185] (n=5) · Δabs=+0.265 · mean(λ0−DANN)=+0.261 |
| IWDAN_ORACLE | sd198 | +0.368 ± 0.096 [0.358, 0.378] (n=5) | +0.952 ± 0.119 [0.863, 1.047] (n=5) · Δabs=+0.368 · mean(λ0−DANN)=+0.383 |

## Cell means (Mahalanobis)

- **pad_heldout:** DANN 0.368±0.017; IWDAN 0.513±0.014; oracle 0.647±0.032
- **fitzpatrick17k:** DANN 0.340±0.065; IWDAN 0.508±0.078; oracle 0.717±0.044
- **mednode:** DANN 0.298±0.032; IWDAN 0.350±0.046; oracle 0.563±0.050
- **sd198:** DANN 0.325±0.069; IWDAN 0.488±0.075; oracle 0.693±0.052

**On origin/paperb/reviewer-r1:** `7fcfb1a`.

