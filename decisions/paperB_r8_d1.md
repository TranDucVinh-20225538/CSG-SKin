# Paper B R8 / D1 — single-encoder invariance screen (seed 42)

Committed before any D1 job runs. Fixed now; not changed after results are seen. Deviations are appended at the end.

## Question

W1: no single-encoder model has reached invariance so far (P9 EfficientNet-B3: V1 0.699 after the toolbox; R7 T1b DINOv2-S: V1 0.884). R8 is a last attempt with a pre-committed stop: if nothing passes D1, the paper keeps "not tested".

## Design

- Blind to OOD: D1 jobs compute held-out leakage V1, ID balanced accuracy (val and test) and the degraded flag only. Do not compute any OOD score in D1. Save features for D2.
- Seed 42 only. Lesion-disjoint ISIC split (seed 42), pad_adv / pad_heldout, last.ckpt. Held-out leakage probe V1 as Table 1 / P6 (`r5_p9v2.heldout_probe` on BN-pooled `z`).
- GPU cap: **30 GPU-h** for D1. Stop early if the cap is reached; report what ran.
- Already done in P9 v2 (**do not repeat**): Ganin schedule with warm-up at λ ∈ {2, 4, 8}, adversary LR × {10, 30, 100}, head depth 1–3, one head, no inner steps.

### EfficientNet-B3 (base: P9 v2 BEST `gw_lr10_d2`: adversary LR × 10, head depth 2). 12 runs

(a) λ = 16 with the Ganin schedule (1); λ ∈ {4, 8, 16} with a fixed λ from step 0, no warm-up (3).
(b) head depth 3 with k ∈ {3, 5} adversary steps per encoder step on `z.detach()`, λ = 8, Ganin schedule (2).
(c) GRL replaced by MK-MMD or scaled CORAL as defined in `decisions/paperB_r5_p11.md`, on the same feature the domain head sees, weights w ∈ {3, 10, 30} (6).

### DINOv2-S/14 (base: R7 T1b recipe, backbone LR × 0.1). 15 runs

(a) λ ∈ {4, 8, 16} with the Ganin schedule (3) and with a fixed λ (3).
(b) head depth 3 with k ∈ {1, 3, 5}, λ = 8, Ganin schedule (3).
(c) MK-MMD and scaled CORAL, w ∈ {3, 10, 30} (6).

## Pass criterion (per run; locked before the first run)

held-out leakage V1 ≤ 0.60 and ID test balanced accuracy ≥ ERM reference − 0.05 (EfficientNet-B3 single ERM 0.558 → ≥ 0.508; DINOv2-S ERM 0.565 → ≥ 0.515) and not degraded.

Degraded = ID test balanced accuracy < 0.25 (Table 1 protocol flag).

## Selection (per backbone; locked before the first run)

Among passing runs, GRL configurations first (they are the standard DANN that W1 is about); within a family, the lowest V1; ties within 0.01 → smaller λ or weight, then smaller k. Written to `SELECTED_<backbone>.json` before any D2 run. If only MK-MMD / CORAL pass, select that and state in this decisions file (deviation) that D2 then tests single-encoder alignment, not DANN.

## Stop rule

No run passes for a backbone → no D2 for that backbone; write "W1 not addressed by R8 for \<backbone\>" in REPORT.md. Stop D1 early if the cap (30 GPU-h) is reached; report what ran.

## Outputs

`results/paperB/r8/d1/runs/<tag>/`, `results/paperB/r8/d1/{summary.csv,SELECTED_effb3.json,SELECTED_dinov2_vits14.json,REPORT.md}`.

## Deviations
