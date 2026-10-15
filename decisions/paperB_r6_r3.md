# Paper B R6 / R3 — toy model with label shift

Committed before any R3 run (no R3 output exists at this commit). Fixed now; deviations appended.

## What the earlier synthetic reproduction contained (Phase 13.5 v1 / v2; read from the code and PREREGISTER_v2.json)

- World: D = 50, K = 8 Gaussian classes (means = 2 × orthonormal basis), domain shift δ = 2 along a direction u, τ = 1;
  third domain δ2 = 3 along u2. MLP encoder [128, 128] → m = 16, mlp_derm adversary, adversary LR ×30, 40 epochs,
  batch 256, AdamW 1e-3 / 1e-4, classifier on the source only, λ ∈ {0, …, 8}, seeds 0–4.
- Class priors: v1 source uniform over 8 classes, target uniform over 6 (`rng.choice(classes)`); v2 changed the source to
  6 equal common classes + 2 rare classes (n = 80 each) with heavy tails and 3 % contamination, target still uniform.
  **No class-prior shift among the shared classes and no majority class in either version** — only the two absent
  classes differ.
- OOD evaluation set = the adversary's own target sample (no held-out target images); GRL coefficient λ and adversary loss
  weight λ (gradient ∝ λ²).
- Outcome: no inversion with BN off (v2 BN-off Mahalanobis 0.693 → 0.786 with λ); v1's sub-chance with BN on was a
  BatchNorm-on-mixed-batches artefact present at λ = 0.

## Question

With everything else held at the locked v1 world, does a class-prior shift between the two training domains (majority
class ≈ 50 % in source, ≈ 12 % in target) produce the inversion, and does it grow with the size of the shift?

## Design (code: `scripts/r6_r3_toy_labelshift.py`, CPU)

- Source prior (ISIC-like): class 0 (majority) 0.50, then 0.18, 0.13, 0.10, 0.035, 0.025, 0.015, 0.015.
- Target prior over the 6 shared classes: q(s) = (1 − s)·q0 + s·q_PAD, q0 = source prior restricted to classes 0–5
  (majority 0.515), q_PAD = (0.12, 0.025, 0.36, 0.10, 0.315, 0.08) (PAD-UFES-like: majority 0.12, class 4 "AK-like"
  0.315). s ∈ {0, 0.25, 0.5, 0.75, 1}; TV on shared classes reported per s.
- Target images grouped into patients (≈ 1.7 images per patient, isotropic patient offset σ_p = 0.5); 2,900 target
  images; 30 % of patients held out: the adversary never sees them and they are the OOD evaluation set ("held-out target
  cluster", the pad_heldout analogue). Secondary OOD: a never-seen third domain (δ2 = 3 along u2, prior q(s)).
- GRL convention of the main model: reversal coefficient 1, adversary loss weighted by λ.
- λ ∈ {0, 0.25, 0.5, 1, 2, 4}; seeds 0–4.

Cells:
| cell | BN | priors | s | role |
|---|---|---|---|---|
| primary | off | ISIC-like / q(s) | 0, .25, .5, .75, 1 | the test |
| bn_on | on | ISIC-like / q(s) | 0, 1 | control (v1 confound) |
| uniform | off | v1 uniform / uniform | — | no-label-shift control (v1 world + held-out patients) |
| oracle | off | ISIC-like / q(s), adversary's source half-batch resampled to q(s) | 0.5, 1 | label-shift-corrected alignment |

Metrics per run: Mahalanobis AUROC (fit on source train; ID = source test; OOD = held-out target patients); domain-2
AUROC; MSP, Energy, cosine, kNN k=50; majority-centroid share of OOD, of ID test and of domain 2 (nearest class mean under
the same Mahalanobis metric); predicted-majority share on OOD; class-4 recall on OOD; AUROC with ID reweighted to the OOD
class mix; leakage probe (logistic, ID test vs OOD).

## Reported cells

Every (cell, s, λ): mean ± s.d. over 5 seeds and k/5 seeds below 0.5. Plain paragraph: minimum over λ of the mean
AUROC per s (primary), uniform control, oracle.

## PREDICTED OUTCOME (written now, as the work order states it)

- Primary (BN off): at s = 0 no inversion at any λ (minimum mean AUROC ≥ 0.5); inversion appears only with label shift
  and grows with it: the minimum over λ of the mean AUROC falls monotonically with s, below 0.5 for s ≥ 0.5, and
  0.35–0.45 at s = 1 (λ ≥ 1). The majority-centroid share of OOD rises with λ and with s (to ≥ 0.4 at s = 1, λ ≥ 1,
  against a target majority prior of 0.12), and class-4 recall on OOD falls toward 0.
- Uniform control: no inversion (≥ 0.5 at every λ), as in v1/v2 BN off.
- Oracle (alignment against the target class mix): no inversion at s = 0.5 or 1 (minimum ≥ 0.5).
- BN on: λ = 0 may already be below 0.5 (the v1 artefact); read only as a control.
My confidence that the primary prediction holds as written: ≈ 50 %. The alternative I consider likely: perfect marginal
alignment of a Gaussian toy gives AUROC → 0.5 from above ("collapse without inversion") at every s, because the toy
has no heavy-tailed, high-density NV mode for target points to be pulled into — the majority-centroid share would
still rise with s (label-shift-driven misalignment), but the AUROC would stay ≥ 0.47.

## Gates (flag, never drop)

λ = 0 primary AUROC < 0.6 at any s → flagged "no headroom" for that s. Leakage at λ = 0 < 0.8 → flagged (domain shift
too weak to align).

## Outputs

`results/paperB/r6/r3/`: `config.json`, `runs/*.json` (one per cell × s × λ × seed), `summary.csv`, `cells.csv`, `REPORT.md`.

## Cost

GPU 0. 300 runs (primary 150, bn_on 60, uniform 30, oracle 60); one run ≈ 2–4 min single-threaded (v1: 40 epochs of a
small MLP on 16k + 2k points) → 10–20 CPU-h, ≈ 0.5–1 h wall on 24 workers.

## Stop rules

No retuning after results: world, priors, s grid and λ grid fixed now. 6 h sbatch cap.

## Deviations
