# Paper B R5 / P11 — moment matching instead of an adversary, compared with GRL at matched invariance

Committed before any P11 job runs. Fixed now; not changed after results are seen. Deviations are appended at the end.
This item absorbs Q2's CORAL arm (user decision); GroupDRO is dropped.

## Question

Is the inversion on pad_heldout specific to adversarial training, or does any objective that aligns the ISIC and PAD
marginals of `z_lesion_norm` to the same degree produce it? Earlier moment-matching attempts do not answer this:
the Phase 15 MMD (single EfficientNet-B3, image-level, fixed σ ∈ {1, …, 16} on 1,536-d features) reached leakage
≥ 0.836 and degraded the classifier, and the Phase 15b CORAL term (‖C_s − C_t‖²_F / 4d², no mean term) let the domain
means drift apart (CORAL anomaly). P11 uses properly scaled versions on the dual encoder's 16-d scoring space, tunes
them to the GRL runs' invariance (target leakage ≈ 0.60) with the classifier intact, and compares OOD behaviour at
matched leakage.

## Design (scripts/train_r5_dual_isic.py, `--penalty`)

Dual encoder EfficientNet-B3 (the Table 1 model), lesion-level split, λ_adv = 0 (adversary term off; context and
orthogonality terms unchanged), plus w · penalty(z_lesion_norm[ISIC rows], z_lesion_norm[PAD rows]) per batch
(32 + 32, domain-balanced):

- MK-MMD (Long et al. 2015): Gaussian kernels with bandwidths = median pairwise squared distance of the pooled batch
  (detached) × {0.25, 0.5, 1, 2, 4}; unbiased MMD² estimator; kernels averaged. Scale-free in the feature dimension.
- Scaled CORAL: (‖μ_s − μ_t‖² + ‖C_s − C_t‖²_F) / d, d = 16. The mean term closes the Phase 15b anomaly.

Stages (seed 42 first, all selection on validation-side numbers):

1. Control: no penalty, λ_adv = 0, seed 42 (the selection reference in this pipeline).
2. Scan: {MK-MMD, CORAL} × w ∈ {1, 3, 10, 30} at the recipe's epochs (40); longer schedule: {MK-MMD, CORAL} × w = 10 at
   80 epochs.
3. Selection (per penalty; `r5_p11_report.py --select`): among scan configs with "classifier intact" (ISIC val balanced
   accuracy ≥ 0.95 × the control's), the one whose selection probe (logistic probe ISIC val vs pad_adv on
   z_lesion_norm, training-side images only) is closest to 0.60; ties within 0.01 → smaller weight, then 40 epochs. If
   no config is intact: the one with the highest val balanced accuracy among those with selection probe ≤ 0.75,
   flagged "classifier not intact"; if none, the penalty is reported as "target not reached" and has no seed stage.
   No test, OOD or Table 1 number enters the selection.
4. Seeds: the selected config per penalty at seeds 52, 62, 72, 82 (seed 42 from the scan) → 5 seeds per penalty.

A single-encoder arm is not run here (P9 covers single encoders; the comparison of interest is with the dual-encoder
GRL runs of Table 1).

## Data / splits

Lesion-level ISIC split (train 16,211 / val 4,053 / test 5,067), PAD pad_adv (training) / pad_heldout (716 images / 412
patients), Fitzpatrick17k, P7a clinical sets when present.

## Seeds and grid

Scan: seed 42, w ∈ {1, 3, 10, 30} (40 epochs) and w = 10 (80 epochs), both penalties. Seeds stage: 42, 52, 62, 72, 82.

## Matched-invariance comparison with GRL

GRL comparators: the 30 published lesion-level GRL runs with λ > 0 (R2 item 4: λ ∈ {0.25, 0.5, 1, 2, 4, 8}, Table 1 /
R4). For each selected P11 cell with mean Table 1 leakage ℓ, the GRL runs whose leakage is within ±0.05 of ℓ form the
matched comparator (n, mean λ, pad_heldout and Fitzpatrick Mahalanobis, mean ± SD). Reported side by side; no test of
the difference is pre-registered (descriptive comparison, numbers only).

## OOD sets, detectors, metrics

Table 1 columns via `r5_common.eval_isic` (leakage = ISIC test vs PAD full probe; ID balanced accuracy; ID ECE; OOD ECE
pad_heldout; Mahalanobis pad_heldout; Fitzpatrick; X-dom six-class); kNN, cosine, MSP, Energy; NV-nearest fraction;
pad_heldout reweighted to the ISIC class mix (decomposition check); P7a sets.

## Gates (flag, never drop)

Classifier intact (above). Penalty value logged per epoch (adversary_log.jsonl `penalty`); a run whose penalty does not
fall below its epoch-1 value by the last epoch is flagged "penalty not optimised".

## Controls

The no-penalty λ_adv = 0 control (seed 42) and the published λ = 0 rows (5 seeds).

## CI / bootstrap and rule A

PREREGISTER_BOOTSTRAP.json unchanged; ID by lesion group, pad_heldout by patient, Fitzpatrick by image, SCIN by case.
Rule A (pooled CI below 0.5 and ≥ 3 of 5 seed CIs below 0.5, Mahalanobis) per selected cell on pad_heldout and
Fitzpatrick. The scan (one seed) has no CI.

## Reported cells

Scan table (penalty, weight, epochs, selection probe, val balanced accuracy, leakage, ID balanced accuracy, pad_heldout
Mahalanobis); selection; per selected penalty: Table 1 row (5 seeds), bootstrap rules, matched GRL comparator,
diagnostics; LaTeX rows; one plain paragraph per selected cell.

## Predicted outcome (written before any score exists)

- MK-MMD reaches a selection probe of 0.60–0.70 with the classifier intact at w = 10–30; scaled CORAL reaches
  0.65–0.75 (first and second moments leave higher-order differences a linear probe can still use).
- At matched leakage, pad_heldout Mahalanobis AUROC is below 0.5 for MK-MMD (0.40–0.48, rule A) and within 0.05 of the
  matched GRL runs; CORAL below 0.5 in mean (rule A or B). Fitzpatrick follows pad_heldout.
- Reading: the inversion is a property of aligning the two marginals under label shift, not of the adversary.

## Outputs

`results/paperB/r5/p11/runs/<tag>/` (config.json, adversary_log.jsonl, summary.json, scores.npz, features/);
`SELECTION.json`, `summary.csv`, `BOOTSTRAP.json`, `REPORT.md`. Checkpoints `checkpoints/r5_p11/`. `--recipe p8` runs
carry `_recp8` and report to `p11/recipe_recp8/`.

## GPU-h estimate and basis

Basis: lesion-level dual EfficientNet-B3 run ≈ 1.6 GPU-h at 40 epochs (3.2 at 80); the penalties act on 16-d vectors of
32 + 32 rows (negligible).

| Stage | Runs | GPU-h |
|---|---:|---:|
| Control | 1 | 1.6 |
| Scan, 40 epochs | 8 | 12.8 |
| Scan, 80 epochs | 2 | 6.4 |
| Seeds (2 penalties × 4) | 8 | 12.8 (both 40-epoch) to 25.6 (both 80-epoch) |
| Total | 19 | 33.6–46.4 |

The approved budget is ≈ 40 GPU-h; if both selections are 80-epoch configs the seed stage exceeds it by ≈ 6 GPU-h and
the user decides before `launch_r5_p11.sh seeds`. CPU: selection ≈ 0.05 h, report ≈ 0.3 h.

## Stop rules and caps

Scan fixed (11 runs). Seed stage only for a selected penalty. A failed run is re-run once with the same seed; a second
failure is reported.

## Deviations

## Amendment 2026-10-16 07:33 (UTC+7) — queue order, MK-MMD citation, comparable invariance (committed before any P11 run)

User decision 2026-10-16 07:32. Fixed now; not changed after results are seen.

### Queue

P11 is raised to run **right after P9 and before P7(b)**. Dependency: P11 scan starts after the P9 v2 report job
(68090); P7(b) is held until the P11 seeds report finishes. Recipe: `RECIPE=auto` → the main-recipe rule outcome
(`current` / old Table 1 recipe). Checkpoint policy: **last.ckpt** primary (P8 deviation 2026-10-15 21:57), best-val
secondary.

### MK-MMD (DAN)

MK-MMD is as in DAN (Long et al. 2015): multi-kernel Gaussian MMD with bandwidths = median pairwise squared distance
of the pooled batch (detached) × {0.25, 0.5, 1, 2, 4}, unbiased MMD², kernels averaged (`scripts/r5_models.py:mk_mmd`).
Wu et al. (2023, arXiv:2309.11705) report that under DAN on rotated MNIST, OOD detection stays weak (Mahalanobis AUROC
32.9, ODIN 30.8) — i.e. alignment need not produce a useful OOD detector; P11 tests the complementary claim on derm
data (whether matched invariance with MK-MMD / CORAL produces detector collapse or inversion).

### Comparable invariance (outcome rule; replaces the old “matched leakage ±0.05 on Table 1 pad_full probe” claim)

Verbatim:

> “Comparable invariance”: held-out-probe leakage (balanced accuracy, same probe as P6/A3) within 0.03 of GRL at λ = 2
> on the same probe, with ID bal acc within 0.05 of λ = 0.
> If no MMD / MK-MMD setting meets both, report “equivalent invariance not reached”, not “MMD does not collapse”.
> If a setting meets both, report all 5 detectors on pad_heldout and Fitzpatrick with rule A, and state whether it
> collapses or inverts.

Operationalisation (fixed):
- Held-out probe = P6 V1: ISIC test ∪ pad_heldout, 70/30 GroupShuffleSplit by lesion / patient, probe seeds 42/52/62,
  logistic probe on `z_lesion_norm` (same function as `r5_p9v2.v1_probe` / P6 A3). Chance floor 0.5.
- GRL λ = 2 reference: mean V1 over the five Table 1 runs `runB_orth1_ladv2_s{42,52,62,72,82}` from
  `results/paperB/r5/p6/per_run/*.json` (`z_lesion_norm.V1.bal_acc_mean`). GRL λ = 0 reference: mean ID test balanced
  accuracy over the five Table 1 λ = 0 runs (item-4 summaries).
- A P11 setting (penalty, weight, epochs; after the seed stage if selected, else the seed-42 scan row) **meets**
  comparable invariance iff |V1_P11 − V1_GRL_λ2| ≤ 0.03 and |ID_P11 − ID_GRL_λ0| ≤ 0.05 (means over seeds when n ≥ 3;
  seed-42 only for unscanned reporting of non-selected weights).
- **Collapse** = mean Mahalanobis AUROC on pad_heldout in [0.45, 0.55] (descriptive band). **Inversion** = rule A met
  on pad_heldout (pooled CI upper bound < 0.5 and ≥ 3/5 seed CIs upper bound < 0.5). Both stated; neither is a
  pass/fail for the item.
- Scan / weight selection (val vs pad_adv → 0.60, classifier intact) is unchanged and is only a pipeline to pick which
  configs get 5 seeds. The claim language above is the only outcome language for the MMD / MK-MMD arms.

### Predicted outcome (written before any P11 score exists)

- P1: at least one MK-MMD weight in {10, 30} (40 epochs) reaches comparable invariance (V1 within 0.03 of GRL λ = 2,
  ID within 0.05 of λ = 0); scaled CORAL does not (selection probe can fall, but held-out V1 stays > GRL λ = 2 + 0.03).
- P2: every MK-MMD setting that meets comparable invariance **inverts** on pad_heldout (rule A met; mean Maha
  0.40–0.48), Fitzpatrick in the same direction; reading: inversion is not adversary-specific.
- P3: if P1 fails for all MK-MMD and CORAL weights, the report states only “equivalent invariance not reached” (no
  claim that “MMD does not collapse”).
