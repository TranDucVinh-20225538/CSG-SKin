# Paper B R5 / D1 — does inversion follow the direction the adversary removes?

Committed before any D1 number is computed. Fixed now; not changed after results are seen. Deviations are appended at
the end.

## Question (user, 2026-10-15)

"Domain-direction" hypothesis: inversion happens when an OOD set differs from ISIC mainly along the direction the
adversary removes. If so, within a set, the images that the λ = 0 model places furthest toward PAD (the adversary's
other domain) should be the ones pulled most toward the ID class centroids when the adversary is switched on, and
the part of an image's deviation that is orthogonal to that direction should predict the pull much less.

## Data and runs (CPU, existing features)

- **Lesion-level dual encoder** (Table 1 runs, `results/paperB/r2/item4/features/runB_orth1_ladv*`): λ = 0 and λ = 2,
  seeds 42 / 52 / 62 / 72 / 82, matched by seed (5 pairs). Sets: pad_heldout (716 images / 412 patients),
  Fitzpatrick17k (3,887), MED-NODE (170), SD-198, SCIN (10,406), from the cached feature files.
- **BCN↔HAM** (phase16, EfficientNet-B3 single-encoder DANN): BCN→HAM (ID BCN, OOD HAM held-out; adversary pool
  ham_adv) and HAM→BCN (ID HAM, OOD BCN held-out; adversary pool bcn_adv). λ = 2 exists only at seed 42, so the
  only matched pair is λ = 0 s42 vs λ = 2 s42 (1 pair per direction). id_train / id_test / ood features come from the
  S-extract job (`results/paperB/r5/s_flip/features/`). The adversary-pool features of the two λ = 0 s42 checkpoints
  do not exist yet: they are extracted on CPU in this item with the same path as S-extract (SingleDannLightning,
  build_eval_tf(448), collect_all). This is the only forward pass in D1.

## Domain score (λ = 0 checkpoints only)

Linear domain classifier: StandardScaler + LogisticRegression (max_iter 4000, C = 1, class_weight = balanced).
- Lesion-level runs: fit on ISIC train (domain 0) vs pad_adv (domain 1, the 1,582 PAD images the adversary trained on).
- Phase16 runs: fit on the ID-site train split vs the adversary pool (other site), i.e. the domains of that model's
  adversary.

It is fit separately on two representations of the λ = 0 model:
- **latent**: z_lesion_norm (16-d; the space the adversary acts on and the Table 1 detector space);
- **backbone**: backbone_raw_lesion (1,536-d).

The phase16 single encoder has one representation (its pooled BN features z), reported once as "latent".

- **Domain score** of an image = the classifier's decision function (log-odds of the adversary's second domain),
  computed for every image of every set.
- **Validity check**: balanced accuracy of the classifier on ISIC test vs pad_heldout (lesion-level) and on ID test
  vs OOD held-out (phase16).

## Orthogonal control

In the classifier's standardised space, u = (x − μ_train) / σ_train and ŵ = coef / ‖coef‖. The control is
o = ‖u − (u·ŵ) ŵ‖, the size of the image's deviation from the training mean orthogonal to the domain direction.

## Pull (per image, λ = 0 → λ = 2, same seed)

Mahalanobis in the Table 1 detector space of each run (z_lesion_norm for the lesion-level runs, z for phase16):
minimum over classes of the squared distance, shared covariance + 1e-3·I, fit on ISIC train (phase16: ID train), as
in Table 1. d = sqrt(min d²).

Distances are not on a common scale across runs, so:
- **primary pull** = d₀ / med₀ − d₂ / med₂, where med is the median d over that run's ID test images (the distance
  drop in units of the run's typical ID distance);
- **secondary**: the percentile drop P₀ − P₂ (P = fraction of the run's ID test images with d ≤ the image's d; this
  is the per-image term of the AUROC), and the raw drop d₀ − d₂.

Positive pull = the image moved toward the ID centroids relative to ID.

## Per-image test (primary)

Within each set and representation, per seed:
- ρ_dom = Spearman(domain score, primary pull);
- ρ_orth = Spearman(o, primary pull).

The statistics are the 5-seed means (1 seed for phase16) of ρ_dom and ρ_orth, and Δ = ρ_dom − |ρ_orth| (the control
counts as predictive in either sign).

Bootstrap (PREREGISTER: 2,000 draws, rng seed 20261005): images are resampled by group (pad_heldout by patient,
Fitzpatrick by P13 cluster_union, MED-NODE / SD-198 / SCIN by the P7a `group` column, HAM / BCN by lesion_id). Draws
are shared across seeds and between ρ_dom and ρ_orth. CI = percentile 95% of the seed mean.

Secondary per set (points, plus the same bootstrap for the partial correlation):
- ρ_dom and ρ_orth for the percentile and raw pulls;
- the partial Spearman of domain score vs primary pull, controlling for the λ = 0 percentile P₀ (images that start
  far from ID can only move down, i.e. regression to the mean);
- the same partial for o.

**Per-set criterion.** The domain score "predicts the pull better than the orthogonal control" in a set if ρ_dom > 0
and the CI of Δ is entirely above 0 (latent representation = primary; backbone reported with the same criterion).

**Refutation (fixed now).** Seven sets: pad_heldout, Fitzpatrick, MED-NODE, SD-198, SCIN, BCN→HAM (HAM held-out) and
HAM→BCN (BCN held-out). If the criterion holds in fewer than 4 of 7 sets (latent), the hypothesis is not supported.
If it holds in ≥ 4 of 7, it is supported at the per-image level. The backbone count is reported next to it and does
not change the verdict.

## Across sets (secondary, descriptive only; n = 7)

Per set: mean domain score (λ = 0, mean over seeds) and ΔAUROC = Mahalanobis AUROC(λ = 2) − AUROC(λ = 0), mean over
matched seeds (recomputed from the same features). A table and the Spearman over sets are reported with no CI and no
test. The phase16 domain score comes from a different classifier (ID site vs other site), so it is reported both with
and without the two BCN↔HAM sets.

## Class mix (label shift next to domain direction)

- From R6-R5 (`results/paperB/r6/r5/summary.csv`): TV, KL, majority shares for HAM→BCN, BCN→HAM, ISIC→PAD and
  ISIC→Fitzpatrick.
- MED-NODE, SD-198 and SCIN are added with the same `shift()` function (r6_r5_label_shift), vs the lesion-level ISIC
  train class mix, on the P7a `mapped` labels; unmapped counts are reported.

The table puts TV next to ΔAUROC and next to the per-set ρ_dom, so label shift and domain direction can be read on the
same sets. No combined model is fitted (n = 7).

## Predictions (written before any D1 number exists)

1. Validity: the λ = 0 domain classifier's held-out balanced accuracy is ≥ 0.90 (latent) and ≥ 0.95 (backbone) for
   ISIC vs pad_heldout, and 0.75–0.90 for the BCN↔HAM site classifiers.
2. pad_heldout (latent): ρ_dom 0.30–0.60; |ρ_orth| ≤ 0.25; criterion met.
3. Fitzpatrick, MED-NODE, SD-198 (latent): ρ_dom 0.10–0.40; criterion met in 2 of these 3.
4. SCIN (the set that inverts least, ≈ 0.53 at λ ≥ 0.5): ρ_dom between −0.05 and 0.20; criterion not met.
5. BCN→HAM (no change with λ, 0.33–0.43 at every λ): ρ_dom within ±0.15, criterion not met. HAM→BCN (0.85 → 0.65):
   ρ_dom 0.15–0.40, criterion met.
6. Overall: the criterion holds in 4 of 7 sets (latent) and the hypothesis is supported at the per-image level, but
   narrowly (3–5 of 7 plausible). Backbone ρ_dom is lower than latent by about 0.1 in most sets.
7. ρ_orth is positive (0.05–0.30) in most sets, because images that start far from ID move most in any direction.
   The partial ρ_dom (given P₀) is smaller than the raw ρ_dom by ≥ 0.05 in most sets.
8. Across sets: pad_heldout has the highest mean latent domain score among the five lesion-level sets. The Spearman
   between mean domain score and −ΔAUROC over the five lesion-level sets is ≥ 0.5 (descriptive).
9. Class mix: TV does not order the sets the same way as ΔAUROC (e.g. BCN→HAM has a large TV, 0.36, and no
   inversion change).

## Outputs

`results/paperB/r5/d1_domain/`: `features_adv/<run>/adv_pool.npz` (phase16 adversary-pool features),
`per_image/<set>_<rep>_s<seed>.parquet` (domain score, o, d₀, d₂, P₀, P₂, pulls), `per_run/*.json`,
`summary.csv`, `D1_RESULTS.json`, `REPORT.md` with LaTeX rows.

## Compute

CPU only: script `scripts/r5_d1_domain_direction.py`, one `pB-` job, 8 CPUs. The phase16 adversary-pool forward pass
(2 checkpoints, a few thousand images each at 448 px) takes about 0.5–1 h on CPU; the analysis about 0.5 h. The job
starts after S-extract (68070) so that the phase16 id/ood features exist.

## Deviations
- 2026-10-15 (before any D1 number). R6-R5's summary.csv already contains ISIC→MED-NODE, ISIC→SD-198 and ISIC→SCIN
  (mapped) rows computed with the same `shift()`; D1 reads all class-mix rows from that file instead of recomputing.
