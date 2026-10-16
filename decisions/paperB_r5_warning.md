# Paper B R5 — two-tier warning signal (ID-only vs deployment)

Committed before any run of `scripts/r5_warning_signal.py`. Fixed now; deviations appended.
Does not change the GPU order. Reuses Q4 / R6 R4 geometry where named; the threshold and evaluation
protocol below supersede the Youden leave-one-group-out rule of R6 R4 for the paper's warning readout.

## Question

Can a pre-committed, λ = 0–calibrated warning flag inverted runs (rule A) without using OOD labels at
flag time — and do the ID-only tier and the deployment tier differ?

## Two tiers (reported separately)

**Tier 1 — ID-only.** Centroid-collapse statistic on ISIC validation and test features (no deployment
features, no OOD labels). Fit class means and the majority prior on ISIC train (pooled-covariance
Mahalanobis + 1e-3·I, same geometry as the detectors). Statistic = majority-centroid nearest share
minus the train majority prior, evaluated on ISIC val and on ISIC test (both reported; primary within
the tier = ISIC test). Code name: `centroid_collapse_excess` (Q4 `maj_share_excess`).

**Tier 2 — deployment target.** On the unlabeled deployment set (ISIC runs: pad_heldout):
(a) majority-class prediction rate minus the ID-train prior (`pred_maj_frac_excess`, R6 R4 W1);
(b) BBSE prior shift: TV between the BBSE estimate of the deployment class mix and the source prior
(`bbse_tv`, R6 R4 W2). Labels of the deployment set are never used by either signal.

## Pre-committed threshold

For each statistic, the flag threshold is the **95th percentile of that statistic over all λ = 0 runs
in the evaluation corpus**, computed from λ = 0 runs only and written to disk **before** any OOD
AUROC or rule-A label is read for thresholding. A run is flagged on a statistic if its value exceeds
that λ = 0 percentile. No Youden, no looking at inverted runs when setting the cut.

## Corpus (primary evaluation is per run)

- Old recipe (R2 item 4 / Table 1 lesion-level): 35 runs (7 λ × 5 seeds).
- A4 (P8 main, last.ckpt): 15 runs (3 λ × 5 seeds).
- Plus P9 / P11 / P10 (and further) runs as their feature directories arrive (same columns via manifest
  extension). Pending rows are listed, not dropped.

## Outcomes and analysis (primary)

Per run, against **“inverts under rule A”** on the OOD set under evaluation (primary OOD = pad_heldout;
Fitzpatrick stored):
- Seed-level rule-A proxy: Mahalanobis bootstrap CI for that seed entirely below 0.5 when a bootstrap
  exists; otherwise the point Mahalanobis AUROC < 0.5 (labelled proxy in the report).
- For each tier/statistic flag: sensitivity and specificity vs that invert label.
- Spearman ρ between each statistic and OOD AUROC (Mahalanobis; also all five detectors stored),
  with a 95% run-bootstrap CI (2,000, rng 20261005).

## Secondary (labelled weak)

Leave-one-setting-out transfer of the λ = 0 95th-percentile rule (settings = old recipe vs A4 vs each
later family as it appears). Labelled **weak**: few settings. Not the primary claim.

## Predicted outcome (written before scores)

- Tier 1 (ID-only) barely moves with λ; sensitivity vs rule-A invert ≤ 0.5; |ρ| with Mahalanobis AUROC ≤ 0.3.
- Tier 2 `pred_maj_frac_excess` tracks inversion: sensitivity ≥ 0.9 on pad_heldout among old+A4 runs;
  ρ ≤ −0.7 with Mahalanobis AUROC.
- Tier 2 `bbse_tv` weaker (|ρ| 0.3–0.6) because the ID confusion matrix is itself distorted under collapse.

## Outputs

`results/paperB/r5/warning/`: `thresholds_lambda0.json` (written first), `per_run/*.json`, `summary.csv`,
`sens_spec.csv`, `correlations.csv`, `leave_one_setting_out.csv` (weak), `REPORT.md`.

## Cost

GPU 0. CPU ≈ 0.5–1 h on 8–16 CPUs for the current corpus.

## Stop rules

Tier definitions, λ = 0 95th-percentile rule, and primary vs weak evaluation fixed now.

## Deviations

- 2026-10-16 (+07). Expanded readout: (a) specificity recomputed on λ > 0 non-inverting runs only, with
  n_pos/n_neg beside every sens/spec; (b) R1 shift 0%/100% (λ = 2 now; other λ as features arrive) plus
  R2/P9/P11/P10 when feature dirs appear; (c) invert labels on Fitzpatrick as well as pad_heldout;
  (d) BBSE TV audit — positive Spearman with AUROC is not a sign bug (TV unsigned; BBSE shrinks under
  collapse). **Frozen** `thresholds_lambda0.json` unchanged (not refit). Pre-committed verdict “tier 1
  failed” kept visible in `REPORT.md` / `precommitted_primary.json`.
