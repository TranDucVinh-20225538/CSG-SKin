# Paper B R5 / P8 — stronger dual-encoder base model

Committed before any P8 run (no P8 training, feature extraction or scoring has run at this commit).
Fixed now; not changed after results are seen. Deviations are appended at the end.

## Question

The lesion-level dual encoder reaches ID balanced accuracy 0.496 ± 0.016 at λ = 0 (R4 Table 1; best val macro recall
≈ 0.47 in `runB_orth1_ladv0_s42`). Can a recipe change make it a competent classifier, chosen on the validation split
at λ = 0 only, and does the inversion survive with that model (λ ∈ {0, 0.5, 2} × 5 seeds; all 7 λ if it survives)?
User decision: if P8 makes the model competent, the P8 recipe becomes the paper's main recipe and R1/R2 (and P7b) use it
with their own λ = 0 baselines; otherwise the current recipe stays. Recipes are never mixed within a table.

## Design

Code: `scripts/train_r5_p8.py` (stages), `scripts/r6_common.py` (shared path), report `scripts/r6_report.py --item p8`.
The training path is the lesion-level Table 1 path (`train_r2_item4_lesion_split.py` → `train_phase3_sweep.py`)
re-implemented on the same objects: `CSGLiteLightning` subclass `R6Lightning`, same constructor arguments except the
recipe fields, AdamW lr 1e-4, wd 1e-4, adversary LR × 30, GRL ramp (γ 20, power 0.5, α_min 0.2), λ_ctx 1, λ_orth 1,
λ_supcon 0, batch 32 ISIC + 32 PAD, 16-mixed precision, checkpoint = best `val/acc` (torchmetrics MulticlassAccuracy,
macro average = balanced accuracy on ISIC val). Existing scripts and their defaults are unchanged.

Recipe knobs (work order: latent size, classifier LR multiplier, longer schedule, class-balanced loss/sampling):

| id | latent | classifier LR × | class balancing | note |
|---|---:|---:|---|---|
| A0 | 16 | 0.2 | none | current recipe |
| A1 | 64 | 1.0 | none | centre point |
| A2 | 128 | 1.0 | none | latent |
| A3 | 64 | 5.0 | none | classifier LR |
| A4 | 64 | 1.0 | loss (inverse frequency, Σ π_c w_c = 1) | balancing by loss |
| A5 | 64 | 1.0 | sampling (uniform class mix in the classifier stream) | balancing by sampling |
| A6 | 16 | 1.0 | none | classifier LR at the current latent |
| A7 | 128 | 1.0 | loss | combination |

Class-balanced sampling (A5) uses the three-stream batch (classifier stream resampled to a uniform class mix; a separate
ISIC adversary stream at the natural ISIC mix; PAD stream unchanged). In the legacy two-stream batch the same resampled
ISIC images would also feed the adversary and change the label shift it sees (uniform ISIC mix has 12.5% nevus, close to
PAD's 9.9%), confounding P8 with R6 R1. The latent size also scales the adversary head (hidden = max(d, 32),
max(d/2, 16), unchanged template).

Stages:

1. **Stage A**: the 8 recipes at λ = 0, 40 epochs, seeds 42 and 52 (16 runs). Evaluation on ISIC val only (val balanced
   accuracy, accuracy, ECE, per-class recall). No test or OOD feature is extracted for grid runs.
   Plus a comparator: the single-encoder ERM (`train_phase15_1_single_dann.py`, λ = 0, unchanged code) on the
   lesion-level split, seeds 42 and 52 (2 runs). Not a candidate recipe.
2. **Selection A**: highest mean val balanced accuracy (2 seeds). Among recipes within 0.01 of the top: fewest changes
   from the current recipe, then smaller latent, then grid order.
3. **Stage B** (longer schedule): the stage-A winner at 80 epochs, seeds 42, 52. 80 epochs is kept only if the mean val
   balanced accuracy gains ≥ 0.02 (the threshold is above 0.01 because 80 epochs doubles the cost of every later item
   that uses the recipe). Output `results/paperB/r5/p8/selection_stageB.json`, plus `results/paperB/r5/p8/SELECTED_RECIPE.json` in the flat
   format `scripts/r5_recipe.py` reads for `--recipe p8` (Q1/P9/P10/Q3/P11): `lesion_latent_dim`, `cls_lr_multiplier`,
   `max_epochs`, `class_balance` (loss → `weighted_ce`, sampling → `sampler`).
4. **Main**: selected recipe, λ ∈ {0, 0.5, 2} × seeds 42/52/62/72/82, full evaluation. λ = 0 seeds 42/52 reuse the stage
   A/B checkpoint of the selected recipe (same recipe, λ and seed); only the full evaluation is run on it.
5. **Extension** (conditional): λ ∈ {0.25, 1, 4, 8} × 5 seeds, run only if the inversion survives (rule A met at λ = 2,
   Mahalanobis, pad_heldout) under the P8 recipe. The gate is read from `RECIPE_DECISION.json` written by the report.

## Data and splits

Lesion-level ISIC split of R2 item 4 (StratifiedGroupKFold by lesion_id, null lesion_id = own group): train 16,211,
val 4,053, test 5,067. PAD: Phase 2 patient split, pad_adv 1,582 (adversary) / pad_heldout 716 (OOD). Selection uses
val only; test enters only the main-stage evaluation after the recipe is frozen.

## Seeds, λ grid, OOD sets

Seeds 42/52 (grid), 42/52/62/72/82 (main). λ main {0, 0.5, 2}; extension {0.25, 1, 4, 8}.
OOD: pad_heldout (primary), pad_full, Fitzpatrick17k, and the P7a sets (MED-NODE, SD-198, SCIN;
`results/paperB/r5/p7a/sets/*.csv`) when their CSVs exist at run time.

## Columns, detectors, bootstrap

Table 1 columns (R4 order): leakage (logistic probe on z_lesion_norm, ISIC test vs PAD full, 3 probe seeds, balanced
accuracy), ID bal acc, ID ECE (15 bins), OOD ECE (pad_heldout, 6-class restricted), OOD AUROC (Mahalanobis
class-conditional, pad_heldout), Fitzpatrick AUROC, X-domain balanced accuracy (pad_heldout, 6 classes). Five detectors
(Mahalanobis, kNN k = 50, cosine, MSP, Energy T = 1), fit on ISIC train only. Extra: class-matched leakage (ISIC test
subsample at the PAD class mix), AK recall, NV-predicted share, NV-nearest fraction.
Bootstrap rule A: `PREREGISTER_BOOTSTRAP.json` unchanged (2,000 draws, percentile 95%, draws shared across seeds within a
cell; a fresh generator seeded 20261005 per cell and OOD set). ID resampled by lesion group (as R3 A2), pad_heldout and
pad_full by patient, Fitzpatrick by image (anti-conservative), P7a sets by their group column. Rule A: at λ = 2 the
pooled CI entirely below 0.5 and ≥ 3/5 per-seed CIs below 0.5 (Mahalanobis); reported at every λ.
Per-class AUROC and decomposition check (R3 A4 / R4): ID-class AUROC_c, ID reweighted to the OOD class mix, the identity
Σ_c π_OOD(c)·AUROC_c, OOD-side per-class AUROC, OOD NV vs non-NV.

## Competence target and gates (flag, never drop)

- Competence flag: λ = 0 mean ID test balanced accuracy (5 seeds) ≥ 0.70. Reported next to the image-level
  single-encoder ERM 0.779 (paper) and the lesion-level single-encoder ERM comparator. The 0.779 comes from the
  image-level split with plain-accuracy checkpointing, so it is a reference, not a like-for-like target.
- No headroom: λ = 0 Mahalanobis on pad_heldout in [0.45, 0.55].
- Adversary not engaged: λ > 0 cell whose mean leakage is not ≥ 0.05 below the λ = 0 cell.
- The competence flag and the extension gate are written to `RECIPE_DECISION.json`; the main-recipe choice stays with
  the user.

## Controls

A0 (current recipe) inside the grid; the current-recipe lesion-level Table 1 runs reported in a separate table; the
lesion-level single-encoder ERM comparator.

## Predicted outcome (written now)

- Stage A: latent 64 with classifier LR × 1 (A1) raises val balanced accuracy from ≈ 0.47 (A0) to 0.58–0.68; latent
  128 within 0.02 of 64; classifier LR × 5 within 0.02 of × 1; class-balanced loss adds 0.02–0.05; sampling within
  0.02 of loss. Winner: A4 or A7. 80 epochs gains < 0.02 (40 epochs kept).
- Main, λ = 0: ID test balanced accuracy 0.60–0.72 (competence flag set in about half of this range); Mahalanobis
  pad_heldout 0.75–0.85.
- Main, λ = 2: Mahalanobis pad_heldout 0.38–0.48, rule A met (the inversion survives a stronger classifier because its
  driver, the ISIC→PAD label shift seen by the adversary, is unchanged); Fitzpatrick below 0.5; leakage 0.55–0.65.
- Lesion-level single-encoder ERM comparator: 0.65–0.75.

## Outputs

`results/paperB/r5/p8/`: `grid/runs/<recipe>_ladv0_s<seed>/` (val metrics), `selection_stageA.json`,
`selection_stageB.json`, `main/runs/main_ladv<λ>_s<seed>/summary.json`, `main/features/...` (z_lesion, z_lesion_norm,
z_context, logits, labels per split), `erm_ref/`, `summary.csv`, `cells.json`, `REPORT.md` (Table 1 LaTeX rows, one
plain paragraph per result), `RECIPE_DECISION.json`. Checkpoints `checkpoints/r5_p8/`.

## GPU-hour estimate and basis

Basis: measured ≈ 1.6 GPU-h per lesion-level dual-encoder run (40 epochs, two-stream, including evaluation; 83–110 min).
Assumed ×1.5 for three-stream batches (96 instead of 64 images per step; to be checked on the first run), ×2 for 80
epochs, ≈ 0.2 GPU-h for evaluation of a reused checkpoint.

| stage | runs | GPU-h |
|---|---:|---:|
| A (7 two-stream recipes × 2, A5 × 2 three-stream, ERM comparator × 2) | 18 | 22.4 + 4.8 + 3.2 = 30.4 |
| B (winner, 80 epochs × 2) | 2 | 6.4 (two-stream) – 9.6 (A5) |
| main (13 new + 2 reused) | 15 | 21.2 (two-stream, 40 ep) – 62.8 (A5, 80 ep) |
| **P8 without extension** | | **58 typical, ≤ 103** |
| extension (conditional) | 20 | 32 typical, ≤ 96 |

CPU: selection seconds; report ≈ 1 CPU-h (8 cores, < 30 min wall).

## Stop rules and budget caps

- Cap: 110 GPU-h for stages A–main, 96 GPU-h for the extension. If the first completed run takes more than 1.5× its
  estimate, the parent re-estimates before the rest of the array runs.
- If stage A is incomplete, no selection is made (the selection step exits non-zero and the chain stops).
- A run that fails twice is reported with its log and not retried silently.

## Pre-run amendment (user decision 2026-10-15, before any P8 run)

The schedule is fixed at 40 epochs: stage B (80 epochs) is not run (`LOCK_40_EPOCHS = True` in
`scripts/train_r5_p8.py`; the launcher chains selection directly after stage A). Reason given: an 80-epoch recipe would
double the cost of R1, R2 and P7(b), which inherit the P8 recipe. The prediction above ("80 epochs gains < 0.02") is
therefore not tested. Revised estimate: stage A 30.4 + main 21.2 (two-stream) – 31.6 (A5 three-stream) =
**≈ 52 GPU-h typical, ≤ 62**; extension (conditional) 32 typical, ≤ 48.

## Deviations

- 2026-10-15 21:30 (UTC+7), user-directed, before any main-stage job has started. Main-stage task order only: λ = 0
  seed 42 and λ = 2 seed 42 are tasks 0 and 1, so a preliminary 1-seed look is available early. Same 15 runs, recipe,
  seeds and evaluation; nothing else changed.
- 2026-10-15 21:57 (UTC+7), user-directed, before any λ > 0 number of P8 main exists (main tasks 3–14 held; only the
  λ = 0 seed 42 summary exists; λ = 2 seed 42 is mid-training). **Primary checkpoint = last.ckpt (end of epoch 40)** for
  every λ in P8 and in everything that inherits the P8 recipe (R1, R2, P9, P7(b)). Reason: with the selected recipe
  (A4), the best-val checkpoints fall at epochs 0–3. That is before the GRL warm-up takes effect, so those checkpoints
  do not test DANN. The best-val checkpoint is reported as secondary, with its epoch. λ = 0 seeds 42/52 reuse the
  grid run's last.ckpt (same recipe, λ, seed). The stage-A/B recipe selection (mean validation balanced accuracy,
  best-val) is not re-run. The competence comparison is added (old Table 1 recipe vs A4 at λ = 0, val and test
  balanced accuracy, at best-val and last checkpoints), and so is an audit of the selected epoch of the 35 Table 1
  runs.
- 2026-10-15 22:05 (UTC+7), factual note on the 21:57 deviation (decision unchanged). The P8 recipe and the Table 1
  trainer share the `csg_lightning` GRL schedule α = max(0.2, 2/(1+exp(−20 p^0.5)) − 1), p = step / total steps, with
  no adversary warm-up epochs: the adversary and the reversed gradient act from step 0. Logged epoch-mean α (Table 1
  λ = 2 s42): 0.74 (epoch 0), 0.96 (1), 0.99 (2), ≥ 0.994 from epoch 3. "Before the warm-up takes effect" therefore
  applies to epoch 0 (and partly 1), not to epochs 2–3. Training-side adversary accuracy continues to fall towards
  0.5 over later epochs. Item 4 flag criterion: λ > 0 and epoch-mean α < 0.99 at the selected epoch (fixed after the
  35 selected epochs were listed; they range 6–39, so any α-based cut at epochs 0–3 gives the same result). Items 3/4
  are implemented in `scripts/r5_ckpt_audit.py` (outputs `results/paperB/r5/ckpt_audit/`). Re-evaluation of main
  tasks 0–2 under the new policy: separate array jobs after the running tasks end; held tasks 3–14 start after the
  λ = 2 seed 42 re-evaluation.
- 2026-10-15 23:30 (UTC+7), user-directed (23:11 and 23:14), additions only; primary analyses, recipe, seeds and the
  last.ckpt policy are unchanged. (a) Competence report adds the lesion-level single-encoder ERM λ = 0 reference
  (`erm_ref`, seeds 42/52) at best-val and last.ckpt (`r5_ckpt_audit.py erm`); the 0.779 is labelled image-level
  (phase 15 image-level 5-seed values listed next to it). (b) Held-out-probe leakage (P6 V1, one function for every
  row) for P8 main next to the current-recipe dual encoder at the same λ (0, 0.5, 2), both checkpoints
  (`r5_p8_heldout.py`). (c) Main λ = 2 tasks 11–14 (seeds 52–82, not yet started) save per-epoch weights
  (weights only; no effect on training); P8 stage `traj` adds two per-epoch runs (λ = 0 and 2, seed 42), plan in
  `decisions/paperB_r5_p8_traj.md`.
- 2026-10-15 23:30 (UTC+7), user request 23:25, CPU, descriptive (`scripts/r5_val_se.py`, outputs
  `results/paperB/r5/p8/val_se/`). A4 λ = 0 seed 42: lesion-clustered bootstrap SE of validation balanced accuracy on the
  lesion-level ISIC val split (PREREGISTER 2000 draws, seed 20261005) at best-val and last.ckpt; range, SD and
  epoch-to-epoch change of the logged val bal acc over epochs 2–39. Prediction: SE ≈ 0.01–0.02; the epoch-to-epoch range
  after epoch 1 is of the same order as or larger than 2 SE, i.e. the epoch-1 selection is within validation noise.

## Main-recipe rule (user decision 2026-10-15 23:48 UTC+7; committed before the 5-seed P8 main summary exists)

Recorded verbatim:

> Main-recipe rule. Compare A4 with the old Table 1 recipe at λ = 0, on the lesion-level validation split (not test),
> at last.ckpt for both, over seeds 42–82 (5 seeds each, matched by seed). Let D = the mean over seeds of (A4 − old)
> val bal acc, with a 95% paired t-interval across the 5 seeds.
>
> A4 becomes the main recipe only if D ≥ 0.03 and the CI excludes 0. Then P8 ext runs, and R1/R2/P9/P7(b) use A4.
> Otherwise the old recipe stays main: Table 1 unchanged (last.ckpt shown alongside best-val), A4 at 3 λ is reported as
> a robustness check, P8 ext is skipped, and R1/R2/P9/P7(b) use the old recipe with last.ckpt.
> Apply the rule mechanically when the numbers arrive and report D, its CI and the outcome. Until then, hold P8 ext and
> the recipe choice for R1/R2. If GPUs would idle, run P9 pilot round 1 or the Q1 headroom gate.

Implementation (fixed now):
- Old recipe values: `results/paperB/r5/ckpt_audit/compete/runB_orth1_ladv0_s<seed>.json["last_val_bal_acc"]`
  (CPU re-forward of `checkpoints/r2_item4/runB_orth1_ladv0_s<seed>/last.ckpt` on the lesion-level ISIC val split).
  A4 values: `results/paperB/r5/p8/main/runs/main_ladv0_s<seed>/summary.json["metrics"]["val"]["bal_acc"]` with
  `checkpoint_policy == "last"` (seeds 42/52 = the grid run's last.ckpt). Seeds 42, 52, 62, 72, 82 for both.
- d_s = A4_s − old_s; D = mean(d_s); 95 % CI = D ± t(0.975, df = 4) · SD(d_s) / √5. "CI excludes 0" = lower bound > 0
  (D ≥ 0.03 already requires the positive side). All ten values must be present; otherwise no outcome is written.
- Output: `results/paperB/r5/p8/MAIN_RECIPE_RULE.json` {"D", "ci", "per_seed", "outcome": "A4" | "old",
  "main_recipe": "p8" | "current"} and `MAIN_RECIPE_RULE.md` (`scripts/r5_main_recipe_rule.py`).
- `RECIPE=auto` (R1, R2, P7(b), P9; `r6_common.resolve_recipe` and `r5_recipe.concrete`) reads `main_recipe` from this
  file and stops if it is missing. This replaces the competence flag (`RECIPE_DECISION.json["competent"]`) as the
  switch. P8 ext runs only if outcome = A4 (replaces the `inversion_survives` gate). Under "old", R1/R2/P7(b)/P9 run
  the current recipe with the last.ckpt policy, and λ = 0 reuse in R1 points at the Table 1 checkpoints (last.ckpt).
- Holds: P8 ext, R1, R2, P7(b) and P9 GPU stages get a dependency on the rule job. GPU idle filler before the rule: the
  Q1 headroom gate only (it runs with RECIPE=current, so it does not depend on the rule). P9 pilot round 1 runs with
  RECIPE=auto and therefore waits for the rule; it is not used as filler. With P8 main occupying both GPUs until
  ≈ 07:30 and the rule inputs complete at ≈ 02:00, no idle gap is expected.
