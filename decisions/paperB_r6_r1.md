# Paper B R6 / R1 — causal dose test: remove the label shift seen by the adversary

Committed before any R1 run (no R1 training, feature extraction or scoring has run at this commit).
Fixed now; not changed after results are seen. Deviations are appended at the end.

## Question

If the ISIC→PAD label shift seen by the adversary causes the inversion (marginal alignment under label shift pulls
target images into the wrong classes; Zhao et al. 2019; Tachet des Combes et al. 2020), then lowering that label shift
while keeping everything else fixed should weaken the inversion.

## Design

Code: `scripts/train_r6_dose.py`, `scripts/r6_common.py` (mode `three`), report `scripts/r6_report.py --item r1`.
Each training step has three streams of 32 images:

1. ISIC classifier stream: the full lesion-level ISIC train split, natural order and mix. The classification loss uses
   this stream only, so the classifier sees exactly the current data.
2. ISIC adversary stream: drawn with replacement from the same ISIC train split, with per-image probability
   π_shift(c)/n_c. It feeds the adversarial domain head (domain 0) and the context-domain head; it has no
   classification loss.
3. PAD adversary stream: pad_adv (1,582 images) drawn uniformly, unchanged from the current setting (domain 1),
   built from the path column only.

Shift levels: π_shift = s·π_ISIC + (1 − s)·π_PAD with s = 1 (100%, ISIC's own mix, current setting), 0.5 (50%,
halfway), 0 (0%, PAD's mix). π_PAD is the class mix of pad_adv, the PAD images the adversary sees.

| class | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | TV distance to π_PAD |
|---|---|---|---|---|---|---|---|---|---|
| 100% (ISIC train) | 0.179 | 0.508 | 0.131 | 0.034 | 0.104 | 0.009 | 0.010 | 0.025 | 0.583 |
| 50% | 0.103 | 0.303 | 0.245 | 0.184 | 0.102 | 0.005 | 0.005 | 0.052 | 0.291 |
| 0% (pad_adv) | 0.027 | 0.099 | 0.360 | 0.334 | 0.101 | 0 | 0 | 0.080 | 0 |

At 0%, AK images are drawn ≈ 9.8 times per epoch on average, SCC 3.2, BCC 2.7; DF and VASC are not drawn.

**PAD labels are used only to set the resampling proportions π_PAD (computed once from the pad_adv label column at run
start and stored in each run's config). They never enter any loss: the PAD stream is built from image paths only, its
label field is −1, and the supervised-contrastive term is not computed.**

Everything else as the lesion-level Table 1 path (see `decisions/paperB_r5_p8.md`, Design): same model, optimiser,
GRL ramp, λ_ctx = λ_orth = 1, checkpoint on ISIC val macro recall, 40 epochs (or the P8 recipe's).

Runs: λ = 0 once per seed (three-stream, shift 100%), λ ∈ {0.5, 2} × {100%, 50%, 0%} × seeds 42/52/62/72/82 = 35 runs.
Run order: λ = 2 at 0% and 100% first, then λ = 0, λ = 2 at 50%, then λ = 0.5.

Recipe: one `--recipe` argument (`current`, `p8`, or `auto` = P8 if the P8 report flags the λ = 0 model competent).
Each recipe has its own λ = 0 runs and output directory (`.../r1/cur/`, `.../r1/p8/`); recipes are never mixed in a
table.

## Data and splits, OOD sets

Lesion-level ISIC split (train 16,211 / val 4,053 / test 5,067), Phase 2 PAD patient split (pad_adv 1,582 /
pad_heldout 716). OOD: pad_heldout (primary), pad_full, Fitzpatrick17k, P7a sets (MED-NODE, SD-198, SCIN) when their
CSVs exist.

## Reported

Per cell (λ × shift): Table 1 columns; five detectors on every OOD set; rule A bootstrap (Mahalanobis, kNN, cosine);
per-class AUROC and the decomposition check on pad_heldout; leakage (Table 1 probe) and class-matched leakage (ISIC test
subsample at the PAD class mix vs PAD, so that a leakage change is not only a label-mix change); NV-nearest fraction of
pad_heldout (share whose nearest ISIC class mean under the Mahalanobis metric is NV) and of ISIC test; AK recall and NV
predicted share on pad_heldout. Dose table: Mahalanobis AUROC with CI, rule A, Fitzpatrick, leakage, NV-nearest by
shift at each λ. Continuity: the same recipe's two-stream runs (current: R4 lesion-level runs) in a separate table.

## CI / bootstrap

`PREREGISTER_BOOTSTRAP.json` unchanged (2,000 draws, percentile 95%, draws shared across seeds within a cell; fresh
generator seeded 20261005 per cell and set). ID by lesion group, pad_heldout by patient, Fitzpatrick by image
(anti-conservative), P7a sets by group. Rule A per cell (reported for every λ and shift).

## Gates and controls (flag, never drop)

- Adversary not engaged: a λ > 0 cell whose mean Table 1 leakage is not ≥ 0.05 below the λ = 0 cell.
- No headroom: λ = 0 Mahalanobis pad_heldout in [0.45, 0.55].
- Controls: shift 100% (three-stream) vs the two-stream runs of the same recipe (does the stream split itself change
  anything); λ = 0 (no adversarial pressure).

## Predicted outcome (written now)

Work-order prediction recorded verbatim: **inversion weakens as shift → 0 and AUROC returns above 0.5 at 0%.**
Numbers: Mahalanobis pad_heldout at λ = 2 — shift 100%: 0.38–0.45, rule A met; 50%: 0.45–0.60; 0%: 0.55–0.75, rule A
not met, pooled CI entirely above 0.5. Same ordering at λ = 0.5. pad_heldout NV-nearest fraction decreases from 100% to
0%. Table 1 leakage at 0% within 0.05 of 100% (invariance pressure still applied). Fitzpatrick moves in the same
direction but less (it is not the adversary's target). Shift 100% three-stream within 0.05 of the two-stream runs.

## Outputs

`results/paperB/r6/r1/<recipe>/`: `runs/r1_<recipe>_shift<pct>_ladv<λ>_s<seed>/{config.json, summary.json}`,
`features/...`, `summary.csv`, `cells.json`, `REPORT.md`. Checkpoints `checkpoints/r6_r1/<recipe>/`.

## GPU-hour estimate and basis

Basis: 1.6 GPU-h per two-stream 40-epoch run (measured); three streams assumed ×1.5 (96 instead of 64 images per step)
= 2.4 GPU-h. 35 runs × 2.4 = **84 GPU-h** (current recipe, or a 40-epoch P8 recipe); **168 GPU-h** if P8 selects
80 epochs. CPU: report ≈ 1.5 CPU-h.
Cheaper variant not chosen: importance-weighting the ISIC rows in the adversary loss instead of resampling them
(two-stream, ≈ 56 GPU-h). It is not what the work order specifies and it equals IWDAN with oracle weights, which would
blur R1 into R2.

## Stop rules and budget caps

Cap 90 GPU-h (current / 40-epoch recipe), 175 GPU-h (80 epochs). If the first run exceeds 1.5× its estimate, the parent
re-estimates before the rest of the array. If a 96-image step does not fit in GPU memory, the run is stopped and
reported (no silent change of batch size).

## Pre-run amendment (user decisions 2026-10-15, before any R1 run)

1. λ = 0 is not retrained: each seed's λ = 0 row evaluates the existing λ = 0 checkpoint of the same recipe (current
   recipe: Table 1 lesion-level runs `checkpoints/r2_item4/runB_orth1_ladv0_s*`; P8 recipe: the P8 main λ = 0 runs).
   These were trained with the recipe's own stream layout (two streams for the current recipe), not the three-stream
   layout; at λ = 0 the adversary loss is zero, so the only difference is the extra ISIC stream feeding the
   context-domain branch. Summaries carry `lambda0_reused = true`.
2. The 50 % shift level runs with 3 seeds (42, 52, 62); 100 % and 0 % keep 5 seeds.
Jobs: 5 reused λ = 0 + λ ∈ {0.5, 2} × (5 + 5 + 3) = 31 (26 trained). Revised estimate: 26 × 2.4 + 5 × 0.2 =
**≈ 63 GPU-h** (cap 70).

## Leakage-matched test: label shift vs invariance depth

Committed 2026-10-16 12:51 (UTC+7), **before** the remaining R1 λ = 2 shift-100% seeds (tasks after 67547_5)
finish. Fixed now; not changed after those numbers are seen. Complements the within-R1 dose tables above; does not
replace them.

**Pool (lesion-level, seed-level points).** (i) R1 three-stream: shift ∈ {0%, 50%, 100%} × λ ∈ {0.5, 2}; (ii) Table 1
old recipe (`r2/item4`, all λ); (iii) A4 P8 main (`r5/p8/main`, λ ∈ {0, 0.5, 2}). Same **pad_full** leakage probe
(Table 1 definition: logistic domain probe on `z_lesion_norm`, ISIC test vs all of PAD) for every run. Held-out /
class-matched leakage probes are secondary only. R1 degraded runs (ID bal acc &lt; 0.25) are flagged and **excluded**
from (a) and (b); they may appear on the (c) plot marked as degraded.

**Sets.** Primary (count toward support): Fitzpatrick17k, MED-NODE, SD-198. **pad_heldout** (adversary target domain)
is reported separately and **does not count** toward the ≥ 2/3 rule. SCIN is not in this test’s primary set list
(still scored in the main R1 report).

**(a) Regression (attempted primary).** Per unseen set, seed-level OLS:
Mahalanobis AUROC ~ leakage + shift_level + recipe, where `shift_level` ∈ {0, 0.5, 1} for R1 and = 1 (100%) for
Table 1 and A4 (two-stream / no shift dose), and `recipe` is a categorical (R1 / Table1 / A4). CIs by wild cluster
bootstrap (Webb weights), clustered by **setting** = (recipe, λ, shift) cell; 2,000 draws, rng seed 20261005,
percentile 95%. **Identification:** the shift coefficient is identified from variation **within R1**; A4 and Table 1
only extend the leakage range at shift = 100%. State this in the report. Report VIF of shift vs recipe and of shift
vs leakage. If **either VIF &gt; 5**, report (a) as **not identifiable**, not as a null shift effect. **Supported** if
the shift coefficient is in the predicted direction (higher shift → lower AUROC) with CI excluding 0 on **≥ 2 of 3**
unseen primary sets.

**(b) Matched comparison (primary if (a) is not identifiable).** Cross-recipe, per unseen primary set: R1 shift 0%
λ = 2 (leakage ≈ 0.79) vs A4 λ = 2 (shift 100%, leakage ≈ 0.78). Cluster bootstrap of the mean AUROC difference
(same PREREGISTER settings / units as the main R1 report). Label explicitly **cross-recipe**. Supported if the
difference is in the predicted direction with CI excluding 0 on ≥ 2 of 3 unseen sets.

**(c) Plot.** Per set (including pad_heldout, marked as adversary-target / not counted): Mahalanobis AUROC vs
pad_full leakage, points coloured by shift level (recipe shape or marker secondary).

**Prediction (written now).** Label shift lowers Mahalanobis AUROC at matched leakage on the unseen primary sets. If
the shift effect is null (or (a) not identifiable and (b) null) and points fall on one leakage curve, report
**leakage / invariance depth** as the main variable, not adversary label shift.

Outputs under `results/paperB/r6/r1/<recipe>/leakage_matched/`: regression JSON, matched JSON, plots, and a section
in `REPORT.md`. CPU only after the pooled runs exist.

## Deviations

- 2026-10-16 10:05 (UTC+7), user-directed, after seed 52 at shift 0% λ = 2 finished with ID bal acc 0.139
  (chance = 1/8 = 0.125) and before further seeds are summarised. **Degraded-run rule for R1.** A run is degraded if
  its primary-checkpoint ID test balanced accuracy is &lt; 0.25 (2× chance). Degraded runs are reported and flagged and
  **excluded from the rule A verdict**; the verdict is also shown with them included. Separately, a run whose ID bal
  acc is more than 0.10 below the matched λ = 0 run (same recipe and seed) is flagged but **kept in the main verdict**.
  No reruns of failed seeds unless pre-specified: if more than 1 of 5 seeds degrade in a cell, run 2 replacement seeds
  92 and 102 for that cell only. Implementation: `scripts/r6_common.py` (`DEGRADED_ID_BAL`, `ID_DROP_FLAG`),
  `scripts/r6_report.py --item r1` (dual rule A; nearest-centroid class-share Δ table). Seed-52 diagnosis written under
  `results/paperB/r6/r1/cur/diag_s52_shift0_ladv2.json`.
- 2026-10-16 11:20 (UTC+7), before shift-100% λ = 2 cells are complete. **All unseen sets at every shift:** each
  R1 cell is scored on pad_heldout, Fitzpatrick17k, MED-NODE, SD-198, SCIN with the same five detectors and the same
  cluster/patient bootstraps (Fitzpatrick uses P13 `cluster_union` when present; else image). Report key comparison at
  λ = 2: R1 shift 0% vs R1's own shift 100% (`last.ckpt`), per set, plus shift 50% when available
  (`scripts/r6_report.py --item r1`). Predictions unchanged.
- 2026-10-16 12:51 (UTC+7), before remaining λ = 2 shift-100% seeds finish. **Leakage-matched test** (section above)
  pre-committed: pool R1 + Table 1 + A4; primary sets Fitz / MED-NODE / SD-198; (a) regression with VIF gate;
  (b) cross-recipe matched R1 shift0 λ2 vs A4 λ2; (c) AUROC–leakage plot. No analysis numbers computed at this commit.

