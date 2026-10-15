# Paper B R5 / P9 — can a single encoder reach the dual encoder's invariance? (ResNet-50, EfficientNet-B3)

Committed before any P9 job runs. Fixed now; not changed after results are seen. Deviations are appended at the end.

## Question

The single-encoder DANN baselines did not reach the dual encoder's leakage: image-level single EfficientNet-B3 0.97 →
0.76–0.81 (Phase 15.1), ResNet-50 0.965 / 0.939 / 0.936 at λ = 0 / 0.5 / 2 (R1 item 5; λ = 2 adversary gate
"inconclusive": adversary CE ≈ 0.686, the head never left ln 2). The dual encoder reaches 0.555–0.66 at λ > 0 on the
lesion-level split. Is the gap a property of the architecture or of an under-powered adversary? We apply the standard
invariance toolbox to the single encoder until its leakage reaches the dual encoder's level, or a 20 GPU-h cap.

## Existing code reused

`scripts/train_reviewer_r1_item5_resnet50.py` / `train_phase15_1_single_dann.py` define the single-encoder recipe
(backbone → BN → linear classifier; derm domain head on GRL(z); adversary LR ×30; current GRL schedule with floor 0.2).
P9 re-implements this recipe in `r5_models.SingleEncToolboxLit` (config `baseline` reproduces it: same loss placement,
same optimiser groups) and adds the toolbox switches; nothing in the old scripts changes.

## Design (scripts/train_r5_single_isic.py)

Lesion-level split (R2 item 4) and PAD patient split, as Table 1. Domain-balanced batches are already the default
(one ISIC + one PAD image per index, 32 + 32). Ladder at λ = 2, seed 42, rungs cumulative, run in order, one GPU per
backbone:

| Rung | Adds |
|---|---|
| baseline | Phase 15.1 / item-5 recipe on the lesion-level split (reference) |
| T1_ganin_warmup | Ganin & Lempitsky α schedule (α = 0 at step 0, γ = 10, linear progress) + 2 adversary warm-up epochs (heads train on z.detach(); no reversed gradient) |
| T2_bank_reinit | 3 heads (linear = the probe's family, derm 2-layer, 3-layer), loss averaged; 2 head-only inner steps per batch on z.detach() (fp32, outside AMP); one head re-initialised every 500 steps (round robin, Adam state cleared) |
| T3_strong | 5 heads (2 linear, derm, 2× wide derm, 3-layer); 5 inner steps; head LR ×10 instead of ×30 (more steps, smaller steps) |
| T4_long | T3 at twice the recipe's epochs (80) |

Stop rule, checked by each job before it trains: skip if an earlier rung of the same backbone reached Table 1 leakage
≤ 0.60, or if the GPU-hours already used by P9 ladder runs (both backbones) + this run's estimate exceed 20. Skips are
written as `skipped.json` and reported.

Selection: the first rung (ladder order) with leakage ≤ 0.60 is written to `SELECTED_<backbone>.json`.

Stage 2 (conditional; outside the 20 GPU-h cap; needs user approval): the selected rung × λ ∈ {0, 0.5, 2} × seeds
42, 52, 62, 72, 82, full Table 1 evaluation and bootstrap. The Q3 ResNet-50 cells (λ ∈ {0, 2}) are a subset of the
ResNet-50 stage 2 and share its run directories.

## Data / splits, seeds, λ grid

ISIC lesion-level train 16,211 / val 4,053 / test 5,067; PAD pad_adv / pad_heldout (716 images / 412 patients);
Fitzpatrick17k; MED-NODE / SD-198 / SCIN sets from P7a when present. Ladder: λ = 2, seed 42. Stage 2: λ ∈ {0, 0.5, 2}
× 5 seeds.

## Metrics, OOD sets

Table 1 columns (r5_common.eval_isic): leakage (logistic probe ISIC test vs PAD full on the scoring space, 70/30, probe
seeds 42/52/62, balanced accuracy), ID balanced accuracy, ID ECE, OOD ECE pad_heldout, Mahalanobis AUROC pad_heldout,
Fitzpatrick17k, X-dom six-class balanced accuracy; plus kNN, cosine, MSP, Energy; per-class and nevus-nearest blocks.
Scoring space = the BN'd pooled features (2,048-d ResNet-50, 1,536-d EfficientNet-B3). Because a linear probe on 2,048
dimensions is stronger than on the dual encoder's 16, the same probe on the 16 leading ISIC-train principal components
is reported next to it (secondary; the stop rule uses the primary probe). Selection-side probe ISIC val vs pad_adv
reported. Adversary log per epoch: head CE / accuracy, inner-step CE, number of re-inits; adversary gate status.

## Gates (flag, never drop)

Adversary gate per run (phase15_common.adversary_gate). A rung whose ID balanced accuracy falls more than 0.05 below
the baseline rung is flagged "classifier degraded" (it still counts for the stop rule, which is about leakage only).

## Controls

Baseline rung (same split, λ = 2, no toolbox) and, in stage 2, λ = 0.

## CI / bootstrap

Stage 2 only (ladder is one seed): PREREGISTER_BOOTSTRAP.json; ID by lesion group, pad_heldout by patient,
Fitzpatrick by image; rule A at λ = 2.

## Reported cells

Ladder table per backbone (rung, leakage, PCA-16 leakage, selection probe, ID bal acc, pad_heldout and Fitzpatrick
Mahalanobis, adversary gate, best / last head CE, last inner CE, GPU-h); skips; cumulative GPU-h. Stage 2: Table 1
rows per λ with bootstrap rules, LaTeX rows.

## Predicted outcome (written before any score exists)

- Baseline rung (lesion-level, λ = 2): leakage 0.92–0.95 for both backbones (as image-level).
- T1: ≤ 0.03 lower. T2 (linear head + inner steps; the head can now follow the encoder): the largest step, to
  0.75–0.88. T3 / T4: a further 0.00–0.05.
- Neither backbone reaches 0.60 within 20 GPU-h. The PCA-16 probe is 0.05–0.15 lower than the full-dimension probe.
- ID balanced accuracy within 0.05 of the baseline rung for T1–T3; T4 may gain (longer training).
- If a rung does reach 0.60, its pad_heldout Mahalanobis AUROC at λ = 2 falls below 0.5 as for the dual encoder.

## Outputs

`results/paperB/r5/p9/ladder/<tag>/`, `stage2/<tag>/` (config.json, adversary_log.jsonl, summary.json, scores.npz,
features/); `summary.csv`, `SELECTED_<backbone>.json`, `BOOTSTRAP_stage2.json`, `REPORT.md`. Checkpoints in
`checkpoints/r5_p9/`. `--recipe p8` runs carry the `_recp8` suffix; their report goes to `p9/recipe_recp8/`; recipe
keys that do not apply to a single encoder (latent sizes, λ_orth, λ_ctx) are listed as not applicable in config.json.

## GPU-h estimate and basis

Basis: lesion-level ResNet-50 / single EfficientNet-B3 run ≈ 1.0 / 1.07 GPU-h (R1 item 5, Phase 15.1); inner steps and
extra heads act on a 2,048-d vector (negligible); evaluation adds ≈ 0.1 h. Ladder per backbone ≈ 1.1 + 1.2 + 1.3 +
1.4 + 2.6 ≈ 7.6 (ResNet-50) and ≈ 8.2 (EfficientNet-B3): ≈ 16 GPU-h if no rung stops early; hard cap 20.
Stage 2 (conditional): 15 runs × ≈ 1.2 ≈ 18 GPU-h per backbone. CPU: report ≈ 0.2 h (≈ 0.5 h with stage-2 bootstrap).

## Stop rules and caps

As above: per-backbone stop at leakage ≤ 0.60; 20 GPU-h cap over the ladder; stage 2 only with a SELECTED file and the
user's approval. A failed run is re-run once; a second failure is reported.

## Deviations

## Amendment 2026-10-15 — user-directed redesign (written before any P9 job has run)

This section replaces the Design, ladder, selection, stage 2, stop-rule, GPU-h and prediction sections above. Nothing
above has run; the earlier text is kept only as a record. GPU order (user): P8 → R1 → R2 → P9 → P7(b) → Q1 → rest.

**Hypothesis.** The single encoder does not invert on pad_heldout because it does not reach the dual encoder's
invariance (Table 1 leakage 0.555 vs 0.76); inversion needs that deeper invariance.

**Backbone.** EfficientNet-B3 only (the dual encoder's backbone and the tab:single single encoder). The ResNet-50 arm
is dropped from P9. Recipe: `--recipe auto` (p8 if RECIPE_DECISION.json says competent, else current), the same for
every P9 run including the λ = 0 control.

**Invariance measure (primary).** The P6/A3 held-out probe (P6 variant V1): logistic probe (StandardScaler,
max_iter 4000) on ISIC test ∪ pad_heldout, i.e. only images the adversary never saw, 70/30 split grouped by lesion
(ISIC) / patient (PAD), probe seeds 42/52/62, balanced accuracy. For the single encoder, it is applied to the scoring
space (BN'd pooled features, 1,536-d). For the dual encoder, it is applied to z_lesion_norm. Reported next to it: the
same probe on the single encoder's 16 leading ISIC-train principal components, on the dual encoder's
backbone_raw_lesion (1,536-d, dimension-matched), and the Table 1 pad_full probe.

**Target.** T = (mean over the 5 dual-encoder λ = 2 seeds of the V1 probe on z_lesion_norm, from
`results/paperB/r5/p6/per_run/`) + 0.05. It is written to `results/paperB/r5/p9/v2/TARGET.json` by
`r5_p9v2.py target` before any P9 summary exists; the job refuses to write it if a P9 summary is already present.
A configuration "reaches the target" when the mean V1 over its seeds is ≤ T.

**Toolbox and pilot (1 seed, 42).** Every adversarial config uses the Ganin & Lempitsky λ warm-up schedule (α = 0 at
step 0, γ = 10, linear progress) plus 2 adversary warm-up epochs (heads train on z.detach()), one adversary head, no
inner steps, no re-init.
- Round 1 (9 runs): λ ∈ {2, 4, 8} × adversary LR ×{10, 30, 100}, head depth 2 (the recipe's head).
- λ = 0 control (1 run): same recipe, no adversary.
- Round 2 (6 runs): for each λ, the round-1 LR with the lowest V1 among runs that are not flagged as degraded (all
  flagged: lowest V1; ties: smaller LR), at head depth 1 and 3. Written to ROUND2.json before round 2.

**Best config.** Among the 15 adversarial pilot runs that are not flagged as degraded, the one with the lowest V1
(ties: smaller λ, then smaller LR, then shallower head). If every run is flagged, the lowest V1 overall, carrying the
flag. Written to BEST.json before seed runs.
- 3 seeds: best config at seeds 52 and 62, plus the λ = 0 control at seeds 52 and 62.
- 5 seeds (only if the 3-seed mean V1 ≤ T): best config at seeds 72 and 82. Otherwise NO_GO5.json is written and
  the outcome is "untested".

**Degraded flag.** For every run: ID balanced accuracy more than 0.05 below the λ = 0 control (the mean over the
available λ = 0 seeds) is flagged "degraded". ID balanced accuracy and cross-domain accuracy (six-class pad_heldout
balanced accuracy) are logged for every run.

**Inverts (as specified by the user; stricter than the shared rule A in r5_common, which needs ≥ 3/5 seeds and
Mahalanobis only).** On pad_heldout with the patient-clustered bootstrap (PREREGISTER_BOOTSTRAP.json: 2,000 draws,
ID by lesion group, PAD by patient, draws shared across seeds; r5_common.isic_bootstrap), all three must hold:
- the Mahalanobis 5-seed-mean CI upper bound is < 0.5;
- at least 4 of 5 seeds have their own Mahalanobis CI upper bound < 0.5;
- at least 3 of the 5 detectors (Mahalanobis, kNN k = 50, cosine, MSP, Energy) have a 5-seed-mean CI upper bound
  < 0.5.

Otherwise the verdict is "collapse" if the Mahalanobis 5-seed mean is within [0.45, 0.55], else "no collapse". A
verdict is assigned only at 5 seeds; pilot and 3-seed cells show points and CIs only.

**Outcomes (fixed now).**
- Target reached at 5 seeds, config not flagged, and the model inverts: consistent with the hypothesis.
- Target reached at 5 seeds, config not flagged, and the model does not invert: the hypothesis "inversion needs deep
  invariance" is refuted.
- Target reached only by a flagged (degraded) config: reported as "reached with a degraded classifier"; it neither
  supports nor refutes.
- Target not reached (3-seed mean > T, or the cap stops the sequence first): untested, not support.

**Cap: 30 GPU-h** over all P9 runs, including the λ = 0 control. Each job sums `gpu_hours` from the P9 summaries
written so far, adds its own estimate (1.2 GPU-h × epochs / 40), and skips (writing skipped.json) if the total would
exceed 30. Plan: pilot 16 × 1.2 ≈ 19.2, 3 seeds 4 × 1.2 ≈ 4.8, 5 seeds 2 × 1.2 ≈ 2.4, total ≈ 26.4 GPU-h, leaving
about 3.6 for one failed rerun. CPU: about 0.5 h (probes and bootstrap).

**Predictions (written before any P9 score exists).**
1. Dual-encoder V1 at λ = 2: 0.56–0.64, so T ≈ 0.61–0.69.
2. Single-encoder λ = 0 V1: 0.90–0.97. Best pilot V1: 0.70–0.85, below the Table 1 image-level 0.76–0.81 only
   modestly. The largest gains come from adversary LR ×100 and λ = 8. Head depth changes V1 by ≤ 0.03.
3. The target is not reached, so the hypothesis remains untested (most likely outcome, ~65%).
4. At least one of the λ = 8 runs with LR ×100 is flagged as degraded.
5. Best-config pad_heldout Mahalanobis (point, any number of seeds): 0.48–0.60.
6. Hypothesis-side prediction (user): if the single encoder reaches V1 ≤ T, it inverts on pad_heldout.
   My own expectation, conditional on the target being reached: inversion is less likely than not (~40%), because the
   single encoder has no separate context branch to absorb the PAD-specific variance.

**Reported cells.** A pilot table: config, λ, LR, depth, V1 (and the probe-seed SD), PCA-16 V1, Table 1 pad_full
probe, ID balanced accuracy, cross-domain accuracy, flag, adversary gate, pad_heldout Mahalanobis / kNN / cosine /
MSP / Energy with the single-seed patient-clustered CI, Fitzpatrick Mahalanobis, and GPU-h. Also a seed table (best
config and λ = 0 at 3 or 5 seeds) with mean CIs per detector, per-seed Mahalanobis CIs and the verdict, a dual-encoder
comparison row (λ = 0, λ = 2) under the same probes, the target, the outcome line, cumulative GPU-h and LaTeX rows.
Outputs go to `results/paperB/r5/p9/v2/` (runs in `v2/<tag>/`, plus TARGET.json, ROUND2.json, BEST.json, GO5.json or
NO_GO5.json, summary.csv and REPORT.md).

**Q3 consequence.** Q3's ResNet-50 cells used config `selected` from the ResNet-50 ladder, which no longer exists. Q3
falls back to the `baseline` config (recorded in Q3's decisions when Q3 is launched).
