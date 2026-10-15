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
