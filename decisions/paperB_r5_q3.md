# Paper B R5 / Q3 — does the λ pattern depend on the backbone?

Committed before any Q3 job runs. Fixed now; not changed after results are seen. Deviations are appended at the end.

## Question

Every published Paper B dual-encoder run uses EfficientNet-B3. Is the λ = 0 → λ = 2 pattern (leakage falls; pad_heldout
Mahalanobis AUROC from ≈ 0.81 to below 0.5; Fitzpatrick likewise) a property of the objective or of that backbone?

## Design (scripts/train_r5_dual_isic.py; scripts/train_r5_single_isic.py for ResNet-50)

- Dual encoder, backbone swapped in both branches (`r5_models.CSGAnyNet`: same 16-d `z_lesion_norm`, 64-d context
  latent, heads, GRL schedule, orthogonality, grey lesion path), ImageNet initialisation:
  - ConvNeXt-T (torchvision IMAGENET1K_V1, 768-d), and
  - DINOv2-S/14 (self-supervised ViT, 384-d CLS token; `facebookresearch/dinov2` hub repo, local copy). Fallback if the
    S/14 weights cannot be fetched: DINOv2-B/14 (cached), logged as a deviation.
  - Task adaptation fixed now: ViT backbones use a backbone LR multiplier of 0.1 (1e-5), the usual ViT fine-tuning
    setting; the projector, heads and classifier keep the main-task LRs. ConvNeXt-T uses the main-task LRs unchanged.
- ResNet-50 (missing from the lesion-level table: the existing ResNet-50 runs are image-level and the λ = 2 adversary
  never left ln 2): single-encoder lesion-level λ ∈ {0, 2} × 5 seeds through the P9 script, with P9's selected rung if
  P9 found one (`SELECTED_resnet50.json`), otherwise the baseline single-encoder recipe. These runs are written into
  `p9/stage2/` and shared with P9 stage 2 (no double run). The dual EfficientNet-B3 rows of R2 item 4 are the reference
  (not re-run).

## Data / splits

Lesion-level ISIC split (R2 item 4: train 16,211 / val 4,053 / test 5,067), PAD pad_adv / pad_heldout (patient split),
Fitzpatrick17k, and the P7a clinical sets (MED-NODE, SD-198, SCIN) when their set files exist.

## Seeds and λ grid

λ ∈ {0, 2} × seeds 42, 52, 62, 72, 82 per arm (the seeds of the published λ ∈ {0, 2} rows).

## OOD sets, detectors, metrics

Table 1 columns via `r5_common.eval_isic`: leakage (logistic probe on the scoring space, ISIC test vs PAD full),
ID balanced accuracy, ID ECE, OOD ECE pad_heldout, Mahalanobis AUROC pad_heldout (OOD AUROC), Fitzpatrick17k,
X-dom six-class balanced accuracy; kNN / cosine / MSP / Energy; NV-nearest fraction; per-class blocks; MED-NODE, SD-198,
SCIN Mahalanobis. Scoring space: `z_lesion_norm` (dual), BN'd pooled features (ResNet-50; PCA-16 probe also shown).

## Gates (flag, never drop)

Adversary gate per run (phase15_common.adversary_gate). Near-chance λ = 0 pad_heldout Mahalanobis (0.45–0.55) flags
"no headroom" for that arm. ID balanced accuracy at λ = 0 more than 0.05 below the EfficientNet-B3 reference (0.496)
flags "weaker base model" (reported next to the arm, not a reason to drop it).

## Controls

λ = 0 for each arm; the EfficientNet-B3 dual-encoder rows (R2 item 4) as the reference.

## CI / bootstrap and rule A

PREREGISTER_BOOTSTRAP.json unchanged (2,000 draws, rng seed 20261005, percentile 95 %, draws shared across seeds). ID by
lesion group, pad_heldout by patient, Fitzpatrick / MED-NODE / SD-198 by image, SCIN by case. Rule A at λ = 2: pooled
CI entirely below 0.5 and ≥ 3 of 5 per-seed CIs entirely below 0.5 (Mahalanobis). Reported for every cell.

## Reported cells

Per arm × λ: Table 1 columns (mean ± SD), bootstrap rule on pad_heldout and Fitzpatrick, the extra OOD sets, kNN,
NV-nearest fraction; LaTeX rows; one plain paragraph per arm.

## Predicted outcome (written before any score exists)

- Dual ConvNeXt-T: λ = 0 leakage 0.90–0.96, pad_heldout Mahalanobis 0.75–0.85; λ = 2 leakage 0.55–0.70, pad_heldout
  Mahalanobis 0.38–0.48 with rule A; Fitzpatrick below 0.5.
- Dual DINOv2-S/14: the same direction; λ = 0 Mahalanobis higher (0.80–0.90, stronger generic features); λ = 2 below 0.5
  in mean, rule A or B (more seed variance with a frozen-ish ViT at LR 1e-5).
- Single ResNet-50 (baseline config): λ = 2 leakage ≥ 0.90 and pad_heldout Mahalanobis ≥ 0.5 (no inversion without
  invariance). If P9 found a rung reaching leakage ≤ 0.60, that rung's λ = 2 Mahalanobis falls below 0.5.

## Outputs

`results/paperB/r5/q3/runs/<tag>/` (config.json, adversary_log.jsonl, summary.json, scores.npz, features/),
`results/paperB/r5/p9/stage2/single_resnet50_*` (ResNet-50 cells), `results/paperB/r5/q3/{summary.csv, BOOTSTRAP.json,
REPORT.md}`. Checkpoints `checkpoints/r5_q3/`. `--recipe p8` runs carry `_recp8` and report to `q3/recipe_recp8/`.

## Weights

Cached: ConvNeXt-T (`convnext_tiny-983f1562.pth`), DINOv2-B/14, ResNet-50. Not cached: DINOv2-S/14. Fetch once on the
login node before launch (not done at this commit):
`.venv/bin/python -c "import torch; torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth')"`.
The launcher refuses to submit while a needed weight file is missing.

## GPU-h estimate and basis

Basis: dual EfficientNet-B3 lesion-level run ≈ 1.6 GPU-h; ConvNeXt-T has ≈ 4.5 GFLOPs per 224² image vs ≈ 1.8 for
EfficientNet-B3, partly offset by fewer memory-bound depthwise layers under AMP; data loading is shared: ≈ 2.2 GPU-h
per run; DINOv2-S/14 ≈ 4.6
GFLOPs: ≈ 2.4 GPU-h; DINOv2-B/14 fallback ≈ 3.5 GPU-h. ResNet-50 single ≈ 1.2 GPU-h.

| Arm | Runs | GPU-h |
|---|---:|---:|
| Dual ConvNeXt-T | 10 | 22 |
| Dual DINOv2-S/14 (B/14 fallback) | 10 | 24 (35) |
| Single ResNet-50 (0 if P9 stage 2 for ResNet-50 already ran) | 10 | 12 |
| Total | 30 | ≈ 58 (≈ 69 with the fallback) |

CPU: report ≈ 0.5 h (bootstrap over 6 cells × 5 OOD sets).

## Stop rules and caps

No adaptive stopping; 30 runs. A failed run is re-run once with the same seed; a second failure is reported. Out of
memory on a ViT arm: batch kept at 32 + 32 and gradient checkpointing is not added; the arm is reported as not run.

## Pre-run amendment (user decision 2026-10-15, before any Q3 run)

The dual ConvNeXt-T arm is dropped. Reason given by the user: the paper already has ResNet-50 (single encoder) as a
second CNN; the question a reviewer will ask is whether the pattern holds for a transformer, which ConvNeXt does not
answer. Arms: dual DINOv2-S/14 (B/14 fallback) and single ResNet-50, λ ∈ {0, 2} × 5 seeds = 20 runs. Revised estimate
**≈ 36 GPU-h (≈ 47 with the B/14 fallback)**. DINOv2-S/14 weights prefetched to the torch hub cache on 2026-10-15.

## Deviations
