# Paper B R5 / P12 — stronger frozen encoders as the monitor (mitigation target on pad_heldout)

Committed before any P12 extraction or score (no P12 output exists; no weights downloaded). Fixed now; deviations appended.

## Question

The frozen ImageNet ResNet-50 monitor reached Mahalanobis 0.833 on pad_heldout (target ≥ 0.90; R2 item 1) and 0.988 on
Fitzpatrick17k. Does a stronger frozen encoder (DINOv2-B/L, a dermatology foundation model) reach the target on both?

## Encoders (availability and licences checked from files / model cards, nothing downloaded)

| Encoder | Status | Weights / size | Licence |
|---|---|---|---|
| ImageNet ResNet-50 (reference) | local (torchvision cache) | 98 MB | BSD-3 (torchvision) |
| DINOv2 ViT-B/14 | local (`~/.cache/torch/hub/checkpoints`, hub repo `facebookresearch_dinov2_main`) | 330 MB | Apache-2.0 |
| DINOv2 ViT-L/14 | local (same) | 1.1 GB | Apache-2.0 |
| MONET (Kim et al., Nat Med 2024; HF `suinleelab/monet`) | downloadable, ungated | 1.71 GB safetensors (CLIP ViT-L/14) | model card states no licence; intended use research only → usable for a research comparison with that caveat; the user decides before download |
| PanDerm (Yan et al. 2025) | not on the cluster; Google Drive link | ViT-L, ≈ 1.2 GB | CC BY-NC-ND 4.0 (non-commercial research allowed); needs the PanDerm repository's model code; loader not wired here |
| DermLIP | gated (auto-approve) | — | CC BY-NC-ND; needs `open_clip` (not installed in the venv) |
| Google Derm Foundation | gated | — | Health AI Developer Foundations terms; needs acceptance by the user |

Primary set for this item: ImageNet ResNet-50, DINOv2-B/14, DINOv2-L/14 (all local). MONET is run only after the user
accepts the no-licence caveat (separate sbatch index); PanDerm only if the user obtains the weights.

## Design (code: `scripts/r5_frozen_frames.py` (CPU, metadata), `scripts/r5_frozen_extract.py` (GPU-light),
`scripts/r5_p12_frozen_encoders.py` (CPU scoring, reuses `r5_p3_frozen_monitor.score_frozen` / `report`))

Frames: lesion-level ISIC train / test, pad_full, pad_heldout, Fitzpatrick17k (sorted), same row order as the model
features. Preprocessing: ResNet-50 Resize 256 / CenterCrop 224 / ImageNet norm; DINOv2 bicubic Resize 256 / CenterCrop
224 / ImageNet norm, CLS token after the final norm (primary) and patch-token mean (sensitivity); MONET CLIP
preprocessing (bicubic 224, CLIP mean/std), `get_image_features`. RGB input for all frozen encoders (the model uses
grayscale; the monitor is a separate system). Detectors: Mahalanobis class-cond (pooled + 1e-3·I; pre-registered),
Ledoit–Wolf Mahalanobis, kNN k=50 cosine-normalised; fit on ISIC lesion-level train.

## CI

As P3: percentile 95%, 2,000 draws, rng 20261005; ID by lesion group, pad_heldout by patient, Fitzpatrick by image and
by P13 near-duplicate cluster when available.

## Target and reporting

Target (R2 pre-registration, unchanged): Mahalanobis ≥ 0.90 on pad_heldout AND on Fitzpatrick17k. Reported as numbers
next to the target for each encoder; the model's λ = 0 / 2 numbers on the same ID test alongside.

## PREDICTED OUTCOME (written now)

- ResNet-50 (full lesion-level frames): pad_heldout 0.81–0.86, Fitzpatrick 0.97–0.99.
- DINOv2-B/14 CLS: pad_heldout 0.86–0.93, Fitzpatrick ≥ 0.98. DINOv2-L/14 CLS: pad_heldout 0.88–0.95, Fitzpatrick ≥ 0.98.
  At least one DINOv2 variant reaches ≥ 0.90 on both sets (my confidence ≈ 60 %). Patch-mean ≤ CLS by 0–0.03.
- MONET (if run): pad_heldout 0.80–0.92 (trained on clinical + dermoscopy images, so PAD photos may look less foreign).
- kNN ≥ Mahalanobis by 0.02–0.08 on pad_heldout for every encoder (ResNet-50: 0.913 vs 0.833).

## Outputs

`results/paperB/r5/frozen/{frames,features/<encoder>}/`, `results/paperB/r5/p12/` (`results.json`, `summary.csv`, `REPORT.md`).

## Cost

GPU (light): inference on ≈ 28k images per encoder — ResNet-50 ≈ 0.15 GPU-h, DINOv2-B ≈ 0.2, DINOv2-L ≈ 0.5, MONET ≈ 0.5
(basis: ViT-L/14 at 224 px ≈ 150–250 img/s on one GPU with 8 loader workers; JPEG decoding dominates). Total ≈ 0.9 GPU-h
(1.4 with MONET). CPU scoring ≈ 1 h on 8 CPUs (1,024–2,048-d covariances, 2,000-draw CIs per cell).

## Stop rules

GPU cap 2 GPU-h for this item. No download beyond MONET (1.7 GB) without the user's approval.

## Pre-run amendment (user decision 2026-10-15, before any P12 run)

MONET is included (array index 3 of `slurm/r5_frozen_extract.sbatch`, now `--array=0-3%2`). Licence note for the
report: the `suinleelab/monet` model card states no licence and describes research use; it is used here for a research
comparison only and the weights are not redistributed. Weights prefetched to the Hugging Face cache on 2026-10-15. The
project venv cannot import `transformers` (it requires torch ≥ 2.5; the venv has 2.1.1), so the MONET index runs under
the `benhda` conda env (torch 2.4.1, transformers 4.46); the other encoders are unchanged. GPU ≈ 1.4 GPU-h.

## Deviations
