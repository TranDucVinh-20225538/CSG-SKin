# Paper B R5 / P5 — single-encoder GRL at λ = 4: n, per-detector AUROC with bootstrap CI, seeds below 0.5

Committed before any P5 run (no P5 feature or score exists at this commit). Fixed now; deviations appended.

## Question

The text cites the single-encoder GRL control (Phase 15.1) at λ = 4. State n, every detector's AUROC with a bootstrap
CI, and how many seeds lie below 0.5.

## What exists (read from files only)

`results/paperB/phase15/single_dann/single_dann_ladv{0,0p25,0p5,1,2,4,8}_s*`: 25 runs, EfficientNet-B3 single encoder
with a GRL head (`train_phase15_1_single_dann.py`), trained on the IMAGE-level ISIC split (Phase 2 `isic_splits`), not
the lesion-level split. λ = 4 has seeds 42/52/62 only (n = 3). No cached features; checkpoints in `summary.json`
`best_checkpoint`. Stored pad_heldout Mahalanobis at λ = 4: 0.482 / 0.522 / 0.534 (seeds 42/52/62); leakage (PAD-full
probe) 0.79–0.81; Fitzpatrick17k not stored.

## Design (code: `scripts/r5_p5_single_encoder.py`, `slurm/r5_p5_extract.sbatch`)

Stage 1 (CPU, loads checkpoints, reads images): `SingleDannLightning.load_from_checkpoint` → `collect_all` on isic_train,
isic_test (image-level split the runs were trained on), pad_full, pad_heldout, Fitzpatrick17k; saves z (the feature the
Phase 15.1 detectors use), logits, labels. Default: the 3 λ = 4 runs; optional all 25 (feeds Q4/R4).
Stage 2 (CPU): five detectors (Mahalanobis class-cond, kNN k=50, cosine on z; MSP, Energy T=1 on logits), fit on ISIC
train only; pipeline check against the stored `summary.json` pad_heldout values (|Δ| reported; > 0.005 flagged).

## CI / bootstrap

PREREGISTER_BOOTSTRAP.json unchanged: 2,000 draws, rng 20261005, percentile 95%, shared draws across seeds. ID resample
unit = image (as pre-registered for the image-level split; primary) and lesion group (sensitivity, from
`isic_image_split_with_lesion.csv`); pad_heldout by patient; Fitzpatrick by image. Reported per detector and OOD set:
mean ± s.d., pooled CI, per-seed points and CIs, k/3 seeds with CI entirely below 0.5, k/3 with point below 0.5.

## Seeds, λ, OOD sets

λ = 4, seeds 42/52/62 (n = 3; reported as n = 3, no seeds added by P5). All λ listed as context. OOD: pad_heldout,
Fitzpatrick17k.

## Gates (flag, never drop)

n = 3 < 5 → every λ = 4 cell carries "n = 3". Image-level split → flagged "image-level split, not comparable to Table 1
without the split caveat".

## PREDICTED OUTCOME (written now)

pad_heldout Mahalanobis 0.513 ± 0.03 (stored values), pooled CI ≈ [0.47, 0.56] straddling 0.5; 0/3 seeds with CI below
0.5, 1/3 with point below 0.5; the other four detectors 0.47–0.54. Fitzpatrick17k Mahalanobis 0.50–0.70. Reading: the
single-encoder GRL at λ = 4 sits at chance, not below it.

## Optional GPU (not part of this item's budget)

Two extra λ = 4 seeds (72, 82) to reach n = 5: 2 × ≈ 1.6 GPU-h ≈ 3.2 GPU-h (`train_phase15_1_single_dann.py`). Only on
the user's request.

## Outputs

`results/paperB/r5/p5/`: `features/<run>/*.npz`, `summary.csv`, `results.json` (per-run and bootstrap), `REPORT.md`.

## Cost

GPU 0. CPU: extraction ≈ 1 h per run on 8 CPUs (≈ 28k images through EfficientNet-B3 at 224 px), 3 runs in parallel
≈ 1 h wall (all 25: ≈ 5 h wall at 5 parallel); scoring + bootstrap ≈ 0.3 h.

## Stop rules

If the pipeline check differs from the stored value by > 0.02 on any seed, stop and report before bootstrapping.

## Deviations
