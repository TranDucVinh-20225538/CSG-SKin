# Paper B — Phase 1.5 report

CPU + one short GPU job. No retraining. Closes the attack surface on `z_context` Maha AUROC > 0.9999 (not 1.0000 ± 0.0000).

## Verdict

**The ISIC↔PAD Mahalanobis OOD protocol is not trivially solvable by the five hand-crafted feature sets.**

Under the identical Phase 1 protocol (class-conditional Mahalanobis, fit on ISIC train only, scored on ISIC test vs PAD):

| Feature set | Dim | Maha AUROC | AUPR-in | AUPR-out | FPR@95 |
|---|---:|---:|---:|---:|---:|
| rgb_moments | 6 | 0.5059 | 0.6859 | 0.3285 | 0.9548 |
| color_hist | 48 | 0.4569 | 0.6538 | 0.2997 | 0.9664 |
| gray_downsample | 256 | 0.4972 | 0.7060 | 0.2954 | 0.9238 |
| hsv_moments | 6 | 0.5634 | 0.7388 | 0.3795 | 0.9019 |
| laplacian_stats | 4 | **0.6474** | 0.8143 | 0.3793 | 0.7953 |

Max trivial Maha AUROC = **0.6474** (laplacian). None reach 0.95. None reach ~0.85.

Supervised logistic regression (ISIC vs PAD, 70/30 stratified, 5 seeds) on the same features is the domain-separability *ceiling*, not the OOD protocol:

| Feature set | Bal acc | Logistic AUROC |
|---|---:|---:|
| rgb_moments | 0.7527 ± 0.0073 | 0.8650 ± 0.0083 |
| color_hist | 0.7498 ± 0.0103 | 0.8515 ± 0.0074 |
| gray_downsample | 0.6904 ± 0.0103 | 0.8096 ± 0.0072 |
| hsv_moments | 0.7697 ± 0.0083 | 0.8702 ± 0.0056 |
| laplacian_stats | 0.7459 ± 0.0067 | **0.8761 ± 0.0045** |

Max logistic AUROC = **0.8761 < 0.95**. The ≥0.95 bar that would make the benchmark a colour/exposure test is not met. The logistic ceiling sits in 0.81–0.88, i.e. around the ~0.85 line in the work order, and well below `z_context` (> 0.9999).

So: low-level colour, exposure and sharpness *do* carry domain signal (logistic **plain acc** ~0.81 vs majority **0.688**; **balanced acc** 0.69–0.77 vs floor **0.5**; AUROC 0.81–0.88). They do **not** reproduce the Phase 1 Maha number. `z_context` (> 0.9999) is not a 6-d RGB-moment detector. Do not compare AUROC to the 0.688 accuracy majority.

Raw-source comparison (resize-to-224 only, no soft-crop): **BLOCKED**. PAD original PNGs are mode 600 / owner `anhnv`; this user cannot read them. Delta (processed − raw) cannot be computed. No claim is made about whether the soft-crop pipeline injects or removes domain signal.

## 1.5b — Separation-margin diagnostics

`runB_orth1` seed 42, checkpoint `best-36.ckpt`. Higher score = more OOD.

### z_context (domain-supervised, 64-d)

| | n | min | max | mean | std | p01 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ID (ISIC test) | 5067 | 16.58 | 1015.93 | 47.42 | 26.71 | 22.58 | 109.68 |
| OOD (PAD) | 2298 | 876.91 | 9398.10 | 3405.46 | 670.38 | 2026.80 | 5237.59 |

- `min_OOD − max_ID` = **−139.02** (ranges overlap).
- Overlap is a **tail of 1 ID + 2 OOD images**: one ID score at 1015.93 and two OOD scores at 876.91 / 881.75. Bulk: ID p99 = 109.7 vs OOD p01 = 2026.8.
- Maha AUROC on this seed = 0.99999983 (**2 discordant pairs / 11.6M**). Report as **AUROC > 0.9999**, never 1.0000 ± 0.0000. Ranking is essentially perfect; it is **not** a hard gap with a threshold that separates every sample.
- PCA: **k=1 already gives Maha AUROC = 1.0 and PC1 AUROC = 1.0** (84.5% variance). The monitor is a one-dimensional "domain-ness" scalar. That is a caveat (the 64-d context space is not being used as 64-d) and an interpretability result.

| k | Maha AUROC | var explained |
|---:|---:|---:|
| 1 | 1.0000 | 0.845 |
| 2 | 1.0000 | 0.889 |
| 4 | 1.0000 | 0.925 |
| 8 | 1.0000 | 0.958 |
| 16 | 1.0000 | 0.978 |
| 32 | 1.0000 | 0.992 |
| 64 | 1.0000 | 1.000 |

### z_lesion (unsupervised, 16-d) — contrast

| | n | min | max | mean | std |
|---|---:|---:|---:|---:|---:|
| ID | 5067 | 0.89 | 413.79 | 19.71 | 20.48 |
| OOD | 2298 | 0.52 | 427.73 | 16.72 | 23.99 |

Maha AUROC 0.381 on this seed (Phase 1 n=5: 0.4089). Full overlap. PCA k=1 AUROC 0.52; k=16 AUROC 0.366. No 1-d domain axis.

## 1.5c — Preprocessing-artifact audit

Eval path is **byte-identical** across domains after soft-crop:

- Soft-crop: PIL BILINEAR → 224×224 JPEG quality=95, both domains.
- Processed files: ISIC 25331/25331 and PAD 2298/2298 are 224×224 RGB JPEG.
- Eval: `Resize(256, bilinear) → CenterCrop(224) → ToTensor → ImageNet Normalize` on already-224 images (upsample then recrop), same for both.
- Lesion branch at eval: in-model `_rgb_to_gray3` after ImageNet norm (`x_lesion=None`). The METHODS `Grayscale` transform is dead code.

Asymmetries (not fixed):

1. **Source encoding.** ISIC sources are JPEG; PAD-UFES sources are PNG. Soft-crop re-encodes both to JPEG q=95. A detector on *processed* pixels is not reading PNG vs JPEG, but a detector on *raw* files would be.
2. **Source resolution (ISIC, readable).** Mix of native sizes: 1024×1024 (12414), 600×450 (10015), 1024×680 (1121), plus a long tail. Aspect mean 1.18 ± 0.18 (min 0.75, max 1.55). After soft-crop this is gone.
3. **Source resolution (PAD).** Unreadable (mode 600). Cannot test whether PAD native resolution/aspect differs systematically from ISIC. A detector *could* be reading the preprocessing pipeline rather than modality; that hypothesis is open until PAD originals are readable.
4. **PAD PNG → JPEG q=95** is an encode step ISIC (already JPEG) also goes through, but from a different codec history.

## 1.5d — Preview curve (correlational)

`results/paperB/figures/preview_leakage_vs_ood.pdf` (+ png, 300 dpi).

Four heterogeneous points. **Not an intervention.** Phase 3 `λ_adv` sweep is the controlled result.

| Point | Leakage | OOD AUROC | ID Bal Acc |
|---|---:|---:|---:|
| z_context | 0.9997 | > 0.9999 | — |
| ResNet-50 baseline | 0.979 | 0.868 | 0.655 |
| EffB3 single control | 0.802 | 0.726 | 0.683 |
| z_lesion | 0.724 | 0.409 | 0.702 |

OLS: slope = 1.774, intercept = −0.804, Pearson r = 0.942 (n=4). Bootstrap (2000, skip degenerate resamples, n_boot=1976): slope mean 2.062, 95% CI **[0.80, 6.38]**. The CI is wide because n=4 and the points are different architectures / dims / supervision regimes.

## 1.5e — Corrected branch table

EffNet-B3 is the primary control. ID bal acc gap is **0.683 → 0.702 = +1.9pp**, not vs ResNet-50's 0.655. Maha 0.726 is the published 16-d `z` number from `results/effb3_control/` (this pass's 1536-d `backbone_raw` Maha was 0.824; that is not the control row).

| Branch | Domain probe acc | OOD AUROC (Maha) | ID Bal Acc | supervision |
|---|---|---|---|---|
| Baseline (entangled, ResNet-50) | 0.979 | 0.868 | 0.655 | unsupervised |
| EffNet-B3 single-encoder (**primary control**) | 0.802 | 0.726 | 0.683 | unsupervised |
| z_lesion (invariant) | 0.724 | 0.409 | 0.702 | unsupervised |
| z_context (leaky) | 0.9997 | > 0.9999 | n/a | **domain-supervised** |

`PHASE1_REPORT.md` has been updated with the same row.

## Outputs

- `results/paperB/phase1_5/trivial_maha.csv`
- `results/paperB/phase1_5/trivial_logistic.csv`
- `results/paperB/phase1_5/preprocess_audit.json`
- `results/paperB/phase1_5/separation_margin.json`
- `results/paperB/figures/preview_leakage_vs_ood.pdf`

## What this does *not* do

It does not un-supervise `z_context`. The > 0.9999 remains a domain-supervised number. Phase 4 (semantic OOD, domain fixed) and Phase 11 (unseen third domain) are the remaining defences.
