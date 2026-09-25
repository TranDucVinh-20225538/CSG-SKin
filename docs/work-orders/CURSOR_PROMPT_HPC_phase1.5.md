# ADDENDUM — Phase 1.5 + green-light for Phase 2 & 4

Phase 1 PASSED: `z_context` Mahalanobis AUROC = 1.0000 ± 0.0000, `z_lesion` = 0.4089 ± 0.0196, ResNet-50 `backbone_raw` = 0.8679 ± 0.0186.

A perfect 1.0000 is the single most attackable number in the paper. Phase 1.5 closes that attack surface. **It requires no GPU and no retraining.** Do it before submitting Phase 2.

---

## Phase 1.5a — Trivial-detector control (the important one)

**Question:** how much of the ISIC-vs-PAD separation is obtainable without any learned representation at all?

Build detectors on hand-crafted features of the **preprocessed 224×224 images the models actually see** (not the raw source files):

| Feature set | Dim | Description |
|---|---|---|
| `rgb_moments` | 6 | per-channel mean + std |
| `color_hist` | 48 | 16-bin histogram per RGB channel, normalised |
| `gray_downsample` | 256 | 16×16 grayscale, flattened |
| `hsv_moments` | 6 | per-channel mean + std in HSV |
| `laplacian_stats` | 4 | mean/std/skew/kurtosis of the Laplacian (sharpness / focus proxy) |

For each feature set, run the **identical protocol as Phase 1**: class-conditional Mahalanobis fit on **ISIC train only**, scored on ISIC test (ID) vs PAD (OOD). Report AUROC, AUPR-in, AUPR-out, FPR@95.

Also fit a plain logistic regression (ISIC vs PAD, 70/30 stratified, 5 seeds) on each feature set and report balanced accuracy + AUROC, as a domain-separability ceiling.

**Interpretation — write this into the report explicitly:**
- If any trivial feature set reaches AUROC ≥ 0.95: the ISIC-vs-PAD OOD benchmark is **trivially solvable by low-level image statistics**. This is a major finding in its own right and must become a headline result — it means the cross-dataset OOD benchmarks the field reports are measuring colour and exposure statistics. It also means `z_context` must be re-positioned: its value is no longer "it detects domain" but "it generalises to unseen domains" (Phase 11), and the paper must say so plainly.
- If all trivial feature sets stay below ~0.85: `z_context` is learning something non-trivial, and it stands as an architectural contribution.

Either outcome is publishable. **Do not editorialise toward either one.**

Repeat the same on the **raw source images before preprocessing** (resize to a common size only). Comparing the two tells you whether the soft-crop pipeline injects or removes domain signal — report the delta.

## Phase 1.5b — Separation-margin diagnostics

The claim "AUROC = 1.0" is stronger or weaker depending on the geometry. Report for `z_context`:

1. Full score distributions for ID and OOD: min, max, mean, std, and the **gap** between max-ID and min-OOD score (or the overlap, if any).
2. **AUROC after PCA to k dimensions**, k ∈ {1, 2, 4, 8, 16, 32, 64}. If k=1 already gives 1.0, the monitor is effectively a one-dimensional "domain-ness" scalar — that is both a caveat and an interpretability win, so report it either way.
3. Same for `z_lesion`, as a contrast.

## Phase 1.5c — Preprocessing-artifact audit

Confirm ISIC and PAD go through a byte-identical preprocessing path: same interpolation, same resize order, same JPEG/PNG encode settings, same colour space handling. Record the **original resolution and aspect-ratio distributions** of both source datasets.

If the two domains differ systematically in source resolution or encoding, a detector could be reading the preprocessing pipeline rather than the imaging modality. Report any asymmetry found; do not attempt to fix it yet.

## Phase 1.5d — Four-point preview curve (free)

You already have every number needed. Plot leakage (domain probe acc) on x against Mahalanobis OOD AUROC on y:

| Point | Leakage | OOD AUROC | Bal Acc |
|---|---|---|---|
| `z_context` | 0.9997 | 1.000 | — |
| ResNet-50 baseline | 0.979 | 0.868 | 0.655 |
| EffB3 single control | 0.802 | 0.726 | 0.683 |
| `z_lesion` | 0.724 | 0.409 | 0.702 |

Fit and report the trend with a CI. **Label it explicitly as correlational across heterogeneous models — it is not the controlled result.** Phase 3's λ_adv sweep is what turns this into an intervention. Save as `results/paperB/figures/preview_leakage_vs_ood.pdf`.

## Phase 1.5e — Fix the mandatory table

The Phase 1 report's branch table omits the EffNet-B3 single-encoder control row. Per the work order, every comparison table takes EffB3 as the primary control, not ResNet-50. Add the row (numbers in the table above, from `results/effb3_control/`) and restate the ID bal acc gap as **0.683 → 0.702 = +1.9pp**, not as a gap against ResNet-50's 0.655.

**Outputs:** `results/paperB/phase1_5/` + `PHASE1_5_REPORT.md`, with a clear verdict on whether the ISIC↔PAD OOD benchmark is trivially solvable.

---

## Green light: Phase 2 and Phase 4 in parallel

Phase 4 uses ISIC only and does not depend on the Phase 2 PAD split, so the two are independent. **If the queue allows, submit both as SLURM array jobs concurrently** rather than sequentially.

Phase 4 (double dissociation) is the paper's headline figure *and* the defence for the 1.0000 result, so it must not be allowed to queue behind Phase 2.

Run both exactly as specified in the original work order. Phase 3's λ_adv sweep stays gated until Phase 2 and Phase 4 report.

## Phase 11 promoted: optional → required

At AUROC 1.0 with domain supervision, "you trained it to do that" becomes the central objection, and only an unseen third domain answers it. Start the **non-GPU** part now, in parallel with Phase 2/4:

1. Choose a third dermatology dataset never used in training — DDI, Derm7pt, or Fitzpatrick17k.
2. Check and **record the licence and citation terms** (the original manuscript's Limitation #5 flags that this was never done for ISIC or PAD either — fix all three while you are here).
3. Acquire the data and build a metadata CSV matching the existing schema.
4. Report availability and any label-mapping issues into the 8-class space **before** running anything.

Evaluation itself is inference-only and cheap once the data is in place.

---

## Reporting

Append to `results/paperB/MASTER_REPORT.md`:
- Phase 1.5 verdict: is the ISIC↔PAD OOD benchmark trivially solvable? With the numbers.
- The corrected branch table including EffB3.
- The preview curve, labelled correlational.
- Phase 11 dataset choice with licence status.

As before: report what the numbers say, including results that weaken the thesis. A trivial detector scoring 1.0 is not a setback — it is the most interesting thing this project could find.
