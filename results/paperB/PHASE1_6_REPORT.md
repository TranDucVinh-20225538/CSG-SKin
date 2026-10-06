# Paper B — Phase 1.6 report

Inference only. Job `60459` COMPLETED (CPU, `CUDA_VISIBLE_DEVICES=""`). Phase 2 and Phase 4 were not touched by this job.

## Reporting convention (applied retroactively)

Never print **1.0000 ± 0.0000**.

`z_context` Mahalanobis on `runB_orth1` seed 42 is AUROC = 0.99999983 = **2 discordant pairs / 11,643,966 ≈ 11.6M**. n=5 mean 0.9999986 ± 9.0e-7. Manuscript form: **AUROC > 0.9999 (2 discordant pairs / 11.6M)**. A bare 1.0000 reads as a data-leakage bug; the pair count reads as diligence.

Updated: `PHASE1_REPORT.md`, `PHASE1_5_REPORT.md`, `MASTER_REPORT.md`, `figures/preview_leakage_vs_ood.pdf`.

## 1.6b — The three boundary images

Higher Maha = more OOD. The overlap is 1 ISIC test image above 2 PAD images.

| Role | ID | Label | Maha | One-line description |
|---|---|---|---:|---|
| ID outlier (most PAD-like ISIC) | `ISIC_0070534` | BCC | 1015.93 | Circular dermoscopic field of a pink BCC with a central ulcerated/crusted nodule and a featureless erythematous periphery — still clearly dermoscopy (vignette, gel), not a clinical photograph. |
| OOD low-1 (most ISIC-like PAD) | `PAT_2013_4148_303` | NV | 876.91 | Distant clinical snapshot of a small brown macule on pale skin with a large empty surround — smartphone/clinical, not dermoscopy. |
| OOD low-2 | `PAT_1215_752_707` | NV | 881.75 | Close clinical view of a hairy pigmented patch (terminal hairs across a brown field) — body photography, not a dermoscopic FOV. |

The ranking errors are **not** visual modality swaps. The ID outlier is still dermoscopy; both PAD lows are still clinical. Copies: `results/paperB/phase1_6/boundary_images/`.

## 1.6c — Raw-source delta

**BLOCKED.** PAD original PNGs are owned by `anhnv` (mode 600), not this user (`toandq`). They are not ours to `chmod`. Stopped. AUROC delta raw vs soft-crop is unmeasured.

## 1.6a — Supervised domain-head controls

Question: does the factorized architecture provide the monitor, or would a linear domain head on any representation do the same?

Protocol: logistic regression, ISIC vs PAD, 70/30 stratified, 5 seeds (same leakage-probe split). Decision score = P(PAD), reported as OOD AUROC / AUPR / FPR@95 on the 30% hold-out. PCA k=1 fit on ISIC train only.

Representations: `imagenet_resnet50_raw`, `imagenet_effb3_raw`, `baseline_backbone_raw` (trained R50), `effb3_control_backbone_raw`, `z_context` (reference).

Job `60459` COMPLETED. Lead with the k=1 column: AUROC of a linear head is near-parity across backbones.

| Representation | k=1 AUROC (var) | Linear domain-head AUROC | supervision of the representation |
|---|---|---|---|
| ImageNet ResNet-50 (frozen, never saw this data) | 0.647 (14.6%) | **0.998 ± 0.0003** | ImageNet |
| ImageNet EffNet-B3 (frozen) | 0.545 (—) | 0.994 ± 0.001 | ImageNet |
| Trained baseline `backbone_raw` | 0.648 | 0.997 ± 0.001 | class-supervised |
| Trained EffB3 `backbone_raw` | 0.698 | 0.991 ± 0.002 | class-supervised |
| `z_context` | **1.00 (84.5%)** | > 0.9999 | **domain-supervised** |

A frozen ImageNet ResNet-50 that has never seen this data reaches **0.998** with a linear head. A domain monitor is cheap. The ISIC/PAD distinction is linearly decodable from generic visual features.

`z_context` is **not** a better detector. Factorization does not make domain detectable — it already is, from any backbone. Factorization concentrates domain into a single interpretable axis: k=1 gives AUROC 1.00 at 84.5% variance for `z_context`, versus 0.55–0.70 at k=1 for every other representation tested.

The first locked interpretation fired: a domain monitor is obtainable from any representation. The surviving architectural claim is simultaneous factorization (an invariant diagnostic branch *and* a 1-d domain axis in one model, with no information flow between them), which an entangled baseline structurally cannot do.

## 1.6d — 1-D axis projection (Fitzpatrick17k)

n = 3887 images (partial download; eval only). Means on the shared k=1 axis: ISIC 0.07, Fitz 35.56, PAD 44.80. Phase 2.5c replaces means-only with full distributions: Fitz–PAD overlap coefficient **0.47**, Cohen's d −1.18. Supported claim is **not on the ISIC side**, not a tight between-mode. See `PHASE2_5_REPORT.md`.

## Outputs

- `results/paperB/phase1_6/boundary_images.json` + previews
- `results/paperB/phase1_6/raw_source_blocked.json`
- `results/paperB/phase1_6/supervised_head.json`
- `results/paperB/phase1_6/fitz_axis.json`
- `results/paperB/figures/preview_leakage_vs_ood.pdf` (relabelled > 0.9999)
- `results/paperB/figures/domain_axis_distributions.pdf` (Phase 2.5c)
