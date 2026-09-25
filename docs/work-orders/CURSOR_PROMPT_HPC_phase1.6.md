# ADDENDUM — Phase 1.6: the supervised-control gap

Phase 1.5 verdict accepted: the ISIC↔PAD Mahalanobis benchmark is **not** trivially solvable (max hand-crafted Maha AUROC 0.6474), and `z_context` is a 1-D domain axis (PCA k=1 → AUROC 1.0, 84.5% variance).

But Phase 1.5 also showed **supervised logistic on the same hand-crafted features reaches 0.81–0.88 AUROC**. `z_context` is itself domain-supervised. So the honest comparator for `z_context` is another *supervised* detector — which has not been measured. Phase 1.6 measures it.

**No training, no GPU beyond inference. Features from Phase 1 may already be cached.**

---

## 1.6a — Supervised domain-head controls (decisive)

**Question:** does the factorized architecture provide the monitor, or would a linear domain head on any representation do the same?

Fit a linear domain classifier (logistic regression, ISIC vs PAD, 5 seeds, same 70/30 stratified protocol as the leakage probe) on each frozen representation below, then use its decision score as an OOD score on ISIC test vs PAD:

| Representation | Source |
|---|---|
| `imagenet_resnet50_raw` | frozen ImageNet ResNet-50, never trained on this data |
| `imagenet_effb3_raw` | frozen ImageNet EffNet-B3, never trained on this data |
| `baseline_backbone_raw` | the trained ResNet-50 baseline's 2048-d features |
| `effb3_control_backbone_raw` | the EffB3 single-encoder control's features |
| `z_context` | CSG context branch (reference point, already at >0.9999) |

Report AUROC, AUPR-in/out, FPR@95, plus **PCA k=1 AUROC** for each — the k=1 column tells you whether every one of these collapses to a 1-D domain axis, or whether that is special to `z_context`.

**Interpretation, to be written verbatim into the report:**
- If `baseline_backbone_raw` + linear head also reaches ≈ 0.99: **a domain monitor is cheap and obtainable from any representation.** `z_context` is then not valuable as a detector per se. The contribution must be restated as: *factorization yields an invariant diagnostic branch and a monitor simultaneously in one model, with no information flow between them* — which the entangled baseline structurally cannot do, since its single representation cannot be both invariant and domain-informative. State this plainly; do not bury it.
- If the frozen/bolted-on controls plateau well below `z_context`: the context branch learns something the others do not, and it stands as a detector contribution in its own right.

Both outcomes keep the paper. Only silence about this control would sink it.

## 1.6b — Reporting fix for the perfect score

Phase 1.5b found 1 ID outlier above 2 OOD images. That is 2 discordant pairs out of ≈ 5067 × 2298 ≈ 11.6M, i.e. AUROC ≈ 0.99999983.

**Never print `1.0000 ± 0.0000` in the manuscript.** Report as `AUROC > 0.9999 (2 discordant pairs / 11.6M)` and include the discordant-pair count in the table footnote. A bare 1.0000 reads as a data-leakage bug; the pair count reads as diligence. Update every table and figure produced so far.

Also inspect the 3 boundary images (1 ID, 2 OOD) directly and describe them in one line each — they are the most informative images in the dataset and a reviewer will ask.

## 1.6c — Unblock the raw-source delta

PAD PNGs are mode 600. Check ownership: if the files are yours, `chmod u+r` (or `chmod 644`) and rerun the Phase 1.5a trivial detectors on **raw source images** (common resize only, no soft-crop). Report the AUROC delta between raw and soft-cropped.

This answers whether the preprocessing pipeline injects or removes domain signal — a question a reviewer will raise about any preprocessing-heavy pipeline. Do not skip it; it is a ten-minute fix. If the files are owned by another user, say so and stop.

## 1.6d — Phase 11 upgrade: the 1-D axis projection test

The k=1 finding makes Phase 11 much stronger than a binary OOD check.

Fitzpatrick17k is web-scraped from dermatology atlases — **clinical photographs**, so modality-wise it sits closer to PAD (smartphone clinical) than to ISIC (dermoscopy). That gives a directional prediction, not just a detection one:

1. Extract `z_context` for Fitzpatrick17k images (no class labels needed — use all available rows).
2. Project onto the **k=1 domain axis** fit on ISIC vs PAD.
3. Report where the distribution lands relative to the ISIC and PAD modes.

**Prediction P5′:** Fitzpatrick17k lands on the PAD side of the axis, or between the two, **not** on the ISIC side.

If it does, the axis encodes acquisition modality in a way that generalises to a dataset it has never seen — which is the answer to "you supervised it to separate two datasets." If it lands on the ISIC side or bimodally, report that: it would mean the axis is partly dataset-specific, and the claim must be narrowed accordingly.

Produce a 1-D density plot: ISIC / PAD / Fitzpatrick17k on the shared domain axis. This is a strong figure regardless of outcome.

**Licence handling:** Fitzpatrick17k images are web-scraped with mixed copyright. Use for **evaluation only**, do not redistribute, and cite both the Fitzpatrick17k paper and the source atlases. Record this alongside the ISIC CC BY-NC 4.0 and PAD CC BY 4.0 entries.

---

## Do not touch the running jobs

Phase 2 (`60446_[0-4]`) and Phase 4 (`60451_[0-17]`) stay as they are. Phase 3 stays unsubmitted until both report.

Phase 1.6 is inference-only and should be run on a small allocation or CPU nodes so it does not compete with the training arrays for GPUs.

**Outputs:** `results/paperB/phase1_6/` + `PHASE1_6_REPORT.md`, with an explicit verdict on whether a bolted-on linear domain head matches `z_context`, and the corrected reporting convention applied retroactively to all existing tables.
