# ADDENDUM — Phase 2.5: resolving the sub-chance puzzle

**Phase 2 result:** P2 FAILED. `z_lesion` Mahalanobis stayed sub-chance on PAD the adversary never saw (`pad_heldout` 0.427 ± 0.025 vs `pad_adv` 0.407 ± 0.021). Holding PAD out moved AUROC by ~0.02 — nothing.

**What this does and does not mean.** The *phenomenon* is confirmed and stronger than predicted: domain-invariance does not merely destroy covariate-shift OOD detection, it inverts it, robustly, on unseen data. The *proposed mechanism* is refuted. The Phase 2 design could not distinguish "the encoder memorised these specific PAD images" from "the encoder learned a general mapping for PAD-like inputs" — if the latter, held-out PAD behaves identically, which is what we observe.

**No mechanism claim may be written until Phase 2.5 resolves this.** Inference only, no training.

---

## 2.5a — Unseen-domain test (decisive)

Compute `z_lesion` Mahalanobis (and MSP, Energy, cosine, kNN — same detector suite as Phase 1) with **OOD = Fitzpatrick17k** (n=3887, never present in any training branch, any seed, any phase). ID = ISIC test, statistics fit on ISIC train only.

| Outcome | Conclusion |
|---|---|
| AUROC ≈ 0.50 | The sub-chance effect is a **learned domain-specific mapping** that generalises across PAD but not beyond it. Mechanism is adversarial, operating at domain level rather than image level |
| AUROC ≈ 0.41 | The effect is **structural**, not domain-specific. Points to the class-composition confound in 2.5b |

Run across all 5 `runB_orth1` seeds and report mean ± std.

## 2.5b — Class-composition control (the confound I should have caught earlier)

ISIC test spans 8 classes; PAD contains only 6 (no DF, no VASC) and is concentrated in BCC / AK / NEV. Class-conditional Mahalanobis takes distance to the nearest class centroid. ISIC test therefore carries DF, VASC and SCC — rare, poorly fit, large-distance cases — that PAD structurally cannot contribute.

**That asymmetry alone can drive AUROC below 0.5, with no adversarial explanation needed.**

1. Recompute ISIC-test-vs-PAD AUROC with **ISIC test restricted to the 6 classes present in PAD**. If AUROC moves toward 0.50, class composition is the driver.
2. Recompute with ISIC test **reweighted** to match PAD's class distribution (importance weighting on the AUROC estimate), as a second, less lossy version of the same control.
3. Report per-class Mahalanobis distance distributions (median, IQR, n) for ISIC test and for PAD, and identify which classes drive the inversion.
4. Repeat 1–3 for the baseline and the EffB3 control, so the confound is characterised for every model, not just CSG.

Apply the same restriction to the Phase 1 headline numbers and report whether any of them move.

## 2.5c — Fitzpatrick17k axis: distributions, not means

The current report gives only means on the 1-D domain axis (ISIC 0.07, Fitzpatrick17k 36, PAD 45). Means alone cannot support the claim.

Report for each of the three datasets on the shared k=1 axis: full distribution (density plot), mean, std, median, IQR, and **pairwise overlap coefficient** plus a standardised effect size (Cohen's d) for ISIC↔Fitz, Fitz↔PAD, ISIC↔PAD.

**The claim "Fitzpatrick17k lands between, skewed toward PAD" only holds if the spread is small relative to the 0.07→45 range.** If Fitz's std is large enough that it substantially overlaps PAD, say so and weaken the claim to "not on the ISIC side", which is all the data would then support.

Also report what fraction of Fitzpatrick17k images fall outside the ISIC–PAD interval entirely.

---

## Reframing to apply across all reports

**Phase 1.6a settled the detector question: a domain monitor is cheap.** A frozen ImageNet ResNet-50 that has never seen this data reaches 0.998 with a linear head. Report that number prominently — it shows the ISIC/PAD distinction is linearly decodable from generic visual features, which is itself a finding about the benchmark.

`z_context` must therefore no longer be described as a better detector. The surviving, defensible claim is:

> Factorization does not make domain *detectable* — it already is, from any backbone. Factorization **concentrates** domain into a single interpretable axis: k=1 gives AUROC 1.00 at 84.5% variance for `z_context`, versus 0.55–0.70 at k=1 for every other representation tested.

Every table comparing `z_context` to other representations must lead with the **k=1 column**, not the AUROC column, since the AUROC column now shows near-parity.

## Phase 3 is promoted

With the P2 mechanism refuted and the detector claim retired, the **core surviving claim is the phenomenon itself** — invariance inverts covariate-shift OOD detection. Present evidence for it is still correlational across heterogeneous models (the n=4 preview curve, slope CI [0.80, 6.38] — very wide).

The λ_adv sweep is the only experiment that converts this into a controlled intervention within a fixed architecture. **Submit Phase 3 as soon as Phase 4 releases GPUs.** Run it exactly as specified in the original work order, with two additions:

- At every λ_adv point, also score OOD against **Fitzpatrick17k**, not only PAD. If the inversion tracks λ_adv on a domain the adversary never saw, the result is far stronger.
- At every λ_adv point, report the class-restricted AUROC from 2.5b alongside the unrestricted one.

**Outputs:** `results/paperB/phase2_5/` + `PHASE2_5_REPORT.md`, stating plainly which explanation the data supports, including "neither cleanly" if that is the answer.

## Standing instruction

Phase 2's prediction failed and that is being reported as a finding, which is correct. Continue exactly that way. Do not adjust detectors, splits or hyperparameters to move any number toward a predicted value.
