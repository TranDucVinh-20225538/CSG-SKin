# WORK ORDER — Phase 13: technical depth on the derm result

Phase 12 buys **breadth** (does the phenomenon exist outside dermatology). Phase 13 buys **depth** (why does it happen, and where in the network). They are independent; Phase 13 needs no new training for items 13.1–13.4 and runs on cached features and logits from the Phase 3 sweep.

**Two items come first because they test existing load-bearing claims rather than extending them.** Verification before extension.

---

## 13.1 — Dimensional collapse: the mechanism nobody ruled out (highest priority)

Three mechanisms are refuted — image memorisation, class composition, domain specificity. **Dimensional collapse was never tested.**

Hypothesis: adversarial training collapses the effective dimensionality of the latent space, and Mahalanobis degrades because the space collapsed, not because domain information was removed. A supporting clue is already in hand: `z_context` has **84.5% of its variance in a single component**. If `z_lesion` collapses similarly as λ rises, one mechanism explains both branches.

### Measure, per λ ∈ {0, 0.25, 0.5, 1, 2, 4, 8}, all seeds, on ISIC-train features

For `z_lesion`, `z_context`, and `backbone_raw_lesion`:
- Full covariance eigenspectrum (plot on log scale, all λ overlaid)
- **Participation ratio** `PR = (Σλᵢ)² / Σλᵢ²`
- **Effective rank** = exp(spectral entropy)
- Components needed for 90% / 95% / 99% variance
- Condition number

### The discriminating question

Plot PR against λ alongside OOD AUROC against λ. Then ask whether **leakage and dimensionality can be separated**: is there any λ where leakage dropped but PR did not, or the reverse? If they move in lockstep, the two explanations are confounded in this data and the paper must say so. If they separate anywhere, that point identifies which one drives the OOD collapse.

### Robustness check — this one is mandatory

If dimensional collapse is real, **Mahalanobis on a collapsed space is numerically fragile**, and part of the 0.41 could be a conditioning artifact rather than a representational fact.

Recompute every Mahalanobis number in the Phase 3 sweep with **Ledoit-Wolf shrinkage** covariance estimation, and separately with explicit regularisation (`Σ + εI`, ε swept over several magnitudes). Report whether 0.41 survives.

- Survives → the result is robust and this check pre-empts an obvious reviewer attack.
- Moves materially → a genuine caveat that must go in the main text, not a footnote.

Report the condition number of the covariance actually used by the detector at each λ.

## 13.2 — Where in the network does it happen (cheap, and may yield a practical fix)

The Phase 1 work order specified `backbone_raw_lesion` as a feature space, but no report has shown it. Extract and score it now.

Run the full detector suite, plus the domain leakage probe, at three depths of the lesion branch, across all λ:

1. Backbone output, pre-projection (~1536-d)
2. `z_lesion` post-projection (16-d)
3. `z_lesion_norm` post-BatchNorm

| Outcome | Meaning | Consequence |
|---|---|---|
| **(a)** Backbone retains OOD detection while `z_lesion` loses it | Invariance is confined to the projection head | **Practical fix with teeth:** run OOD detection on pre-projection features, not on the invariant latent. No architectural change, no dual encoder needed. This is the answer to "so what do I do about it?" |
| **(b)** Backbone collapses too | Invariance propagates back through the encoder | No cheap fix exists; the finding is more severe and the paper says so |

Both outcomes are valuable; (b) is the stronger result. Also report **where the leakage drop happens** across the three depths — that localises where invariance is actually enforced.

## 13.3 — Confidence on unseen domains (cheap, and the most legible safety statement)

From cached logits at each λ, on ISIC test, `pad_heldout` and Fitzpatrick17k:
- Mean and distribution of max-softmax probability
- Predictive entropy
- The ID-minus-OOD confidence gap
- **ECE computed on OOD data**, not only ID

**The number to find:** the λ at which mean confidence on never-seen-domain data **exceeds** mean confidence on held-out in-distribution data. If that crossing exists, the paper gets this sentence:

> The more invariant the model is made, the more confident it becomes on data it has never seen.

That reads far better to a clinical audience than an AUROC table, and it follows directly from the compression hypothesis.

## 13.4 — Latent geometry (extends Item 2, does not duplicate it)

Item 2 of the final work order measures distance-to-centroid distributions and the OOD÷ID ratio. **Run Item 2 first**; 13.4 builds on its outputs.

Add, per λ, on `z_lesion`:
- Within-class covariance trace ÷ between-class covariance trace (neural-collapse style)
- Variance of OOD samples in latent space vs variance of ID samples
- Distance from each group to the **global** centroid, not only the nearest class centroid
- Per-class versions of all of the above for the 6 classes shared between ISIC and PAD

**Prediction from the compression hypothesis:** as λ rises, OOD samples show *lower* variance and *smaller* distance to centroids than ID samples of the same classes. Note that Phase 2.5's reweighted result (AUROC 0.240 when ISIC test is matched to PAD's class mix) already points this way: within matched classes, PAD sits closer to the centroids than ISIC does.

## 13.5 — Toy model: the part that could explain everything (deepest, most thought, least compute)

Fourteen phases of ablation have not produced a mechanism. A small analysable model might.

### The hypothesis to formalise

The adversary's objective is to make domain 1 indistinguishable from domain 0. With a **capacity-constrained encoder, matching the mode of a distribution is far cheaper than matching its tails.** So the encoder maps domain-1 inputs into the high-density core of domain 0's latent distribution rather than reproducing its full shape.

Consequence: OOD samples become *more typical than typical* — lower variance, closer to centroids than genuine ID samples. Mahalanobis then ranks them as more in-distribution than ID data, giving AUROC below 0.5.

This is consistent with every result so far, including the strengthened inversion (0.240) under class-matched reweighting, which rules out composition as the driver.

### Construction

- `x = a·u_label + b·u_domain + ε`, with `u_label ⊥ u_domain`, `ε ~ N(0, σ²I)`
- Label from `a`; domain from the distribution of `b` (`μ₀ ≠ μ₁`)
- **Mirror the derm asymmetry:** the classification loss is computed on domain 0 only, exactly as `L_cls` used `ignore_index=-1` for PAD. Domain 1 enters only through the adversarial term. This asymmetry is likely essential and must not be dropped
- Encoder: start linear (analytically tractable), then a small MLP
- Train with CE + λ·GRL, sweep λ
- Compute Mahalanobis AUROC analytically where possible, empirically otherwise

### What to show

1. λ=0 → domain 1 displaced along `u_domain` → AUROC > 0.5
2. λ rising → that component shrinks → AUROC → 0.5
3. **The part that matters:** identify what produces AUROC < 0.5. Test whether the linear encoder alone reaches sub-chance. **It may not** — if a linear model saturates at 0.5, then the sub-chance behaviour requires either capacity constraints, non-linearity, or the label-asymmetry above. **Finding out which ingredient is necessary is the result**, and that is worth more than a curve that merely looks like the derm one
4. Vary latent dimension to test whether sub-chance requires a compressed latent (derm used 16-d)

### Honesty requirement

If the toy model cannot reproduce sub-chance behaviour, **report that**. It would mean the effect requires something absent from the model, which narrows the search usefully. Do not tune the toy model until it produces the desired curve — a toy model fitted to its target explains nothing.

## 13.6 — Verify the kNN claim (cheap, corrects a load-bearing statement)

Camelyon17's coarse scan showed Mahalanobis moving while kNN stayed flat. The derm paper claims **all detectors collapse together**, which is the strongest single piece of evidence for a representational rather than a detector-specific effect.

But kNN was only added in the dual-branch evaluation, later than MSP/Energy/cosine/Mahalanobis. **Confirm kNN (k=50) in derm genuinely sits at ~0.41** across the Phase 3 sweep, and report k ∈ {10, 50, 200} for sensitivity.

If kNN does **not** collapse in derm, the "all detectors collapse together" claim must be rewritten precisely, and the Camelyon17 divergence stops being anomalous. Check this before drafting the Results section.

---

## Priority and cost

| Item | Compute | Why it ranks here |
|---|---|---|
| 13.1 | cached features | Untested mechanism + a mandatory robustness check on an existing headline number |
| 13.6 | cached features | Verifies a load-bearing claim before it is written up |
| 13.2 | cached + extraction | Cheap, may produce the practical recommendation the paper lacks |
| 13.3 | cached logits | Cheap, best sentence in the paper |
| 13.4 | cached features | Extends Item 2 toward the mechanism |
| 13.5 | minutes of GPU, days of thought | Deepest payoff; independent of whether Phase 12 succeeds |

13.1–13.4 and 13.6 need **no new training** and do not compete with Phase 12 for GPUs. Run them on CPU or a small allocation.

## Relationship to Phase 12

Independent. 13.5 in particular offers a route to a general claim that does **not** depend on WILDS replicating — if the mechanism is understood and shown to follow from the training objective rather than from dermatology, generality is argued rather than merely asserted.

## Standing rules

Unchanged. Report failures as findings; no retuning toward a prediction; outputs under `results/paperB/phase13/`; existing results untouched. If 13.1's shrinkage check moves the 0.41, that is a finding about your own prior result and gets reported as prominently as anything else here.
