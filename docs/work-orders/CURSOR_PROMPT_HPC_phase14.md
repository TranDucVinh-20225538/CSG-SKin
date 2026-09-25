# WORK ORDER — Phase 14: tails, saturation, and a corrected causal claim

Four items. A and D need essentially no compute. B and C are bounded and must not become open-ended.

**Headline promotion, applies across everything below:** the strongest number in the project is in 13.3 — **OOD ECE rises 0.247 → 0.746 while ID ECE stays flat at ~0.10**, with mean confidence on never-seen-domain data overtaking in-distribution confidence by λ=2. ECE is a metric clinicians and regulators already understand, and it needs no explanation of Mahalanobis. Treat it as the paper's lead result from here on.

---

## Item A — Is the inversion a tail effect? (cached data, minutes, highest value)

13.4 refuted the compression hypothesis. At λ=0 the OOD group is **already** far more concentrated than ID (var 13.2 vs 60.0; distance-to-global-centroid 3.25 vs 7.15), and adversarial training **decompresses** it toward ID (13.2 → 42.7 → 39.5). The adversary is matching distributions, not compressing.

Yet at λ=2, ID remains slightly wider (var 45.7 vs 39.5; dist-global 6.26 vs 5.69). Revised hypothesis:

> The adversary matches the **bulk** of the two distributions. ID retains a heavier upper tail — rare classes, ambiguous cases, artifacts, genuine within-domain outliers — while OOD's tail is clipped by the matching pressure. Mahalanobis is tail-sensitive, so ID produces more extreme scores than OOD and AUROC falls below 0.5.

### A1 — Score distribution comparison

Per λ, all seeds, for `z_lesion_norm`, on ISIC test vs `pad_heldout` vs Fitzpatrick17k:
- Quantiles at 1, 5, 25, 50, 75, 90, 95, 99, 99.9
- Mean, std, skewness, kurtosis
- Overlaid density plots on a log-score axis

**Prediction:** medians converge as λ rises while ID's p95/p99 stay above OOD's.

### A2 — The decomposition test (this is the decisive one)

Recompute AUROC **after excluding the upper tail**: drop samples above the 90th (and separately 95th, 99th) percentile of the *pooled* score distribution, then recompute on what remains.

- Inversion disappears when tails are excluded → **the tail is the driver**, hypothesis confirmed.
- Inversion persists in the bulk → the effect is distributional throughout and the tail hypothesis is wrong. Report it and close this line.

### A3 — What is actually in the ID tail?

Identify the ID samples in the top 1% of scores and break them down by class, by predicted-vs-true correctness, and by softmax confidence.

Note the constraint this must satisfy: **Phase 2.5b showed class restriction does not fix the inversion** (0.414 vs 0.409) and reweighting *strengthens* it (0.240). So the tail cannot be purely rare-class-driven. If it is not rare classes, determine what it is — within-class outliers, mislabelled cases, artifacts. That identity is the mechanism's concrete content.

Also run A1–A2 at the **backbone** depth, which connects to Item D.

→ `results/paperB/phase14/tails/`

## Item D — Correct the causal claim (no compute, do before any drafting)

**The paper must not claim that reducing leakage causes the OOD collapse.** Two independent results now contradict it:

1. **Phase 3:** from λ=0.25 to λ=2, leakage *rises* (0.553 → 0.584) while AUROC *falls* (0.508 → 0.413). Opposite directions.
2. **Phase 13.2:** backbone leakage is flat across the whole sweep (0.982 → 0.941) while backbone Mahalanobis falls 0.749 → 0.475. Domain information is still linearly decodable, and OOD detection collapses anyway.

**Corrected framing:**

> Adversarial training causes both the leakage drop and the OOD collapse, but the OOD collapse is **not mediated by linear domain-decodability**. It follows from geometric changes to the representation that a linear domain probe does not capture.

### Actions

- Audit every report, figure caption and table under `results/paperB/` for causal language linking leakage to OOD collapse. List each instance and correct it.
- **Plot the dial against λ, never against leakage.** An AUROC-vs-leakage plot asserts exactly the mediation just disproved. The n=4 preview curve is now doubly superseded — appendix only, or cut.
- Keep leakage in Table 1 as a measured quantity, not as the explanatory variable.
- Add the 13.2 backbone decoupling as its own result. It is a finding, not a detail.

This makes the paper more precise, not weaker. "Adversarial training breaks OOD detection through a channel invisible to the standard leakage probe" is a sharper claim than the mediated version, and it is what the data supports.

## Item B — Camelyon17: is the adversary saturating? (one diagnostic, then stop)

Outcome (c) may be premature. With `adv_lr_multiplier=30` and λ up to 300, the domain head very likely reaches ~100% accuracy almost immediately. At that point the cross-entropy saturates and the gradient flowing back through the GRL approaches zero — the standard discriminator-too-strong failure mode. **An adversary that is too strong is as ineffective as one that is too weak**, and that would explain leakage sitting at 0.94 while ID accuracy is untouched: the encoder never felt real pressure.

### B1 — Diagnose

For λ ∈ {10, 100, 300}, from existing logs if the metrics were recorded, otherwise one short rerun:
- Adversary training accuracy and loss, per epoch
- **Gradient norm entering the encoder through the GRL**, per epoch
- Encoder gradient norm from `L_cls`, for scale comparison

| Finding | Conclusion |
|---|---|
| Adversary pinned near 1.0 **and** GRL gradient norm → 0 | Gradient vanishing. **(c) is wrong.** Proceed to B2 |
| Adversary not saturated, GRL gradient healthy, leakage still flat | **(c) confirmed.** Close Phase 12, report as a boundary condition |

### B2 — Only if B1 shows vanishing gradients

Replace the GRL-CE objective with a non-saturating alternative — domain-confusion (maximise the entropy of the domain head's output) or GRL-CE with domain-label smoothing. Test at two or three λ values only.

**Hard stop after B2 regardless of result.** No further escalation, no new objectives beyond these. If leakage still does not move, Phase 12 closes as (c) with the diagnostic attached, which is a better-evidenced negative than the current one.

iWildCam stays gated throughout.

## Item C — Toy model, one pre-registered revision

The previous grid failed correctly and was not tuned. It established two things worth keeping: the linear/BN-off configurations sit at chance (0.510 → 0.534, 0.524 → 0.538), matching the closed-form derivation, and the sub-chance in the full cell was a **BatchNorm-on-mixed-batches artifact present already at λ=0**.

The missing ingredient, inferred from Item A's hypothesis rather than from wanting a result: **the toy's ID distribution has no tail to retain.** Eight equally-modelled Gaussian classes cannot produce the ID/OOD tail asymmetry that real ISIC has.

### The single revision

Add heterogeneous, heavy-tailed structure to the **ID domain only**:
- Class-dependent variance — per-class `τ_k` drawn log-uniformly over roughly an order of magnitude
- A few very rare classes (n ≈ 50–100 against thousands for common ones)
- A small contamination component: a few percent of ID samples from a much wider Gaussian

Everything else stays at the locked configuration. **BatchNorm OFF for the primary cell** — it was the confound. Report BN-on separately.

### Pre-registration (mandatory)

Write expected outcomes and pass criteria to `phase13_5/PREREGISTER_v2.json` **before running**. Pass requires all of:

1. **AUROC at λ=0 must be at or above 0.5.** The previous run started at 0.327 and was therefore never in the derm regime. If the revision again starts below 0.5, it is the same class of artifact — **reject it and stop**
2. Sub-chance appears at λ ≥ 0.25 (F2 with the correct starting point, i.e. F1's shape)
3. Detectors collapse together (F3)
4. Reweighting to the OOD class mix strengthens the inversion (F5)
5. A never-adversarially-seen third domain also inverts (F7)

Criteria 1 and 2 together are the minimum. Criteria 3–5 determine whether the mechanism is confirmed or merely not excluded.

### If it fails

**Close the mechanism question.** The paper then reports a robust phenomenon with **five** refuted explanations — image memorisation, class composition, domain specificity, dimensional collapse, and latent compression — plus an explicit statement that a controlled synthetic reproduction failed under a pre-registered grid.

That is an honest and publishable position, and the five refutations are themselves a contribution to whoever solves it next. Do not run a third revision.

---

## Priority

| Item | Compute | Note |
|---|---|---|
| A | cached, minutes | Decisive test of the corrected mechanism |
| D | none | Blocks drafting; do it before writing anything |
| B | one short diagnostic | Determines whether Phase 12 closes correctly |
| C | minutes | One revision only, pre-registered, then closed |

## Standing rules

Unchanged. Report failures as findings. No retuning toward a prediction — Item C's pre-registration exists to enforce this. Outputs under `results/paperB/phase14/` and `phase13_5/`. Existing results untouched.
