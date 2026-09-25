# AMENDMENT — Phase 12.1b corrections (apply before interpreting array 60636)

Audit accepted: 7 of 8 items mismatched. The coarse scan was not testing the derm adversarial recipe, so its four points are void as evidence about the cliff. Matched-recipe rerun is correct.

Four corrections.

## 1 — Add a λ=0 anchor under the matched recipe (blocking for interpretation)

The λ=0 row in the coarse table was trained with the **old** recipe: SGD single group, no BatchNorm on features, unbalanced hospital batches, no classifier LR scaling, GRL on raw features.

It is therefore **not comparable** to the λ ∈ {10, 30, 100, 300} runs. Without a matched-recipe λ=0, no statement of the form "AUROC fell from X to Y" is valid, because X was measured under a different training configuration.

**Add λ=0 to the array** (same seed, same everything else). With λ_adv=0 the GRL contributes no gradient to the encoder while all other recipe changes still apply — a well-defined and necessary control. It is the reference row for every comparison in this phase.

## 2 — The λ range may now overshoot; be ready to extend downward

`{10, 30, 100, 300}` was chosen when the adversary was weak. The matched recipe is far stronger: 30× adversary LR, BN-normalised latent, balanced batches, GRL ramp, proper head.

**Derm's operating point with this exact recipe was λ_adv = 2.** λ=10 is already 5× that. All four points may now sit past the transition, which would teach nothing about where it is.

Decision rule once 60636 reports:

| Observation at λ=10 | Action |
|---|---|
| Leakage already at or near the balanced-accuracy floor | **Extend downward**: λ ∈ {0.25, 0.5, 1, 2, 5}. Do not conclude from four post-cliff points |
| Leakage still high (> ~0.8) | The upward scan was right; continue as planned |
| Leakage mid-range | Dense-sample around λ=10 |

Also watch for training instability at λ=100 and λ=300 — a 30× adversary LR at those weights may diverge. Divergence is not outcome (b); record it separately as a training failure and do not read collapsed ID accuracy from a diverged run as evidence about entanglement.

## 3 — Majority baseline is mis-specified

`0.437` = 132k / 302k, the **plain-accuracy** majority floor.

Leakage is being reported as **balanced accuracy**, whose floor for a 3-class problem is **1/3 ≈ 0.333**, independent of class imbalance (a constant predictor scores recall 1.0 on one class and 0 on the other two).

Comparing balanced accuracy against 0.437 mixes two scales and would cause "invariance achieved" to be declared early. **Use 0.333 as the floor for balanced accuracy**, or report plain accuracy against 0.437 — one or the other, stated explicitly. Fix this in the aggregate script before 60637 runs, and in the derm tables if the same mixing occurred there (derm majority 0.688 is a plain-accuracy floor; its 2-class balanced-accuracy floor is 0.5 — check which was used where).

## 4 — The audit is a result; record it for the manuscript

Seven of eight mismatches, with leakage frozen at 0.96 across four orders of magnitude of λ, establishes something the paper should state:

> A weak adversary produces **neither** debiasing **nor** the safety failure. Leakage stays high and OOD detection stays intact. The danger zone is precisely where the method succeeds.

This is sharper and more useful than "invariance training is unsafe". It tells a practitioner when to worry: the OOD gate degrades as a function of *achieved invariance*, not of the nominal λ or of merely having an adversarial head in the loss. Many deployed DANN setups are too weak to induce invariance and are correspondingly harmless.

Add to the Discussion of the derm paper regardless of how Phase 12 resolves, and record the audit table itself as supporting evidence — it is a concrete demonstration that λ alone does not characterise adversarial pressure.

---

## Unchanged

Hard stop at λ=300 with leakage > ~0.8 → outcome (c), stop and report. iWildCam gated. Dense cliff grid gated until leakage moves. Report failures as findings; no retuning.
