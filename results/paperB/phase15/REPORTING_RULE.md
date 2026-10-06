# Phase 15 — reporting rule

Written before any Phase 15 experiment ran. The tier of a result is fixed by this rule, not by the numbers.

## Tiers

**Main text.** Every experiment that tests the central claim, whatever it shows.

- 15.1 single-encoder DANN on ISIC↔PAD (architecture + port validation)
- 15.2 CORAL / MMD / IRM / GroupDRO on the same backbone (adversarial training vs invariance)

A failure in either is a boundary condition on the claim, not an ugly result to drop. Six refuted predictions are why this work is credible. Selective reporting now would retroactively devalue all six.

**Appendix.** Exploratory or underpowered runs: fewer than 3 seeds, or a dataset where the universal gate in §1 never opens. Status stated in the table.

**Inconclusive, with diagnostic.** A run abandoned for an implementation reason. Never reported as a scientific negative. This is what Camelyon17 should have been called the first time.

## Universal gate (every dataset, before any OOD number)

1. Domain leakage is **balanced accuracy** against floor **1/K** (K = training domains). Not the majority-class rate.
2. Leakage must actually fall toward that floor at some λ. If it does not, the setting is **inconclusive**, not negative.
3. Log the adversary's own training accuracy and loss **per epoch**, and the gradient norm entering the encoder. A head pinned at chance from epoch 1 (Camelyon17) or pinned at 1.0 with vanishing gradient are both **null experiments**.
4. Every OOD statistic is fit on the **training split only**. Asserted in code.
5. The OOD domain is absent from the invariance objective, or a disjoint held-out portion exists (`pad_heldout`).

## This phase

| Item | Tier decided now | Gate |
|---|---|---|
| 15.1 ISIC↔PAD single-encoder DANN | **Main text** (3+ seeds; invariance known achievable on this pair) | Must open, or the port is broken (inconclusive for Camelyon, not a derm negative) |
| 15.2 CORAL / MMD / IRM / GroupDRO | **Main text** if the 15.1 port is sound and ≥3 seeds; else appendix / inconclusive | Same leakage gate per objective |
| 15.3 PACS (4 leave-one-out) | Main text if gate opens and ≥3 seeds; else appendix | 1/3 floor on the 3 train domains |
| 15.4 Camelyon17 retry | Only if 15.1 shows the port is sound. If the head again fails on a validated port: dataset-specific optimisation failure with evidence, **not** “Camelyon resists invariance.” If 15.1 shows the port is broken: do not rerun; Camelyon stays inconclusive. |
| 15.5 iWildCam | Gated on 15.1, not on 15.4. Appendix if gate never opens. | 1/K, K = number of training domain labels used |
| 15.6 chest X-ray | Availability first. Same rule once data exist. | 1/2 if two sites |

Outputs only under `results/paperB/phase15/`. Nothing existing is modified. No retuning toward a prediction. Failures are findings.
