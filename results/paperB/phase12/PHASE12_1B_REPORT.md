# Phase 12.1b — find where leakage actually drops

Audit: `PHASE12_1B_AUDIT.md`. Coarse dense grid was **not** submitted. iWildCam not started.
Coarse λ=0 is **not** the reference row. Every comparison uses matched-recipe λ=0.

**Inconclusive — not a Camelyon result.**

The online adversary never trained: `train_adv_acc` ≈ 0.33 and domain CE = ln(3) at every epoch of every λ, including epoch 1. A post-hoc linear probe still reads ~0.94. GRL does not flip the head (Phase 14 sign check: head cosine +1.000, encoder −1.000). This is not evidence that Camelyon17 resists invariance. It is an unresolved implementation failure. No Camelyon17 number is reported. Do not raise λ. iWildCam stays gated. Phase 12 belongs in Limitations.

## Floors (do not mix scales)

Leakage is **3-class balanced accuracy**. Its chance floor is **1/3 ≈ 0.333**, independent of hospital imbalance. A constant predictor scores recall 1.0 on one class and 0 on the other two.

The figure 0.437 = 132k/302k is the **plain-accuracy** majority (always predict hospital 4). It is reported in the table as `plain maj` and is **not** the invariance target. Declaring invariance against 0.437 would declare it early.

## λ=10 range call

**upward_was_right** — Leakage at matched λ=10 is still high (0.942 > 0.8). The upward scan was right. Continue classification on this grid. Do not extend downward.

## Per-λ (leakage first)

| λ_adv | leak 3-cls bal acc | bal-acc floor | plain maj | ID acc | diverged | Maha h2 | kNN h2 | Maha h1 | kNN h1 | xfer acc h2 | xfer acc h1 |
|---:|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.975 | 0.333 | 0.437 | 0.993 | no | 0.685 | 0.882 | 0.608 | 0.871 | 0.865 | 0.912 |
| 10 | 0.942 | 0.333 | 0.437 | 0.992 | no | 0.580 | 0.911 | 0.536 | 0.830 | 0.874 | 0.864 |
| 30 | 0.947 | 0.333 | 0.437 | 0.988 | no | 0.487 | 0.768 | 0.445 | 0.732 | 0.884 | 0.758 |
| 100 | 0.943 | 0.333 | 0.437 | 0.989 | no | 0.624 | 0.857 | 0.614 | 0.744 | 0.801 | 0.789 |
| 300 | 0.894 | 0.333 | 0.437 | 0.979 | no | 0.467 | 0.863 | 0.483 | 0.895 | 0.833 | 0.702 |

OOD AUROC is secondary until leakage moves. Do not call a Maha-only drop the derm phenomenon. Do not write 'AUROC fell from X to Y' unless X is matched-recipe λ=0.

Figure: `phase12/figures/fig_12_1b_leakage_id.{png,pdf}`.

**STOP.** Dense cliff grid and iWildCam remain gated until leakage moves and the range call is acted on.

## Phase 14.B1 — saturation diagnostic (attached to (c))

Hypothesis: `adv_lr_multiplier=30` and λ up to 300 pinned the domain head at ~100% and starved the GRL gradient.

**Rejected.** `train_adv_acc` is 0.33–0.36 at every epoch of every matched λ, including 10, 100 and 300. Checkpoint domain CE is exactly ln(3). The online head is a constant uniform predictor, not a saturated discriminator. ||g_enc through GRL|| is 0 because ∂L_adv/∂z = 0, not because acc→1. A post-hoc linear probe still reads 0.94, so the representation was never pressured.

B2 is **not** run. The GRL graph is standard; the head still never left ln(3). Phase 12 does not close as “Camelyon resists invariance.” It is inconclusive and is not reported as a result. iWildCam stays gated.

Full write-up: `results/paperB/phase14/camelyon_b1/PHASE14_B1.md`.
