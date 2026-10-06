# Phase 14.B1 — Camelyon adversary saturation

Existing 12.1b logs. Gradient norms were not logged; they are measured on saved `best.pt` over 8 hospital-balanced batches. This is not a retrain.

**Finding: the adversary was never trained.** train_adv_acc sits at 3-class chance (~0.33) for every matched λ including 10, 100 and 300, from epoch 1 through 10, and checkpoint CE equals ln(3). A learning head starts near ln(3) and improves. This one never did. ||g_enc through GRL|| = 0 is the symptom of a constant uniform head. The post-hoc probe still reads 0.94.

This is **not** outcome (c) as “Camelyon resists invariance.” Phase 12 is inconclusive. No Camelyon number is reported.

B2 launched: **no**. GRL sign check (separate): head is **not** flipped. See `PHASE14_B_GRL_SIGN.md`.

iWildCam stays gated.

## Training-log adversary (every epoch)

| λ | leak 3-cls | ID acc | adv acc epoch-1 | adv acc last | adv acc max | train loss last |
|---:|---|---|---|---|---|---|
| 0 | 0.975 | 0.993 | 0.326 | 0.342 | 0.346 | 0.008 |
| 10 | 0.942 | 0.992 | 0.354 | 0.333 | 0.354 | 11.002 |
| 30 | 0.947 | 0.989 | 0.346 | 0.333 | 0.346 | 32.986 |
| 100 | 0.943 | 0.979 | 0.360 | 0.335 | 0.360 | 110.067 |
| 300 | 0.894 | 0.979 | 0.353 | 0.333 | 0.353 | 329.622 |

Chance for a 3-class hospital-balanced batch is 1/3. A discriminator pinned at 1.0 would be the vanishing-gradient suspect. Every λ, including the λ=0 control (domain loss off), sits at chance. At λ>0 the domain CE is in the graph and the head still never leaves 0.33–0.36.

## Checkpoint gradient norms (8 batches, CPU, eval-transform)

| λ | GRL α | adv acc | adv loss | ||g_enc from L_cls|| | ||g_enc through GRL|| | GRL / L_cls |
|---:|---|---|---|---|---|---|
| 10 | 1.000 | 0.333 | 1.099 | 0.2699 | 0 | 0.000 |
| 100 | 1.000 | 0.333 | 1.099 | 0.7139 | 0 | 0.000 |
| 300 | 1.000 | 0.333 | 1.099 | 1.293 | 0 | 0.000 |

GRL path is `λ_adv * CE(domain_head(GRL_α(z)))`. A saturated discriminator (acc→1, CE→0) would drive ||g_enc through GRL|| → 0. Here CE is exactly ln(3) and acc is exactly 1/3: the head is a constant uniform predictor, so ∂L_adv/∂z = 0 and the encoder never feels the adversary. That is not the B2 trigger.

## Decision table

| Pre-registered finding | Observed |
|---|---|
| Adversary pinned near 1.0 and GRL gradient → 0 | **No.** Adv acc ≈ 0.33, CE = ln(3). GRL ||g|| is 0 because the head is constant, not because it saturated |
| Adversary not saturated, leakage still flat | **Yes.** Leakage 0.942 / 0.943 / 0.894 at λ=10 / 100 / 300. Encoder never felt a competent adversary |

B2 is **not** run. An adversary that never leaves chance is not the discriminator-too-strong failure mode, and it is also not a finding about Camelyon stain. Phase 12 is dropped from Results. Limitations: tried, implementation unresolved, left for later work.

