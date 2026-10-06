# Phase 12.1b — adversary audit (no compute)

The coarse scan never induced invariance: 3-class leakage stayed 0.977 → 0.963. That is **balanced accuracy**; the chance floor is **1/3 ≈ 0.333**, not the plain-accuracy majority 0.437 (= 132k/302k, always hospital 4). P12.1 / P12.3 are not yet testable. The λ=10 Maha drop is not the derm signature (kNN stayed 0.82; leakage stayed 0.96).

The four coarse points are **void as evidence about the cliff**: they were not trained with the derm adversarial recipe. Coarse λ=0 is not a valid anchor for the matched-recipe runs.

This file compares the Camelyon17 DANN (coarse) against the derm CSG adversary that actually moved leakage (`src/models/csg_lite.py`, `src/models/csg_lightning.py`, `scripts/train_csg.py`, Phase 3 `train_phase3_sweep.py`).

Derm values are those used in Paper B Phase 3 (`CSGLiteLightning` defaults + `lr=1e-4`). `train_csg.py` defaults match except `lr=2e-4` and `lambda_orth=10`; the **adversary** knobs below are identical.

## Line-by-line

| Item | Derm (CSG / Phase 3) | Camelyon17 coarse | Match? |
|---|---|---|---|
| `adv_lr_multiplier` | **30.0** — domain head is its own AdamW group at `lr × 30` | **1.0** — one SGD group, domain head at the same `lr` as the encoder | **NO — prime suspect** |
| GRL λ schedule | DANN ramp: `α = 2/(1+exp(-γ p^k))-1`, `γ=20`, `k=0.5`, **`α_min=0.2`**. Applied every step via `set_grl_lambda(α)`. | Constant `1.0`. No ramp, no floor. | **NO** |
| Adversary head | `Linear → BN → ReLU → Dropout(0.5) → Linear → BN → ReLU → Dropout(0.5) → Linear`. Hidden `max(d,32)` then `max(d//2,16)` on **16-d** latent (32 / 16). | `Linear(1024,256) → ReLU → Linear(256,3)`. No BN, no Dropout. | **NO** |
| Domain loss weighting | `L = L_cls + λ_adv * CE(d_adv, domain)` (plus ctx/orth unused here) | `L = L_cls + λ_adv * CE(d, hospital)` | yes, the scalar form |
| Batch composition | Every step is **50/50** ISIC+PAD (`CombinedTrainDataset` + paired collate; PAD labels `ignore_index=-1` for `L_cls` only) | Natural hospital mix: train h0/h3/h4 = 53k / 117k / 132k. No balancing. | **NO** |
| Where GRL is inserted | On **BN-normalised** 16-d `z_lesion_norm` | On **raw** 1024-d DenseNet features | **NO** |
| Classifier LR | `lesion_cls_lr_multiplier=0.2` | Same LR as encoder | **NO** |
| Optimiser | AdamW, three param groups | SGD momentum 0.9, one group | **NO** |
| Base `lr` / `wd` | AdamW `1e-4` / `1e-4` (Phase 3) | SGD `1e-3` / `0.01` (WILDS-ish) | **NO** (intentional for ERM; will match derm for 12.1b) |

## Manuscript result (independent of how 12.1b resolves)

Seven of eight mismatches, with leakage frozen at 0.96 across four orders of magnitude of λ, is a result the derm paper should state:

A weak adversary produces neither debiasing nor the safety failure. Leakage stays high and OOD detection stays intact. The danger zone is precisely where the method succeeds.

A weak adversary produces neither the leakage drop nor the OOD collapse. Both are effects of a working adversarial objective; the collapse is **not** mediated by the leakage probe (Phase 3 opposite directions; Phase 13.2 backbone decoupling). Many deployed DANN setups are too weak to induce either effect. λ alone does not characterise adversarial pressure. This table is the supporting evidence of that last sentence.

Copied into `MASTER_REPORT.md` Discussion.

## Reading for the matched rerun

Raising λ_adv with a weak adversary cannot substitute for a strong one. Coarse λ=10 already weighted `L_adv` ten times `L_cls`, but the domain head trained at the encoder's SGD rate, without BN/Dropout, on unnormalised 1024-d features, on unbalanced hospital batches. The probe staying at 0.96 is the expected outcome of that recipe, not evidence that Camelyon stain is un-removable.

12.1b therefore **re-trains** λ ∈ {0, 10, 30, 100, 300} with the derm-matched adversary (does not reuse coarse checkpoints). λ=0 is the matched-recipe control: GRL/`L_adv` contribute no encoder gradient; BN, balanced batches, AdamW groups and classifier LR scaling still apply. Coarse `{0, 0.1, 1, 2, 4, 10}` dense grid is **not** submitted.

If leakage at matched λ=10 is already near the balanced-accuracy floor (1/3), the upward grid overshot: extend downward `{0.25, 0.5, 1, 2, 5}` rather than concluding from four post-cliff points. Not launched until 60636 + λ=0 report.

## Matched recipe locked for 12.1b

- DenseNet-121 still the backbone (the claim is about invariance training, not about swapping to EffNet).
- `feat_bn = BatchNorm1d(1024)` on backbone features. Classifier and GRL both see `z_norm`.
- Domain head: `Linear(1024,1024)→BN→ReLU→Dropout(0.5)→Linear(1024,512)→BN→ReLU→Dropout(0.5)→Linear(512,3)` (same template as derm, dimensions from `d=1024`).
- GRL on `z_norm`; schedule identical to `CSGLiteLightning._compute_grl_alpha`.
- AdamW, three groups: encoder+BN at `1e-4`; domain head at `1e-4 × 30`; classifier at `1e-4 × 0.2`. `wd=1e-4`.
- Hospital-balanced batches: 10 samples from each of {0, 3, 4} per step (batch 30).
- `λ_adv` still the scalar on `L_adv`. Seed 42. 10 epochs. Checkpoint on `id_val_select`. Metrics on `id_val_score`.
- Divergence at λ=100/300 is a training failure, not outcome (b).

Hard stop: if λ=300 still has leakage balanced acc > ~0.8, outcome **(c)**; do not escalate.
