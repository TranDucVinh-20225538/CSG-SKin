# Block B — ERM → dual initialization audit

Date: 2026-11-09. Seed-matched ERM: `checkpoints/r5_p8/erm_ref/single_dann_ladv0_s{seed}/last.ckpt`.

## Mapping (verified shapes)

| ERM key | Dual key(s) | n tensors | action |
|---|---|---|---|
| `net.backbone.features.*` | `model.lesion_backbone.features.*` | 572 | copy |
| `net.backbone.features.*` | `model.context_backbone.features.*` | 572 | copy (same ERM weights into both branches) |
| `net.feat_bn.*`, `net.classifier.*` | `model.lesion_bn` / `lesion_projector` / `lesion_classifier` | — | **not loaded** (dim 1536 vs latent 16) |
| `net.domain_head.*` | `model.domain_classifier_adv.*` | — | **not loaded** (1536-d vs 16-d) |

Total mapped: **1144** tensors (572×2). Heads / projectors / BN / domain classifier remain randomly initialized under the job seed.

## Safety

- No silent partial load without audit: each Block B run writes `runs/<tag>/erm_init_audit.json`.
- Strict shape check before `load_state_dict(..., strict=False)`.
- Block B is **not blocked**.
- This is **not** X3 (encoders are trainable; X3 was frozen-head Case C).

## Comparators

Matched from-scratch controls: Block A λ∈{0,2} × prior100 × same five seeds (same fixed-image pipeline).
Pure ERM reference remains `r5/p8/erm_ref` (single encoder); dual λ=0 is not interchangeable with pure ERM.
