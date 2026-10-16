# Paper B R5 D1 follow-up — placebo and dose

Decisions: `decisions/paperB_r5_d1_followup.md`. Commit `b990fb5`. **Features: lesion-level item4 only** (no phase13).

backbone = `backbone_raw_lesion` (pre-projection features of the same DANN-trained model; reversed gradient). pull = change from λ = 0 to λ = k. Domain score from the source λ = 0 model.

## (1) Placebo: λ0→λ2 vs λ0→λ0 (different seed)

Pairs: 42→52, 52→62, 62→72, 72→82, 82→42. DANN-specific if Δρ_dom ≥ 0.15 and CI entirely above 0 in ≥ 4/5 sets (latent).

| set | rep | ρ_dom DANN (λ0→λ2) | ρ_dom placebo | Δρ_dom [CI] | DANN-specific |
|---|---|---|---|---|---|
| pad_heldout | latent | +0.344 | +0.288 | +0.057 [+0.007, +0.105] | not met |
| pad_heldout | backbone | +0.439 | +0.109 | +0.331 [+0.283, +0.374] | met |
| Fitzpatrick17k | latent | +0.419 | +0.205 | +0.214 [+0.196, +0.232] | met |
| Fitzpatrick17k | backbone | +0.475 | +0.092 | +0.383 [+0.364, +0.400] | met |
| MED-NODE | latent | +0.433 | +0.106 | +0.327 [+0.243, +0.407] | met |
| MED-NODE | backbone | +0.443 | +0.093 | +0.349 [+0.256, +0.432] | met |
| SD-198 | latent | +0.467 | +0.203 | +0.264 [+0.250, +0.278] | met |
| SD-198 | backbone | +0.540 | +0.091 | +0.449 [+0.436, +0.461] | met |
| SCIN | latent | +0.333 | +0.150 | +0.183 [+0.172, +0.193] | met |
| SCIN | backbone | +0.317 | +0.078 | +0.239 [+0.227, +0.251] | met |

Placebo verdict: **4/5** latent → DANN-specific supported (backbone 5/5).

## (2) Dose — pending mid-λ lesion-level features

λ0→λ2 ρ_dom is in the placebo DANN column above. Mid-λ dose (0.25…8) waits for lesion-level item4 features at matched seeds 42–82 (`r5_d1_followup_extract.py` after training). phase3/phase13 image-level caches are not used.

