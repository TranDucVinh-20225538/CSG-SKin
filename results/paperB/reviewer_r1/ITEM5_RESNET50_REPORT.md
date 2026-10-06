# Item 5 — ResNet-50 single-encoder DANN (manuscript readout)

**Protocol:** `pad_heldout` leakage probe (ISIC test vs PAD held-out, 16-d embedding). EffNet-B3 single-encoder comparison below uses **`pad_full`** (locked Phase 3 primary) unless noted — same probe family, slightly different OOD pool size.

## Table A — ResNet-50 (3 seeds, mean)

| λ_adv | leakage (bal) | Mahalanobis | kNN-50 | cosine |
|------:|--------------:|------------:|-------:|-------:|
| 0 | **0.965** | 0.944 | 0.962 | 0.922 |
| 0.5 | **0.939** | 0.814 | 0.563 | 0.543 |
| 2 | **0.936** | 0.766 | 0.640 | 0.568 |

Per-seed leakage @ λ=2: 0.928, 0.943, 0.935.

## Table B — EffNet-B3 single-encoder (`phase15/single_dann`, pad_heldout)

| λ | leakage | Maha | kNN | cos |
|--:|--------:|-----:|----:|----:|
| 0 | 0.952 | 0.700 | 0.935 | 0.875 |
| 0.5 | 0.742 | 0.524 | 0.539 | 0.519 |
| 2 | 0.760 | 0.521 | 0.542 | 0.514 |

EffNet **`pad_full`** leakage (manuscript primary): λ=0 **0.970** → λ=0.5 **0.795** → λ=2 **0.807**.

## Interpretation (aligned with PI readout)

1. **Non-mediation (ResNet):** kNN/cosine collapse λ=0→0.5 (**Δ≈0.38–0.40**) while leakage stays **~0.94** (floor 0.5). Strongest decoupling in the project — not mediated by measured invariance.

2. **No cross-backbone inversion:** Mahalanobis **0.94→0.77**, stays **above chance**; collapse without sub-chance inversion.

3. **Backbone (do not over-claim):** ResNet never reaches EffNet-like invariance on the same `pad_heldout` probe (0.965→**0.939** vs 0.952→**0.742** @ λ=0.5). At the only **valid** ResNet point (λ=0.5), Mahalanobis has **not** inverted — the backbone question stays **open**, not “inversion is EffNet-only.”

4. **Primary Item 5 evidence @ λ=0.5:** Gate `valid`, adversary learned, leakage Δ≈**0.026** while kNN/cosine drop ≈**0.40** — cleanest **non-mediation** in the project (single run family, not cross-table comparison).

5. **Exclude λ=2 ResNet entirely:** Gate `inconclusive` (CE ~0.686, null DANN). Do **not** cite Maha/kNN @ λ=2 as support or refutation.

## Adversary gate (must resolve before § text)

Summaries initially showed `gate.status=inconclusive` / `no adversary epoch log` because training logged **step** JSON, not **epoch** rows expected by `adversary_gate`.

**Rescore:** `scripts/rescore_resnet50_gate.py` on existing `adversary_log.jsonl` (1012 step lines → 40 epoch means).

| λ | Typical gate (3 seeds) | Meaning |
|--:|------------------------|---------|
| 0 | `valid_control` | λ_adv=0 control |
| 0.5 | **`valid`, learned=True** | Adversary trained; encoder + GRL active — **use this λ for non-mediation** |
| 2 | **`inconclusive`, learned=False** | Adversary CE never meaningfully below ln K — treat λ=2 ResNet as **null / inconclusive DANN**, not evidence against PI story |

**Do not** cite λ=2 ResNet Mahalanobis/kNN until gate is `valid`. **Do** cite λ=0.5 with rescored gate + flat leakage + detector collapse.

Raw logs: `reviewer_r1/resnet50_single_dann/*/adversary_log.jsonl`; CSV: `lightning_logs/version_*/metrics.csv` (`train/adv_acc` if enabled in future runs).
