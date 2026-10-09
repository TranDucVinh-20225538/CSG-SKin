# R3 A4 — PAD-reweight discrepancy

Commit: `fca1ce6`.

**Verdict:** The identity reproduces 0.240 -> the manuscript states that the per-class decomposition reproduces all three manipulations; the wrong explanation is deleted outright.

## Provenance of 0.240 (Phase 2.5b)

| Question | Answer |
|---|---|
| Source | `results/paperB/phase2_5/per_seed/runB_orth1_s*.json` → `ood_sets.pad.mahalanobis_classcond.id_reweighted_to_pad`; aggregated in `PHASE2_5_REPORT.md` 2.5b (0.240 ± 0.014); code `scripts/eval_phase25.py::weighted_auroc` |
| PAD mix | **pad_full** (n = 2298; `phase2_5/class_priors.json`), not pad_heldout (716) |
| Model / λ | `runB_orth1` Phase 1 checkpoints, λ_adv = 2, image-level ISIC split |
| Seeds | 42, 52, 62, 72, 82 (n = 5) |
| Weighting | exact `sample_weight` in `roc_auc_score`, w = p_pad(c) / p_isic(c) per ID image, OOD weight 1; no resampling |
| Six-class restriction | implicit, inside the weighting: DF and VASC get w = 0 because p_pad = 0 (no separate restriction step) |
| Seed s.d. | 0.014 (n = 5) |

PAD class mixes: pad_full MEL 0.023, NV 0.106, BCC 0.368, AK 0.318, BKL 0.102, SCC 0.084; pad_heldout MEL 0.013, NV 0.123, BCC 0.385, AK 0.282, BKL 0.105, SCC 0.092.

## Identity under the exact protocol (pad_full), mean ± s.d. over 5 seeds

| Quantity | Value |
|---|---|
| Stored reweighted AUROC (the 0.240) | 0.2399 ± 0.0144 |
| Recomputed reweighted AUROC, same features | 0.2399 ± 0.0144 |
| Σ_c w_c AUROC_c, six classes, pad_full weights, per-class AUROC on pad_full | 0.2399 ± 0.0144 |
| max per-seed \|identity − stored\| | 8.1e-07 |
| Two-group form (NV vs non-NV lumped at ISIC proportions), pad_full weights | 0.3060 ± 0.0163 |
| Stored unrestricted AUROC on pad_full | 0.4088 ± 0.0220 |

## Where 0.2585 comes from

The 0.2585 prediction combines pad_heldout weights with per-class AUROCs on pad_heldout and is compared with a measurement on pad_full. Same five seeds, Phase 13 λ = 2 features for pad_heldout:

| Weights | Per-class AUROCs on | Σ w_c AUROC_c | Measured reweighted AUROC (same OOD set) |
|---|---|---|---|
| pad_full | pad_full | 0.2399 ± 0.0144 | 0.2399 ± 0.0144 |
| pad_heldout | pad_heldout | 0.2585 ± 0.0222 | 0.2585 ± 0.0222 |
| pad_heldout | pad_full | 0.2458 ± 0.0144 | — |
| pad_full | pad_heldout | 0.2522 ± 0.0219 | — |
| two-group, pad_heldout weights | pad_heldout | 0.3286 ± 0.0224 | — |

## Per-class Mahalanobis AUROC (ID class c vs OOD), mean ± s.d.

| Class | vs pad_full (Phase 1 ckpt) | vs pad_heldout (Phase 13 λ=2) |
|---|---|---|
| MEL | 0.364 ± 0.022 | 0.388 ± 0.025 |
| NV | 0.540 ± 0.032 | 0.559 ± 0.029 |
| BCC | 0.237 ± 0.020 | 0.257 ± 0.031 |
| AK | 0.155 ± 0.015 | 0.158 ± 0.018 |
| BKL | 0.257 ± 0.015 | 0.270 ± 0.024 |
| DF | 0.130 ± 0.021 | 0.115 ± 0.022 |
| VASC | 0.162 ± 0.029 | 0.157 ± 0.053 |
| SCC | 0.141 ± 0.015 | 0.142 ± 0.015 |

## Caveats

- Reweighting the ID set changes only w_c; model, scores and OOD set are unchanged, so every AUROC_c is unchanged. The manuscript sentence attributing the gap to shifted per-class AUROCs is deleted in every outcome.
- The pad_heldout rows use Phase 13 features of the λ = 2 sweep runs (the same Phase 1 checkpoints re-extracted); the pad_full rows use the Phase 2.5 features.
