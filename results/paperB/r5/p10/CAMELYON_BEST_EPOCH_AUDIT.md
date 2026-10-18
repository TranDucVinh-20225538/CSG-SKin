# Camelyon checkpoint-selection audit (best by ID-select?)

## Selection rule by wave

| Wave | Script | Selects `best.pt` by | Eval metrics from |
|---|---|---|---|
| Phase12 Camelyon DANN | `train_phase12_camelyon_dann.py` | **ID select acc** (`sel_acc > best`) | `best.pt` |
| R2 item5 Camelyon2 | `train_r2_item5_camelyon2.py` | **ID select acc** (`sel > best`) | `best.pt` |
| R5 P10 + pilot | `train_r5_p10_camelyon_toolbox.py` | **ID select acc** (same) | `best.pt` |
| R5 P9 | derm single-encoder ladder (ISIC/PAD) | **not Camelyon** | N/A here |

All Camelyon Paper B / phase12 paths above use **best-by-ID-select**, unlike derm R1/R2 (`last.ckpt`).

## `best_epoch` counts

### R5 P10 pilot + shift_arms (2026-10-17/18)
**15 / 15 → `best_epoch = 1`** (adv warmup, GRL coef = 0). Locked gate / “invariance not reached” /
pilot failure numbers are **pre-adversary encoders**. See `BEST_EPOCHS.md`.

### R2 item5 (`results/paperB/r2/item5/runs/`)
| tag | best_epoch | λ |
|---|---:|---:|
| camelyon2_ladv0_s42 | 10 | 0 |
| camelyon2_ladv0_s43 | 6 | 0 |
| camelyon2_ladv0_s44 | 9 | 0 |
| camelyon2_ladv0p1_s42 | 6 | 0.1 |
| camelyon2_ladv0p3_s{42,43,44} | 8–10 | 0.3 |
| camelyon2_ladv1_s{42,43,44} | 8–9 | 1 |
| camelyon2_ladv3_s42 | 8 | 3 |
| camelyon2_ladv10_s42 | 10 | 10 |

**0 / 12 at epoch 1.** ID-select still chose mid/late epochs with high ID select; coef at best > 0 for λ>0.
Item5 Camelyon claims are **not** warmup-only (still ID-select, not last).

### Phase12 coarse (`phase12/camelyon17/coarse/`)
| tag | best_epoch | λ |
|---|---:|---:|
| dann_densenet121_ladv0_s42 | 1 | 0 |
| dann_densenet121_ladv0p1_s42 | 1 | 0.1 |
| dann_densenet121_ladv1_s42 | 2 | 1 |
| dann_densenet121_ladv10_s42 | 1 | 10 |

**3 / 4 at epoch 1** (λ=1 → epoch 2). Coarse DANN drafts that cite these runs are largely warmup / near-warmup encoders.

### Phase12 matched (`phase12/camelyon17/matched/`)
| tag | best_epoch | λ |
|---|---:|---:|
| matched_*_ladv0/10/100_s42 | 7 | 0 / 10 / 100 |
| matched_*_ladv30_s42 | 9 | 30 |
| matched_*_ladv300_s42 | 10 | 300 |

**0 / 5 at epoch 1.** Matched-hospital Camelyon claims are not warmup-only.

## Draft-correction implication

| Source | Warmup-only risk |
|---|---|
| P10 shift-arms + pilot | **Yes — must qualify / rewrite** locked Camelyon invariance language |
| Phase12 coarse | **Mostly yes** — treat as warmup/near-warmup |
| R2 item5 | **No** (best_epoch ≥ 6) — keep, but note ID-select ≠ last |
| Phase12 matched | **No** |
| P9 | Not Camelyon (derm) — out of scope for this warmup bug |

Trajectory smoke (natural s42 + pilot L4) will show whether any post-warmup epoch jointly clears site drop and ID; only then consider a new checkpoint-selection decisions file.
