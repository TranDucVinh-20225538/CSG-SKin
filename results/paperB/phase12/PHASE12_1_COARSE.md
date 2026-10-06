# Phase 12.1 — Camelyon17 coarse λ scan

**Correction (12.1b).** These four points are **void as evidence about the cliff**. They used the unmatched recipe (SGD, `adv_lr_multiplier=1`, no feature BN, unbalanced batches, GRL on raw features). Coarse λ=0 is **not** comparable to the matched-recipe runs. The proposed dense grid `{0, 0.1, 1, 2, 4, 10}` was **not** submitted.

Leakage below is 3-class **balanced** accuracy. Chance floor is **1/3 ≈ 0.333**, not 0.437 (plain-accuracy majority). Leakage never moved (0.977 → 0.963 vs floor 0.333). That is the only load-bearing readout of this table: a weak adversary did not induce invariance.

The λ=10 Maha drop (0.671 → 0.482) is **not** the derm signature: kNN stayed 0.816 and leakage stayed 0.963. Do not report it as a replication.

P12.2 (ID acc flat 0.987 → 0.981) and P12.5 (transfer flat) hold **under the weak recipe**. They are not matched-recipe claims.

Single seed (42). DenseNet-121 DANN. Checkpoint on `id_val_select` (n=16,780). ID metrics on `id_val_score` (n=16,780). Not leaderboard-comparable.

**Camelyon17 is binary.** MSP lives in [0.5, 1] and is not the replication criterion. Lead with Mahalanobis (class-conditional, shared covariance) and kNN (k=50).

Hospitals 1 (`val`) and 2 (`test`) were never in the adversary. Composition is 50/50 on train / id_val / test — no class-mix restriction needed.

| λ_adv | ID acc (`id_val_score`) | leak 3-class bal acc | Maha AUROC h2 (test) | kNN AUROC h2 | Maha AUROC h1 (val) | kNN AUROC h1 | xfer acc h2 | xfer acc h1 | MSP h2 |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | 0.987 | 0.977 | 0.690 | 0.822 | 0.591 | 0.765 | 0.874 | 0.889 | 0.721 |
| 0.1 | 0.988 | 0.969 | 0.707 | 0.863 | 0.628 | 0.783 | 0.832 | 0.881 | 0.805 |
| 1 | 0.984 | 0.970 | 0.671 | 0.830 | 0.629 | 0.777 | 0.888 | 0.900 | 0.736 |
| 10 | 0.981 | 0.963 | 0.482 | 0.816 | 0.552 | 0.752 | 0.864 | 0.854 | 0.702 |

## Where the transition sits

Not interpretable from this table. Dense grid `{0.0, 0.1, 1.0, 2.0, 4.0, 10.0}` is **withdrawn**. See `PHASE12_1B_AUDIT.md` and the matched-recipe rerun.

**STOP.** Dense sweep is not started. iWildCam not started.

Path verification (Amendment F): see `pathcheck.json`. Camelyon 0 missing / 455,954; iWildCam 0 missing / 203,029.

