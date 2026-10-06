# Phase 12.1 — launch record (coarse scan running)

Amendment F, A, and the 4-run coarse scan are done or in flight. Dense sweep is **not** launched.

## Amendment F — path verification

`results/paperB/phase12/pathcheck.json`

| Dataset | metadata rows checked | missing |
|---|---:|---:|
| Camelyon17-WILDS v1.0 | 455,954 | **0** |
| iWildCam-WILDS v2.0 | 203,029 | **0** |

RELEASE markers recorded with SHA-256. Data root is read-only:

`/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds`

## Amendment A — `id_val` split (seed 42, hospital × label)

`results/paperB/phase12/camelyon17/id_val_split.json`

| Half | n | role |
|---|---:|---|
| `id_val_select` | 16,780 | checkpoint selection only |
| `id_val_score` | 16,780 | ID accuracy + ID reference for all OOD detectors |

Disjoint. Union is official `id_val` (33,560). No `id_test` was constructed.

## Coarse scan (running)

SLURM array **60623** (afterok aggregate **60624**). One seed (42), λ ∈ {0, 0.1, 1, 10}. DenseNet-121 DANN. Checkpoint on `id_val_select`. OOD columns: hospital 1 (`val`) and hospital 2 (`test`).

Dense sweep waits for this readout.
