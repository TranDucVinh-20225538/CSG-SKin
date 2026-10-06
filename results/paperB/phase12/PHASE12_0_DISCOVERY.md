# Phase 12.0 — Dataset discovery

**Gate. No Phase 12.1 training. No download.** Prior `results/paperB/` files were not modified. New files live only under `results/paperB/phase12/`.

**Verdict: both WILDS datasets are already on this cluster, extracted, and load with `download=False`. Nothing over 10 GB needs to be fetched. Waiting for confirmation before 12.1.**

Machine-readable dump: `phase12_0_discovery.json`. Licences: `LICENCES.md`.

---

## What was searched

| Location | Result |
|---|---|
| `/data2/hpcshared` (maxdepth 6) | No WILDS trees under Vinh/CSG-Skin. Alternate Camelyon copy under Toan (not used). |
| `/data2/cmdir/home/toandq` | **Hit.** DST-Skin already holds both datasets. |
| `/data2/hpcshared/Vinh/CSG-Skin/data` | ISIC / PAD only. No Camelyon, no iWildCam. |
| `/scratch`, `/work`, `/datasets`, `/data` | Do not exist on this host. |
| `module avail` | No WILDS / Camelyon / iWildCam module. |
| Site dataset registry | None found. |

Existing copy (used for the split probe, read-only):

```
/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds/camelyon17_v1.0
/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds/iwildcam_v2.0
```

Alternate Camelyon tree (not used): `/data2/hpcshared/Toan/Causal_XAI_CAMELYON17/data/data/camelyon17_v1.0` (RELEASE marker present).

---

## Environment

| Item | Status |
|---|---|
| Paper B venv python | `/data2/cmdir/home/toandq/CSG-Skin-paperB/.venv/bin/python` → conda `torch-env` 3.11.5 / torch 2.1.1 |
| `wilds` | **already installed**, version **2.0.0** (`pip show wilds`). `pip install wilds` was not run. |
| Load call | `wilds.get_dataset(..., download=False, split_scheme="official")` |

---

## Disk / quota (before any download)

The work order's 90 GB iWildCam figure is the **original competition dump**. WILDS v2.0 is a height-448 JPEG redistribution. Compressed size in `wilds` is 11,957,420,032 bytes (~11.1 GiB). That copy is already extracted.

| Dataset | On disk | Files | Marker |
|---|---:|---:|---|
| Camelyon17-WILDS v1.0 | **9.92 GiB** | 455,956 | `RELEASE_v1.0.txt` |
| iWildCam-WILDS v2.0 | **11.32 GiB** | 217,644 | `RELEASE_v2.0.txt` |

Volume: `/data2` is 175 T, **39 T free** (79% used). `quota -s` returns no per-user cap ("Please ask your administrator"). Home `/data2/cmdir/home/toandq` is 1.5 T on the same filesystem.

**No download is required. The >10 GB confirmation gate is therefore not triggered.**

---

## Official splits actually obtained (`download=False`)

Spot-check: 200 random files per dataset, **0 missing**.

### Camelyon17 — 455,954 patches, 5 hospitals, 2 classes (tumor / normal), 96×96

WILDS numbering is 0-indexed. Test hospital is **2** (paper's "hospital 5"). Val hospital is **1**. Training hospitals are **0, 3, 4**.

There is **no `id_test` split**. ID evaluation is `id_val`. Do not invent one.

| Split | WILDS name | n | Hospitals | Labels 0 / 1 |
|---|---|---:|---|---|
| `train` | Train | 302,436 | 0, 3, 4 | 151,046 / 151,390 |
| `id_val` | Validation (ID) | 33,560 | 0, 3, 4 | 16,952 / 16,608 |
| `val` | Validation (OOD) | 34,904 | **1 only** | 17,452 / 17,452 |
| `test` | Test (OOD) | 85,054 | **2 only** | 42,527 / 42,527 |

Protocol mapping for 12.1 (not yet run):

- Fit Maha / kNN / centroids on **`train` only**.
- ID accuracy / ID OOD-detector in-distribution scores: **`id_val`** (training hospitals).
- OOD AUROC and cross-domain accuracy: **`test`** (held-out hospital 2).
- `val` (hospital 1) is a second OOD hospital. Using it for early stopping would peek at an unseen hospital. **Propose checkpointing on `id_val` only**, and report `val` as an extra OOD column if computed. Confirm this before 12.1.

Domain head: 3 training hospitals (not 5). GRL multi-class over {0, 3, 4}.

### iWildCam v2.0 — 203,029 images, 182 species, 323 camera-trap locations

`n_classes=182` as loaded. DST-Skin's constant `IWILDCAM_NUM_CLASSES = 186` is stale; do not copy it.

Train–OOD-test location overlap: **0**. Train–ID-test location overlap: 164 (same cameras, different days — as designed).

| Split | WILDS name | n | Locations | Classes present |
|---|---|---:|---:|---:|
| `train` | Train | 129,809 | **243** | 182 |
| `id_val` | Validation (ID/Cis) | 7,314 | 146 | 71 |
| `id_test` | Test (ID/Cis) | 8,154 | 164 | 87 |
| `val` | Validation (OOD/Trans) | 14,961 | 32 | 75 |
| `test` | Test (OOD/Trans) | 42,791 | 48 | 102 |

Domain head: **243** training locations. The work order's "~240" is this number. Full multi-class GRL first; cluster to 10–20 only if training is unstable. That choice will be reported if 12.2 is reached.

12.2 remains gated on P12.1. Not started.

---

## Licences (new + existing)

Recorded in `LICENCES.md`. Short form:

| Dataset | Licence | Cite |
|---|---|---|
| WILDS code | MIT | Koh, Sagawa, et al., ICML 2021 |
| Camelyon17-WILDS | **CC0** (public domain); CAMELYON17 challenge terms for the original WSIs | Bandi et al., IEEE TMI 2018 + Koh et al. 2021 |
| iWildCam-WILDS | **CDLA-Permissive 1.0** | Beery, Cole, Gjoka, arXiv:2004.10340 + Koh et al. 2021 |
| ISIC 2019 | CC BY-NC 4.0 | Tschandl / Combalia / ISIC 2019 |
| PAD-UFES-20 | CC BY 4.0 | Pacheco et al., 2020 |
| Fitzpatrick17k | annotations CC BY-NC-SA 3.0; images mixed web copyright, eval-only, not redistributed | Groh et al., CVPRW 2021 |

---

## What 12.1 will use if confirmed (not started)

- Architecture: single-encoder DANN, WILDS reference **DenseNet-121**, not CSG dual-encoder.
- Data root: the DST-Skin WILDS path above (read-only). Outputs only under `results/paperB/phase12/camelyon17/`.
- Coarse λ scan, 1 seed: `{0, 0.1, 1, 10}`. Then locate the cliff. Do not assume the derm grid `{0, 0.25, …, 8}`.
- Seeds / detectors / four-column table: as specified in the Phase 12 work order.
- Derm GPU array `60585` (Paper B final package) is still running and is not preempted by this phase.

---

## STOP

Phase 12.0 is complete. **Phase 12.1 is not launched.**

Confirm:

1. Use the existing DST-Skin WILDS copies (no re-download).
2. Camelyon ID eval = `id_val` (no `id_test` exists).
3. Checkpoint on `id_val`, not on OOD `val` hospital 1.
4. Start 12.1 only after the derm final-package GPU array has released cards, or accept sharing the 8-GPU QOS cap.

A FAIL on P12.1 is a finding. If the cliff does not replicate, the paper's claim narrows; it is not retuned toward a prediction.
