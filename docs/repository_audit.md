# CSG-Skin — Repository Audit (Paper 3, Phase 1)

**Audit date:** 2026-08-01
**Repository root:** `/mnt/data2/Vinh/CSG-Skin`
**Method:** direct file reads, filesystem enumeration, checkpoint tensor inspection, metadata recomputation. No code was modified. No experiments were run. Everything below was verified against the filesystem; anything not found on disk is explicitly marked **MISSING**.

**Verification environment** (from `.venv`, actually installed — `requirements.txt` is out of date, see §9):
torch 2.11.0+cu130 · torchvision 0.26.0 · pytorch-lightning 2.6.1 · torchmetrics 1.9.0 · scikit-learn 1.8.0 · numpy 2.4.4 · pandas 3.0.2 · scipy 1.17.1 · seaborn 0.13.2 · opencv 4.13.0 · GPU: 1× NVIDIA A100-SXM4-80GB.

**Repository facts:**
- Not a git repository (no `.git`). There is a `.gitignore`, but no version history exists.
- No `README`, no `CLAUDE.md`, no `docs/` prior to this file.
- Total size 31 GB (data 9.6 GB, checkpoints 16 GB).
- `notebooks/` contains only `.gitkeep` — **no notebooks exist**.

---

## 1. Repository structure

```
CSG-Skin/
├── .gitignore                       # ignores checkpoints/, lightning_logs/, data/raw, data/processed
├── requirements.txt                 # 7 lines, INCOMPLETE (see §9)
├── results_final_v1.csv             # frozen n=3 main table (4 methods)
├── RESULTS_TABLES.csv               # n=3 aggregate + per-seed table (hand-assembled, see §9)
│
├── configs/                         # baseline.yaml, csg.yaml — DEAD CONFIG, never loaded (see §9)
├── notebooks/                       # empty (.gitkeep only)
│
├── src/
│   ├── datasets/
│   │   ├── constants.py             # LABELS = [MEL,NV,BCC,AK,BKL,DF,VASC,SCC]; ImageNet mean/std
│   │   ├── preprocess_metadata.py   # builds data/master_metadata.csv (ISIC one-hot + PAD map)
│   │   ├── skin_dataset.py          # SkinDataset, CombinedTrainDataset, collate, SkinDataModule (465 L)
│   │   ├── splits.py                # load_filtered_master, build_id_ood_test_dataloaders
│   │   ├── csg_train_dataset.py     # CSGTrainDataset — UNUSED by any script (see §9)
│   │   ├── isic.py, pad_ufes.py     # 2-line stubs (pointer comments only)
│   ├── models/
│   │   ├── baseline.py              # BaselineNet (ResNet50) + BaselineResNet50 LightningModule
│   │   ├── csg_lite.py              # CSGLite dual-encoder + GradientReversalLayer
│   │   ├── csg_lightning.py         # CSGLiteLightning — 5-term loss, GRL schedule, SupCon
│   │   ├── effb3_single.py          # EffB3SingleNet / EffB3SingleLightning (backbone control)
│   │   ├── backbone.py, csg_full.py # 2-line stubs. csg_full is NOT implemented.
│   ├── losses/
│   │   ├── csg_losses.py            # ClassificationLoss, ContextLoss, AdvLoss, IndependenceLoss
│   │   ├── indep_loss.py, adv_loss.py  # 2-line stubs
│   └── utils/
│       ├── ood_metrics.py           # Mahalanobis, MSP/Energy table, FPR@95, find_checkpoint
│       ├── paths.py, seed.py
│       └── metrics.py, logging_utils.py, visualization.py  # 2-line stubs
│
├── scripts/                         # main pipeline (see §6)
│   ├── train_baseline.py  train_csg.py
│   ├── check_leakage.py   eval_ood_scores.py   aggregate_results.py
│   ├── build_lesion_only_metadata.py  verify_data_layout.py  verify_isic_labels.py
│   ├── run_full_benchmark.sh
│   └── generate_paper_figures.py  plot_confusion_matrix_runb.py  plot_reliability_diagram.py
│       visualize_soft_cropping_examples.py  generate_workflow_framework.py
│
├── cbm_revision/scripts/            # journal-revision layer (n=5 + backbone control)
│   ├── run_one_shot_cbm.sh          # master runner: 5 seeds × 3 methods → aggregate → OOD → stats → figs
│   ├── run_effb3_control.py / .sh   # EfficientNet-B3 single-encoder control
│   ├── eval_ood_benchmarks.py       # MSP/Energy/Cosine/Maha × AUROC/AUPR/FPR95
│   ├── stat_tests_cbm.py            # Welch t + Wilcoxon + Cohen's d
│   ├── plot_cbm_figures.py  write_final_summary.py  make_effb3_publication_files.py
│
├── data/                            # see §1.1
├── checkpoints/                     # 16 GB, 70 .ckpt files — see §3
├── lightning_logs/                  # 34 version dirs (2 empty) — see §5.5
└── results/                         # all JSON/CSV/MD/figures — see §5
```

### 1.1 `data/` layout (verified)

| Path | Contents | Verified count / note |
|---|---|---|
| `data/master_metadata.csv` | `path,label,domain` — original ISIC JPEGs + PAD PNGs | 27,629 rows (25,331 isic + 2,298 pad_ufes); **100 % of paths exist on disk** |
| `data/master_metadata_lesion_only.csv` | lesion-cropped paths | 27,629 rows |
| `data/master_metadata_lesion_only_soft.csv` | lesion-cropped paths | 27,629 rows — **byte-identical to `master_metadata_lesion_only.csv`** (`cmp` reports IDENTICAL); the "aggressive" build was overwritten by the "soft" build in the same output dir |
| `data/ISIC_2019_Training_Metadata.csv` | demographics only (age/site/lesion_id/sex), **no class columns** | 25,331 rows |
| `data/ISIC_2019_Training_Groundtruth.csv` | 8-class one-hot + UNK | 25,330 rows |
| `data/ISIC_2019_Training_Input/ISIC_2019_Training_Input/` | original ISIC JPEGs | 25,333 files, 9.2 GB |
| `data/lesion_only_images/ISIC_2019_Training_Input/` | cropped ISIC | 25,331 files |
| `data/lesion_only_images/images/` | cropped PAD-UFES | 2,298 files (421 MB total for both) |
| `data/pad_ufes20/images` | **symlink** → `/mnt/data2/Vinh/Ban_sao_datn/data/raw/pad_ufes20/images` | external dependency |
| `data/pad_ufes20/metadata.csv` | **symlink** → same sibling repo | external dependency |
| `data/raw/`, `data/processed/` | `.gitkeep` only | **empty** |
| `data/models/` | 4 `.pth` files from a prior project | see §3.4 |
| `data/reader_study/` | 2 CSVs | see §5.6 — **orphaned**, no script in this repo produces or reads them |

Class distribution (from `master_metadata_lesion_only_soft.csv`, recomputed):

| label | ISIC | PAD-UFES |
|---|---:|---:|
| MEL | 4,522 | 52 |
| NV | 12,875 | 244 |
| BCC | 3,323 | 845 |
| AK | 867 | 730 |
| BKL | 2,624 | 235 |
| DF | 239 | **0** |
| VASC | 253 | **0** |
| SCC | 628 | 192 |

PAD-UFES has **no DF and no VASC** samples — a structural asymmetry that matters for any class-conditional cross-domain analysis.

---

## 2. Training pipeline

### 2.1 Data preprocessing chain

```
ISIC_2019_Training_Metadata.csv  +  ISIC_2019_Training_Groundtruth.csv
        │                                      pad_ufes20/metadata.csv (symlink)
        └──────────► src/datasets/preprocess_metadata.py ◄─────────────┘
                              │   PAD_DIAG_MAP: BCC→BCC, MEL→MEL, NEV→NV,
                              │                 ACK→AK, SEK→BKL, SCC→SCC
                              ▼
                     data/master_metadata.csv        (27,629 rows, original images)
                              │
                              ▼  scripts/build_lesion_only_metadata.py  (preset=soft)
                              │  border-median diff → Otsu → tight bbox
                              │  → margin 0.30 → min_crop_ratio 0.70 → resize 224 → JPEG q95
                              │  (autocontrast OFF in soft preset)
                              ▼
        data/master_metadata_lesion_only_soft.csv    ← metadata used by ALL final experiments
```

`scripts/verify_isic_labels.py` re-checks master rows against the GroundTruth one-hot (stdout only, no artifact). `scripts/verify_data_layout.py` checks expected paths exist (stdout only).

### 2.2 Splits (`SkinDataModule.setup` and `splits.build_id_ood_test_dataloaders` — identical logic, both `random_state=42`)

Two nested stratified splits on ISIC only; PAD is never split.

| Split | Source | n (recomputed) |
|---|---|---:|
| ISIC train | 64 % of ISIC | **16,211** |
| ISIC val | 16 % of ISIC | **4,053** |
| ISIC test (= ID test) | 20 % of ISIC | **5,067** |
| PAD-UFES (= OOD test) | 100 % of PAD | **2,298** |
| `dm.test_dataloader()` (leakage-probe set) | ISIC test + PAD | **7,365** |

The split seed is **fixed at 42** (`SplitConfig.random_state`) and is *not* tied to `--seed`; the per-seed runs (42/52/62/72/82) vary model init and data order only, never the split. This is intentional and consistent across every script.

> **Critical:** `SkinDataModule` builds the CSG training stream as `CombinedTrainDataset(isic_train, self.pad_df_all, …)` where `pad_df_all` is **all 2,298 PAD rows**, and `build_id_ood_test_dataloaders` returns `ood_test_ds = SkinDataset(pad_df)` — **also all 2,298 PAD rows**. The CSG OOD test set is 100 % contained in the CSG training set. See §9.1.

### 2.3 Transforms (`src/datasets/skin_dataset.py`)

| Name | Pipeline |
|---|---|
| `build_train_transform_robust` | Resize 256 → RandomResizedCrop 224 (0.7–1.0) → HFlip → VFlip → Rot 20° → ColorJitter(.3,.3,.3,.08) → ToTensor → ImageNet norm → RandomErasing(p=.25) |
| `build_val_transform_robust` | Resize 256 → CenterCrop 224 → ToTensor → ImageNet norm |
| `build_lesion_branch_transform_gray` | Resize 256 → CenterCrop 224 → **Grayscale(3ch)** → ToTensor → ImageNet norm |
| light path (`use_robust_transforms=False`) | train: Resize(224,224) + mild jitter + HFlip; eval: Resize(224,224) |

`baseline_soft` runs use the **light** path (`--no_robust_transforms`). All CSG and EffB3 runs use the **robust** path. This asymmetry causes a reproducible metric discrepancy — see §9.3.

### 2.4 Baseline training — `scripts/train_baseline.py`

- Model: `BaselineResNet50` → ResNet-50 (IMAGENET1K_V1), `fc → Identity`, `Linear(2048, 8)`.
- Loss: CE with `label_smoothing=0.03`. Optimizer: AdamW lr 2e-4, wd 1e-4. Schedule: linear warmup 5 epochs → cosine to 40.
- Batch 96, workers 12, `precision="16-mixed"`, `devices=1`.
- Checkpoint: `monitor="val/acc"`, `mode=max`, `save_top_k=1`, `save_last=False`, `filename="best-{epoch:02d}"`.
- Post-fit: reloads best ckpt → fits Mahalanobis on **`datamodule.train_dataloader()` with training augmentation active** → prints MSP/Energy/Maha table → writes `summary.json` + `config.yaml`.

### 2.5 CSG-Lite training — `scripts/train_csg.py`

Architecture (`src/models/csg_lite.py`, dims confirmed from checkpoint tensors):

```
x_ctx  ──► context_backbone  (EfficientNet-B3, 1536-d) ──► context_projector  Linear(1536→64) ──► z_context (64)
                                                                                    ├─► context_predictor Linear(64→2) → d_ctx
                                                                                    └─► context_orth_projector Linear(64→16, no bias)
x_les  ──► lesion_backbone   (EfficientNet-B3, 1536-d) ──► lesion_projector   Linear(1536→16)
             (= gray3(x_ctx), derived in-model)                     └─► lesion_bn BatchNorm1d(16) ──► z_lesion (16)
                                                                                    ├─► lesion_classifier Linear(16→8) → y_hat
                                                                                    └─► GRL(α) ─► domain_classifier_adv
                                                                                          [16→32→BN→ReLU→Drop .5→16→BN→ReLU→Drop .5→2]
```

Two **fully independent** EfficientNet-B3 encoders (no weight sharing). Both ImageNet-pretrained.

Loss:
`L = L_cls + λ_ctx·L_ctx + λ_adv·L_adv + λ_orth·L_orth + λ_supcon·L_supcon`

| Term | Definition | Notes |
|---|---|---|
| `L_cls` | `F.cross_entropy(y_hat, y, ignore_index=-1)` | PAD rows carry `y=-1` and are excluded |
| `L_ctx` | CE(`d_ctx`, domain) | context head **learns** domain |
| `L_adv` | CE(`d_adv`, domain) through GRL | lesion head **unlearns** domain |
| `L_orth` | `mean(cos_sim(z_lesion, W·z_context)²)` | `W` = `context_orth_projector` |
| `L_supcon` | cross-domain supervised contrastive on L2-normalised `z_lesion`; positives = same class **and** different domain; Python `for` loop over batch rows | `λ_supcon = 0.0` in every final run |

GRL schedule (`_compute_grl_alpha`): `p = global_step/(estimated_stepping_batches−1)`, `α = max(α_min, 2/(1+exp(−γ·p^power)) − 1)` with γ=20, power=0.5, α_min=0.2.

Optimizer: single AdamW with **three parameter groups** — main @ `lr`, `domain_classifier_adv` @ `lr × 30`, `lesion_classifier` @ `lr × 0.2`. **No LR scheduler.**

Batching: paired loader emits `batch_size` ISIC + `batch_size` PAD concatenated → effective 2 × 32 = 64 images/step; `drop_last=True`; steps/epoch = `max(16211, 2298) = 16211`.

`training_step` **forces `images_lesion = None`** (line ~111) so the lesion input is always derived in-model via `_rgb_to_gray3(images_ctx)` — deliberately matching the val path. The externally-collated `images_lesion` tensor is computed by the dataloader and then discarded every step (wasted work, and see §9.2).

Post-fit: reload best ckpt → `collect_z_lesion_labels_csg` on the **paired train loader** → Mahalanobis params → AUROC/FPR95 on (ISIC test vs PAD) → `summary.json`.

### 2.6 EfficientNet-B3 single-encoder control — `cbm_revision/scripts/run_effb3_control.py`

`EffB3SingleNet`: `gray3(x)` → EfficientNet-B3 (1536) → `Linear(1536→16)` → `BatchNorm1d(16)` → `Linear(16→8)`. Plain CE, AdamW lr 1e-4 wd 1e-4, no scheduler, 40 epochs, robust transforms, batch 32. Intentionally mirrors the CSG lesion branch with the context branch and all disentanglement losses removed — the "is it just the backbone?" control.

### 2.7 Orchestration

| Runner | What it does | Seeds |
|---|---|---|
| `scripts/run_full_benchmark.sh` | Trains + probes + aggregates `baseline_soft`, `runA_grl`, `runB`; **overwrites `results_final_v1.csv`** at start | 42, 52, 62 |
| `cbm_revision/scripts/run_one_shot_cbm.sh` | Idempotent (skips if `summary.json`/`leakage.json` exists); `baseline_soft`, `runA_grl`, `runB_orth1`; then aggregate n=5 → `table_main.csv` → OOD benchmarks → stat tests → figures → summary → EffB3 control | 42, 52, 62, 72, 82 |
| `cbm_revision/scripts/run_effb3_control.sh` | EffB3 control only | 42, 52, 62, 72, 82 |

Note: `run_full_benchmark.sh` and `run_one_shot_cbm.sh` define **different** method presets under overlapping names — `run_full_benchmark.sh`'s `runB` is `λ_orth=5.0`, whereas `run_one_shot_cbm.sh` runs `runB_orth1` (`λ_orth=1.0`) and never touches `runB`.

### 2.8 Method presets (verified against per-run `config.yaml` and checkpoint `hyper_parameters`)

| Method | script | λ_ctx | λ_adv | λ_orth | λ_supcon | lr | epochs | transforms |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `baseline_soft` | train_baseline.py | – | – | – | – | 2e-4 | 40 | light |
| `runA_grl` | train_csg.py | 1.0 | 2.0 | **0.0** | 0.0 | 1e-4 | 40 | robust |
| `runB` (orth=5) | train_csg.py | 1.0 | 2.0 | **5.0** | 0.0 | 1e-4 | 40 | robust |
| `runB_orth1` | train_csg.py | 1.0 | 2.0 | **1.0** | 0.0 | 1e-4 | 40 | robust |
| `effb3_single` | run_effb3_control.py | – | – | – | – | 1e-4 | 40 | robust |

All CSG runs: `lesion_latent_dim=16`, `context_latent_dim=64`, `backbone=b3`, `adv_lr_mult=30`, `cls_lr_mult=0.2`, `grl_gamma=20`, `grl_power=0.5`, `grl_alpha_min=0.2`, `wd=1e-4`.

---

## 3. Existing trained checkpoints

70 `.ckpt` files, 16 GB, under `checkpoints/`. Sizes: CSG-Lite 259,974,997 B (dual B3) · ResNet50 baseline 282,756,535 B · EffB3 single 129,515,273 B. All written by pytorch-lightning 2.6.1.

### 3.1 Final / publication checkpoints

| Method | seed | best ckpt (path relative to `checkpoints/`) | epoch | monitored `val/acc` |
|---|---:|---|---:|---|
| baseline_soft | 42 | `baseline/baseline_soft_s42/best-20.ckpt` | 20 | 0.6782 |
| baseline_soft | 52 | `baseline/baseline_soft_s52/best-24.ckpt` | 24 | — |
| baseline_soft | 62 | `baseline/baseline_soft_s62/best-29.ckpt` | 29 | — |
| baseline_soft | 72 | `baseline/baseline_soft_s72/best-26.ckpt` | 26 | — |
| baseline_soft | 82 | `baseline/baseline_soft_s82/best-17.ckpt` | 17 | — |
| runA_grl | 42 | `csg_lite/runA_grl_s42/best-38.ckpt` (+`last.ckpt`) | 38 | — |
| runA_grl | 52 | `csg_lite/runA_grl_s52/best-37.ckpt` (+`last.ckpt`) | 37 | — |
| runA_grl | 62 | `csg_lite/runA_grl_s62/best-34.ckpt` (+`last.ckpt`) | 34 | — |
| runA_grl | 72 | `csg_lite/runA_grl_s72/best-35.ckpt` (+`last.ckpt`) | 35 | — |
| runA_grl | 82 | `csg_lite/runA_grl_s82/best-23.ckpt` (+`last.ckpt`) | 23 | — |
| runB_orth1 | 42 | `csg_lite/runB_orth1_s42/best-36.ckpt` | 36 | 0.6740 |
| runB_orth1 | 52 | `csg_lite/runB_orth1_s52/best-32.ckpt` | 32 | — |
| runB_orth1 | 62 | `csg_lite/runB_orth1_s62/best-34.ckpt` | 34 | — |
| runB_orth1 | 72 | `csg_lite/runB_orth1_s72/best-35.ckpt` | 35 | — |
| runB_orth1 | 82 | `csg_lite/runB_orth1_s82/best-37.ckpt` | 37 | — |
| runB (orth=5) | 42 | `csg_lite/runB_s42/best-39.ckpt` | 39 | — |
| runB (orth=5) | 52 | `csg_lite/runB_s52/best-28.ckpt` | 28 | — |
| runB (orth=5) | 62 | `csg_lite/runB_s62/best-39.ckpt` | 39 | — |
| effb3_single | 42 | `effb3_single/effb3_single_s42/best-39.ckpt` | 39 | 0.6894 |
| effb3_single | 52 | `effb3_single/effb3_single_s52/best-30.ckpt` | 30 | — |
| effb3_single | 62 | `effb3_single/effb3_single_s62/best-29.ckpt` | 29 | — |
| effb3_single | 72 | `effb3_single/effb3_single_s72/best-39.ckpt` | 39 | — |
| effb3_single | 82 | `effb3_single/effb3_single_s82/best-25.ckpt` | 25 | — |

**`runB` (λ_orth=5.0) exists for 3 seeds only (42/52/62).** `runB_s72` / `runB_s82` were never trained — no directories exist.

### 3.2 Stale duplicate checkpoints inside final run dirs

These directories contain checkpoints from **earlier, superseded training passes** of the same run name (`save_last=True` produced `last-vN.ckpt`; re-runs produced extra `best-*.ckpt`). They are a live hazard because several helpers resolve "newest ckpt in dir":

- `csg_lite/runB_s42/`: `best-31`, `best-37`, `best-39`, `last`, `last-v1`, `last-v2`
- `csg_lite/runB_s52/`: `best-01`, `best-28`, `best-34`, `last`, `last-v1`, `last-v2`
- `csg_lite/runB_s62/`: `best-28`, `best-39`, `last`, `last-v1`
- `csg_lite/runB_orth1_s42/`: `best-36`, `best-37`, `last`, `last-v1`

### 3.3 Legacy / exploratory loose checkpoints (not in any run directory)

`checkpoints/baseline/*.ckpt` (6 files) and `checkpoints/csg_lite/*.ckpt` (8 files), dated 2026-04-22, from pre-benchmark tuning:

| File | epoch | `val/acc` | notable hparams |
|---|---:|---:|---|
| `baseline/best-05.ckpt` | 5 | 0.1366 | failed run |
| `baseline/best-06.ckpt` | 6 | 0.1474 | failed run, lr 3e-4 |
| `baseline/best-08.ckpt` | 8 | 0.1374 | failed run |
| `baseline/best-27.ckpt` | 27 | 0.7208 | |
| `baseline/best-30.ckpt` | 30 | **0.7589** | highest val/acc of any ResNet50 here |
| `baseline/best-34.ckpt` | 34 | 0.7518 | |
| `csg_lite/best-14.ckpt` / `last-v3.ckpt` | 14 | 0.6316 | **`lesion_latent_dim=64`**, λ_supcon 0.1 |
| `csg_lite/best-18.ckpt` / `last-v2.ckpt` | 18 | 0.6149 | λ_supcon 0.2 |
| `csg_lite/best-25.ckpt` / `last.ckpt` | 25 | 0.6703 | **λ_adv = 0.0** (no adversary) |
| `csg_lite/best-31_best.ckpt` / `last-v1.ckpt` | 31 | 0.6933 | λ_adv 2, λ_orth 5, λ_supcon 0 |

`csg_lite/best-25.ckpt` (λ_adv=0) is the only trained **ablation without the adversary**; it is a genuinely reusable artifact but was trained on the un-versioned pre-benchmark configuration and its metadata source is not recorded.

### 3.4 `data/models/` — imported weights from a prior project

Not referenced by any script in this repository (verified by grep).

| File | Format | Status |
|---|---|---|
| `resnet18_supcon_stage1_15epochs_best.pth` | `OrderedDict`, 124 tensors, keys `encoder.*` | loads OK |
| `resnet18_baseline_epoch3.pth` | `OrderedDict`, 122 tensors | loads OK |
| `resnet50_robust_epoch_last.pth` | `OrderedDict`, 320 tensors | loads OK |
| `efficientnet_b3_robust_epoch_last.pth` | zip header present, **no central directory** | **CORRUPTED — `torch.load` fails with `PytorchStreamReader … failed finding central directory`. Unusable.** |

---

## 4. Existing embeddings

> **There are NO cached embeddings in this repository.**

An exhaustive filesystem sweep for `*.npy`, `*.npz`, `*.pt`, `*.pth`, `*.h5`, `*.hdf5`, `*.pkl`, `*.pickle`, `*.parquet`, `*.feather`, `*.joblib` outside `.venv/` returns **exactly 4 files**, all of which are the imported model weights in `data/models/` (§3.4). None is an embedding matrix.

A source grep for `np.save`, `np.savez`, `torch.save`, `to_pickle`, `.npy`, `.npz` across `src/`, `scripts/`, `cbm_revision/` finds **no embedding-persistence call anywhere**. The only matches are the string `"--ckpt_for_embedding"` (an argparse flag in `generate_paper_figures.py`) and the method name `extract_embedding` in `effb3_single.py`.

Consequently, per-embedding filename / dimension / source checkpoint / dataset / shape / location cannot be reported — **those artifacts do not exist**. Every representation is recomputed in-memory on every script invocation and discarded on exit.

### 4.1 Embedding spaces that *would* be produced (for Paper 3 planning)

Dimensions verified from checkpoint tensor shapes, not inferred. Row counts are the recomputed split sizes from §2.2.

| Logical name | Dim | Produced by | Source model | Extraction call |
|---|---:|---|---|---|
| `z_lesion` | **16** | `CSGLite.encode_z_lesion` / `forward(return_latents=True)` | CSG-Lite | `lesion_bn(lesion_projector(lesion_backbone(gray3(x))))` |
| `z_context` | **64** | `CSGLite.context_projector(extract_context_features(x))` | CSG-Lite | context branch, pre-domain-head |
| `backbone_raw` (context) | **1536** | `CSGLite.extract_context_features` | CSG-Lite | EfficientNet-B3 pooled |
| baseline features | **2048** | `BaselineNet.forward(return_features=True)` | ResNet50 | pre-FC pooled |
| EffB3 `z` | **16** | `EffB3SingleNet.extract_embedding` | EffB3 single | `bn(projector(backbone(gray3(x))))` |
| logits | **8** | all models | all | — |

Row counts per extraction set: ISIC train **16,211** · ISIC val 4,053 · ISIC test (ID) **5,067** · PAD (OOD) **2,298** · leakage-probe set (ID+OOD) **7,365**.

So e.g. a `z_lesion` cache for the leakage probe would be `(7365, 16)`; the Mahalanobis fit set `(16211, 16)`; the baseline probe set `(7365, 2048)`. **None of these are currently written to disk.**

Collection helpers that already exist and could be reused verbatim:
- `src/utils/ood_metrics.collect_logits_labels_features_baseline` → `(logits, labels, features)`
- `src/utils/ood_metrics.collect_z_lesion_labels_csg` → `(z_lesion, labels)`, drops `label<0` rows
- `scripts/eval_ood_scores.collect_logits_z_labels` → `(logits, z_les, z_ctx, labels)` ← most complete
- `scripts/check_leakage.collect_test_features_csg` → `(z_lesion, z_context, backbone_raw, logits, labels)` ← richest
- `cbm_revision/scripts/eval_ood_benchmarks._collect_csg_features` / `_collect_baseline_features`
- `scripts/generate_paper_figures.collect_latents_and_domains` → `(z_les, z_ctx, domains)`

Six near-duplicate implementations of the same extraction — see §9.

---

## 5. Existing experiment results

### 5.1 Frozen n=3 tables (root)

`results_final_v1.csv` — written by `scripts/run_full_benchmark.sh` / `aggregate_results.py --append_csv`:

| Method | Acc | Bal Acc | ECE | Leakage |
|---|---|---|---|---|
| Baseline Soft (n=3) | 0.7842 ± 0.0119 | 0.6621 ± 0.0091 | 0.1005 ± 0.0051 | 0.9791 ± 0.0010 |
| Run A / GRL (n=3) | 0.8099 ± 0.0074 | 0.6997 ± 0.0104 | 0.0948 ± 0.0053 | 0.7099 ± 0.0108 |
| Run B (orth=5.0) (n=3) | 0.8112 ± 0.0006 | 0.6920 ± 0.0108 | 0.1003 ± 0.0109 | 0.7241 ± 0.0127 |
| Run B (orth=1.0) (n=3) | 0.8090 ± 0.0082 | 0.7006 ± 0.0026 | 0.0917 ± 0.0022 | 0.7131 ± 0.0133 |

`RESULTS_TABLES.csv` — same 4 aggregates plus 12 per-seed rows, with `OOD_Maha_AUROC` / `OOD_Maha_FPR95` columns and a `Notes` column naming each best checkpoint. Not produced by any script in the repo (no writer found) — hand-assembled.

### 5.2 n=3 aggregate JSON/MD (`results/`)

`table1_baseline_soft_aggregate.{md,json}`, `table1_runA_grl_aggregate.{md,json}`, `table1_runB_aggregate.{md,json}`, `table1_runB_orth1_aggregate.{md,json}`, plus `table1_draft.md` and `runB_full_aggregate.{md,json}`.

`runB_full_aggregate.md` reports **different numbers for the same run names** than `table1_runB_aggregate.md` (e.g. `runB_s42`: acc 0.8163 vs 0.8115, leakage 0.7175 vs 0.7234) — it is an aggregate over an earlier training pass of runB whose checkpoints are now the stale duplicates in §3.2. It is superseded and should not be cited.

### 5.3 n=5 CBM-revision results (`results/cbm_revision/`) — **the current canonical set**

`table_main.csv`:

| Method | n | Acc | Bal Acc | ECE | Leakage |
|---|---:|---|---|---|---|
| Baseline Soft | 5 | 0.7824 ± 0.0096 | 0.6548 ± 0.0123 | 0.1014 ± 0.0045 | 0.9792 ± 0.0009 |
| Run A / GRL | 5 | 0.8073 ± 0.0079 | 0.6956 ± 0.0112 | 0.0901 ± 0.0102 | 0.7084 ± 0.0092 |
| Run B (orth=1.0) | 5 | 0.8081 ± 0.0077 | 0.7017 ± 0.0026 | 0.0942 ± 0.0035 | 0.7245 ± 0.0204 |

Also present: `baseline_soft_n5.{md,json}`, `runA_grl_n5.{md,json}`, `runB_orth1_n5.{md,json}` (per-seed + aggregate, including `maha_auroc` / `maha_fpr95`).

`ood_comparison.csv` — 4 detectors × 3 methods, mean over 5 seeds, on `z_lesion` (baseline uses its 2048-d backbone as the stand-in latent):

| Method | Detector | AUROC | AUPR_IN | AUPR_OUT | FPR95 |
|---|---|---:|---:|---:|---:|
| Baseline Soft | MSP | 0.7374 | 0.8332 | 0.5353 | 0.8686 |
| Baseline Soft | Energy | 0.7097 | 0.7991 | 0.5332 | 0.9205 |
| Baseline Soft | Cosine | 0.7162 | 0.8255 | 0.5328 | 0.8480 |
| Baseline Soft | **Mahalanobis** | **0.8679** | 0.9356 | 0.7172 | 0.4550 |
| Run A / GRL | MSP | 0.4199 | 0.6485 | 0.2674 | 0.9480 |
| Run A / GRL | Energy | 0.4287 | 0.6592 | 0.2629 | 0.9563 |
| Run A / GRL | Cosine | 0.4107 | 0.6136 | 0.2747 | 0.9874 |
| Run A / GRL | Mahalanobis | 0.4015 | 0.6093 | 0.2893 | 0.9849 |
| Run B (orth=1.0) | MSP | 0.4138 | 0.6447 | 0.2658 | 0.9521 |
| Run B (orth=1.0) | Energy | 0.4289 | 0.6588 | 0.2634 | 0.9546 |
| Run B (orth=1.0) | Cosine | 0.4131 | 0.6229 | 0.2743 | 0.9779 |
| Run B (orth=1.0) | Mahalanobis | 0.4089 | 0.6175 | 0.2899 | 0.9794 |

**Every CSG OOD AUROC is below 0.5** — the scores are systematically inverted (PAD sits *closer* to the ISIC class-conditional Gaussians than held-out ISIC does). This is reproduced independently by `eval_ood_benchmarks.py`, which fits Mahalanobis on ISIC-train under the **correct eval transform**, so it is not an artifact of the fitting-transform bug in §9.2 — but it is fully consistent with the CSG model having trained on all 2,298 PAD images (§9.1). This result cannot currently support an OOD-detection claim for CSG.

`stat_tests.csv` (n=5, Welch + Wilcoxon + Cohen's d, both vs Baseline):

| Metric | Comparison | Welch p | Cohen's d | Wilcoxon p |
|---|---|---|---:|---|
| Balanced Accuracy | Run A / GRL vs Baseline | 0.00122 | +3.10 | 0.0625 |
| Balanced Accuracy | Run B (orth=1.0) vs Baseline | 0.00123 | +4.71 | 0.0625 |
| Leakage | Run A / GRL vs Baseline | 4.11e-07 | **−37.08** | 0.0625 |
| Leakage | Run B (orth=1.0) vs Baseline | 1.49e-05 | **−15.76** | 0.0625 |

All Wilcoxon p = 0.0625 — that is the **floor** for n=5 (2⁻⁴), i.e. the test cannot reach 0.05 at this sample size. `FINAL_SUMMARY.md` correctly flags this and steers toward Cohen's d. Note also that Wilcoxon *signed-rank* is being applied to two independent method groups paired only by seed index; the pairing is defensible (shared seed) but should be stated.

`FINAL_SUMMARY.md` conclusions: best detector = Mahalanobis; best method = Run B (orth=1.0); n=3→n=5 conclusions hold.

Figures in `results/cbm_revision/figures/`: `utility_leakage_tradeoff_cbm.{png,pdf}`, `seed_stability_cbm.{png,pdf}`, `ood_detector_comparison.{png,pdf}`.
`results/cbm_revision/per_seed/` is **empty** (created by the runner, never populated).

### 5.4 EffNet-B3 backbone control (`results/effb3_control/`) — n=5

`aggregate_stats.csv` (the single most useful comparison table in the repo):

| Method | n | Accuracy | Balanced Acc | ECE | Leakage | Maha OOD AUROC |
|---|---:|---|---|---|---|---|
| ResNet50 Baseline | 5 | 0.7824 ± 0.0096 | 0.6548 ± 0.0123 | 0.1014 ± 0.0045 | 0.9792 ± 0.0009 | 0.8851 ± 0.0210 |
| EffNet-B3 Single Encoder | 5 | 0.8037 ± 0.0092 | 0.6830 ± 0.0126 | 0.0956 ± 0.0083 | 0.8017 ± 0.0201 | 0.7259 ± 0.0213 |
| Run A / GRL | 5 | 0.8073 ± 0.0079 | 0.6956 ± 0.0112 | 0.0901 ± 0.0102 | 0.7084 ± 0.0092 | 0.4013 ± 0.0248 |
| Run B (orth=1.0) | 5 | 0.8081 ± 0.0077 | 0.7017 ± 0.0026 | 0.0942 ± 0.0035 | 0.7245 ± 0.0204 | 0.4084 ± 0.0193 |

This is the cleanest existing evidence that the leakage reduction is not purely a backbone effect: EffB3-single reaches leakage 0.8017 while CSG reaches 0.7084–0.7245.

Also present: `metrics.csv` (per-seed acc/bacc/ece), `leakage_probe.csv` (per-seed), `ood_scores.csv` (5 seeds × 4 detectors), `significance_effb3.csv` (EffB3 vs each of 3 methods × 4 metrics), `comparison_table.csv`, `summary.txt`, `concise_results.md`, `backbone_control_tradeoff.png`, and `per_seed/effb3_single_s{42,52,62,72,82}/summary.json`.

One stray file: `leakage_seed42_best39.json` — a standalone 5-seed probe of `effb3_single_s42/best-39.ckpt` (mean 0.8157). Not written by any script found in the repo.

`results/effb3_control_seed42_check/` and its `per_seed/effb3_single_s{42,52}/` subdirectories exist but are **completely empty**.

### 5.5 Per-run artifacts and training logs

18 run directories each containing `config.yaml`, `summary.json`, `leakage.json`, `csv_logs/{hparams.yaml,metrics.csv}`:

- `results/baseline/baseline_soft_s{42,52,62,72,82}` — metrics.csv 417 lines (41 epochs); columns include per-class `val/acc_{MEL,NV,…}`
- `results/csg_lite/runA_grl_s{42,52,62,72,82}` — metrics.csv 1093 lines
- `results/csg_lite/runB_orth1_s{42,52,62,72,82}` — metrics.csv 1093 lines
- `results/csg_lite/runB_s{42,52,62}` — metrics.csv 1093 lines (**no s72/s82**)

CSG `metrics.csv` columns: `epoch, step, train/acc_adv, train/acc_cls, train/acc_ctx, train/grl_a, train/grl_progress, train/loss_adv, train/loss_cls, train/loss_ctx, train/loss_epoch, train/loss_orth, train/loss_step, train/loss_supcon, val/acc, val/loss` — a complete per-epoch optimization trace for all 13 CSG runs. Directly reusable.

`lightning_logs/` — 34 `version_*` dirs. `version_1` and `version_5` are **empty**. `version_0`–`version_28` are pre-benchmark exploratory runs (dated 2026-04-22, before named run dirs existed). `version_29`–`version_33` (dated 2026-04-24, `hparams: latent_dim 16, lr 1e-4, wd 1e-4`) are the **5 EffB3 control runs** — the EffB3 script never sets a `CSVLogger` save_dir, so its logs went to the default location instead of a run directory.

Full per-run measurements (recomputed from `summary.json` + `leakage.json` on disk):

| run | seed | id_acc¹ | bal_acc¹ | ece¹ | maha_auroc | fpr95 | z_lesion probe | z_context probe | backbone_raw probe | shuffled-label probe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_soft_s42 | 42 | 0.8030 | 0.6994 | 0.0829 | 0.8797 | 0.4409 | 0.9795 | 0.9795 | 0.9795 | 0.5842 |
| baseline_soft_s52 | 52 | 0.8107 | 0.6947 | 0.0855 | 0.8981 | 0.3807 | 0.9778 | 0.9778 | 0.9778 | 0.5896 |
| baseline_soft_s62 | 62 | 0.8133 | 0.6781 | 0.0889 | 0.8516 | 0.5293 | 0.9801 | 0.9801 | 0.9801 | 0.5855 |
| baseline_soft_s72 | 72 | 0.8040 | 0.6823 | 0.0933 | 0.8815 | 0.4608 | 0.9800 | 0.9800 | 0.9800 | 0.5896 |
| baseline_soft_s82 | 82 | 0.8042 | 0.6785 | 0.0865 | 0.9148 | 0.3345 | 0.9786 | 0.9786 | 0.9786 | 0.5679 |
| runA_grl_s42 | 42 | 0.8097 | 0.7041 | 0.0965 | 0.3700 | 0.9917 | 0.6959 | 0.9997 | 0.9995 | 0.6878 |
| runA_grl_s52 | 52 | 0.8190 | 0.6854 | 0.0878 | 0.4201 | 0.9905 | 0.7113 | 0.9997 | 0.9995 | 0.6878 |
| runA_grl_s62 | 62 | 0.8009 | 0.7095 | 0.1004 | 0.3849 | 0.9749 | 0.7223 | 0.9995 | 0.9998 | 0.6878 |
| runA_grl_s72 | 72 | 0.8099 | 0.6989 | 0.0946 | 0.3927 | 0.9921 | 0.7116 | 0.9996 | 0.9991 | 0.6878 |
| runA_grl_s82 | 82 | 0.7963 | 0.6800 | 0.0716 | 0.4387 | 0.9757 | 0.7010 | 0.9995 | 0.9995 | 0.6878 |
| runB_orth1_s42 | 42 | 0.8208 | 0.7044 | 0.0887 | 0.3814 | 0.9976 | 0.6998 | 0.9997 | 0.9994 | 0.6878 |
| runB_orth1_s52 | 52 | 0.8021 | 0.6995 | 0.0933 | 0.4030 | 0.9856 | 0.7081 | 0.9997 | 0.9998 | 0.6878 |
| runB_orth1_s62 | 62 | 0.8044 | 0.6981 | 0.0936 | 0.4405 | 0.9728 | 0.7312 | 0.9994 | 0.9995 | 0.6878 |
| runB_orth1_s72 | 72 | 0.8131 | 0.7050 | 0.0981 | 0.4147 | 0.9696 | 0.7246 | 0.9998 | 0.9995 | 0.6878 |
| runB_orth1_s82 | 82 | 0.8003 | 0.7016 | 0.0978 | 0.4023 | 0.9720 | 0.7586 | 0.9989 | 0.9993 | 0.6878 |
| runB_s42 | 42 | 0.8115 | 0.6967 | 0.1128 | 0.4303 | 0.9432 | 0.7234 | 0.9992 | 0.9992 | 0.6878 |
| runB_s52 | 52 | 0.8117 | 0.6771 | 0.0862 | 0.4152 | 0.9842 | 0.7089 | 1.0000 | 0.9997 | 0.6878 |
| runB_s62 | 62 | 0.8103 | 0.7023 | 0.1018 | 0.3881 | 0.9793 | 0.7400 | 0.9997 | 0.9997 | 0.6878 |

¹ from `summary.json`. **`leakage.json` carries a second, different `id_acc`/`bal_acc`/`ece` for the same checkpoint, and `aggregate_results.py` silently prefers the leakage one** — see §9.3. All published tables therefore use the leakage-side values.

Probe sanity: majority-class baseline = 5067/7365 = **0.6880**; the shuffled-domain-label control lands at exactly 0.6878 for all CSG runs (correct — reduces to majority prediction). Baseline runs show 0.568–0.590, *below* majority, on a 2048-d probe — mild overfitting of the shuffle control, worth noting but not disqualifying.

### 5.6 Reader-study CSVs — orphaned imports

`data/reader_study/reader_cases_effb3.csv` (74 cases) and `reader_cases_selected_96.csv` (96 cases). Columns include `dstskin_maha`, `reliability_flag`, `behavior_group`, `danger_score`; `image_path` values point at `data/raw/isic2018/...` and `data/raw/pad_ufes20/...`. **This repository uses ISIC 2019, not 2018, and `data/raw/` is empty.** No script here reads or writes them. They are imports from a different project (`Ban_sao_datn` / "DST-Skin") and their referenced images do not resolve inside this repo.

---

## 6. Existing analysis scripts

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `src/datasets/preprocess_metadata.py` | build master metadata | ISIC GT + PAD metadata | `data/master_metadata.csv` |
| `scripts/build_lesion_only_metadata.py` | Otsu-based lesion crop | master_metadata | cropped JPEGs + `..._lesion_only_soft.csv` |
| `scripts/verify_data_layout.py` | path/layout check | data/ | **stdout only** |
| `scripts/verify_isic_labels.py` | master vs GroundTruth agreement | master + GT | **stdout only** (exit 3 on mismatch) |
| `scripts/train_baseline.py` | ResNet50 + OOD table | metadata | ckpt, `summary.json`, `config.yaml`, csv_logs |
| `scripts/train_csg.py` | CSG-Lite + Maha OOD | metadata | ckpt, `summary.json`, `config.yaml`, csv_logs |
| `scripts/check_leakage.py` | **domain linear probe** | ckpt + metadata | `leakage.json` |
| `scripts/eval_ood_scores.py` | Energy@T / MSP / Cosine / Maha on z_les & z_ctx | ckpts | **stdout only — no artifact** |
| `scripts/aggregate_results.py` | merge summary+leakage → mean±std | run JSONs | `*_aggregate.{md,json}`, optional CSV append |
| `scripts/generate_paper_figures.py` | t-SNE, trade-off, leakage bar, stability | ckpt + aggregate JSONs + csv_logs | 5 figure pairs |
| `scripts/plot_confusion_matrix_runb.py` | 8×8 row-normalised CM for Run B | ckpt | `fig_confusion_matrix.{png,pdf}` |
| `scripts/plot_reliability_diagram.py` | 15-bin reliability, baseline vs Run B | 2 ckpts | `fig_reliability_baseline_vs_runB.{png,pdf}` |
| `scripts/visualize_soft_cropping_examples.py` | original / mask / crop triptych | master_metadata | `fig_soft_crop_examples.{png,pdf}` |
| `scripts/generate_workflow_framework.py` | hand-drawn Figure 1 (matplotlib patches) | — | `figure1_framework.{png,pdf}` |
| `cbm_revision/scripts/eval_ood_benchmarks.py` | 4 detectors × AUROC/AUPR_IN/AUPR_OUT/FPR95 × 5 seeds | ckpt dirs | `ood_comparison.{csv,json}` |
| `cbm_revision/scripts/stat_tests_cbm.py` | Welch + Wilcoxon + Cohen's d | n5 JSONs | `stat_tests.{csv,json}` |
| `cbm_revision/scripts/plot_cbm_figures.py` | 3 revision figures | n5 JSONs + ood csv | `results/cbm_revision/figures/*` |
| `cbm_revision/scripts/write_final_summary.py` | narrative summary, n=3 vs n=5 | n3 + n5 JSONs | `FINAL_SUMMARY.md` |
| `cbm_revision/scripts/run_effb3_control.py` | train + evaluate EffB3 control | metadata / ckpts | 5 CSVs + per-seed JSONs + `summary.txt` |
| `cbm_revision/scripts/make_effb3_publication_files.py` | publication tables/figure for control | control CSVs + n5 JSONs | `aggregate_stats.csv`, `significance_effb3.csv`, `backbone_control_tradeoff.png`, `concise_results.md` |

### 6.1 Leakage probe — exact specification (`scripts/check_leakage.py`)

This is the paper's central metric, so stated precisely:

1. Load checkpoint; `--model_type auto` infers `csg` (keys `model.lesion_backbone.*`/`model.context_backbone.*`) vs `baseline` (`net.backbone.*`), defaulting to `csg`.
2. Build `SkinDataModule` (**always with default `use_robust_transforms=True`**) and take `test_dataloader()` = ISIC test + PAD, `shuffle=False`, n=7,365.
3. Domain targets from the dataframe column: `isic→0`, `pad_ufes→1`.
4. Extract, in loader order: `z_lesion` (16), `z_context` (64), `backbone_raw` (1536). For a baseline checkpoint all three slots are filled with the same 2048-d backbone features (hence the three identical baseline probe columns in §5.5).
5. For each probe seed (`--probe_seeds`, default `42,52,62`; runners use `42,52,62` for n=3 and `42,52,62,72,82` for n=5): stratified `train_test_split(test_size=0.3)` → `StandardScaler` → `LogisticRegression(max_iter=2000)` → held-out accuracy. Report mean ± std.
6. Sanity control: same probe on `z_lesion` with domain labels shuffled (`--shuffle_test_seed 123`).
7. Also recomputes ID accuracy / balanced accuracy / ECE (15 bins) on the ISIC test loader.
8. Emits `leakage.json`.

### 6.2 Mahalanobis — exact specification (`src/utils/ood_metrics.py`)

- `compute_mahalanobis_params_from_arrays(features, labels, num_classes=8, reg_eps)`: per-class means μ_c; **shared pooled within-class covariance** Σ = Σᵢ(xᵢ−μ_{yᵢ})(xᵢ−μ_{yᵢ})ᵀ / (N−K); Σ += reg_eps·I; precision = `np.linalg.inv(Σ)`. Raises if any class is empty.
- `mahalanobis_min_squared_distances`: min over classes of δᵀPδ. **Pure Python double loop over samples × classes** — O(N·K) Python iterations, float64. On the 16,211-row fit set this is the dominant cost of every evaluation script.
- Score convention: larger min-d² = more OOD. AUROC computed with `y=1` for OOD.
- `reg_eps` differs by caller: **1e-5** in `train_baseline.py`/`train_csg.py` (the library default), **1e-3** in `eval_ood_scores.py`, `eval_ood_benchmarks.py`, and `run_effb3_control.py`.
- The fitting set also differs by caller: `train_csg.py` fits on the **augmented paired train loader**; `eval_ood_benchmarks.py` and `eval_ood_scores.py` fit on ISIC-train under the **eval transform** (the correct, "cleaned" version). The n=5 `maha_auroc` numbers in `*_n5.json` come from the training scripts (reg 1e-5, augmented); `ood_comparison.csv` comes from the cleaned path (reg 1e-3). **They are not the same estimator and should not be presented in the same table without a note.**

---

## 7. Reusable artifacts for Paper 3

**High confidence — use directly:**

1. **23 final checkpoints** across 5 methods (§3.1): baseline_soft ×5, runA_grl ×5, runB_orth1 ×5, runB ×3, effb3_single ×5. All load cleanly under the installed torch/PL; latent dims verified from tensor shapes.
2. **Deterministic split** — `random_state=42`, independent of `--seed`, identically implemented in `SkinDataModule.setup` and `splits.build_id_ood_test_dataloaders`. Any new embedding extraction will align row-for-row with existing `leakage.json` results (`shuffle=False` throughout).
3. **`data/master_metadata_lesion_only_soft.csv`** — the single metadata file behind every final experiment; all 27,629 image paths verified present.
4. **Five extraction helpers** (§4.1) covering every latent space; `check_leakage.collect_test_features_csg` is the richest and returns all four representations plus logits in loader order.
5. **`src/utils/ood_metrics.py`** — Mahalanobis fit/score and FPR@95 are correct and self-contained.
6. **13 CSG `csv_logs/metrics.csv` traces** (1093 lines each) with `loss_adv`, `loss_orth`, `loss_ctx`, `acc_adv`, `acc_ctx`, `grl_a` per epoch — a ready-made optimization-dynamics dataset requiring no GPU.
7. **The n=5 CBM tables** (`table_main.csv`, `*_n5.json`, `ood_comparison.csv`, `stat_tests.csv`, `aggregate_stats.csv`) as the current baseline numbers to extend or contest.
8. **The EffB3 backbone control** — a fully-executed, 5-seed ablation isolating backbone capacity from disentanglement. Its design (`effb3_single.py` mirrors the CSG lesion branch exactly) is the strongest control already in place.
9. **`checkpoints/csg_lite/best-25.ckpt`** — the only trained λ_adv=0 model (no adversary), usable as a qualitative ablation point, with the caveat in §3.3.
10. **Figure-generation scripts** — style is already consistent (`configure_plot_style`, PNG@300dpi + PDF pairs).

**Reusable with modification:**
- `aggregate_results.py` — generic mean±std aggregator; needs the merge-order fix (§9.3).
- `stat_tests_cbm.py` / `make_effb3_publication_files.py` — Cohen's d + Welch scaffolding, straightforward to extend to more methods.

---

## 8. Missing artifacts

Explicitly verified absent:

1. **All cached embeddings.** No `.npy`/`.npz`/`.pt` feature file exists anywhere; no code path writes one. Every latent is recomputed and thrown away. (§4)
2. **`runB` (λ_orth=5.0) seeds 72 and 82.** Only 3 of 5 seeds exist; it is the only method not brought to n=5.
3. **A persisted output for `scripts/eval_ood_scores.py`.** It prints Energy@T=1/10/100, MSP, Cosine and Mahalanobis on both `z_lesion` and `z_context` — the only place `z_context` is scored for OOD — and saves nothing. **No record of those numbers exists on disk.**
4. **`results/cbm_revision/per_seed/`** — directory exists, empty.
5. **`results/effb3_control_seed42_check/`** and its two per-seed subdirectories — exist, completely empty.
6. **`src/models/csg_full.py`** — a 2-line stub. The "full CSG" variant referenced in `configs/csg.yaml` (`variant: csg_lite # or csg_full`) **does not exist**.
7. **`IndependenceLoss`** (cross-covariance, `src/losses/csg_losses.py`) is implemented but **never instantiated by any script** — the independence objective was replaced by the cosine-orthogonality term written inline in `csg_lightning.py`. No trained model uses it.
8. **`CSGTrainDataset`** (`src/datasets/csg_train_dataset.py`) and `splits.build_csg_train_dataframe` — implemented, imported nowhere. Dead alternative to `CombinedTrainDataset`.
9. **Stub modules with no implementation:** `src/models/backbone.py`, `src/datasets/isic.py`, `src/datasets/pad_ufes.py`, `src/losses/indep_loss.py`, `src/losses/adv_loss.py`, `src/utils/metrics.py`, `src/utils/logging_utils.py`, `src/utils/visualization.py`.
10. **UMAP.** `results/figures/figure_umap_tsne_disentanglement.{png,pdf}` is named for UMAP, but `generate_paper_figures.py` computes **t-SNE only** and `umap-learn` is **not installed**. The figure contains no UMAP.
11. **TensorBoard logs.** Both training scripts add a `TensorBoardLogger` conditionally on `importlib.util.find_spec("tensorboard")`; tensorboard is **not installed**, so no `tb_logs/` directory was ever created.
12. **A working `efficientnet_b3_robust_epoch_last.pth`** — the file is corrupted (§3.4).
13. **`data/raw/` and `data/processed/`** — empty; the reader-study CSVs reference paths under `data/raw/isic2018/` that do not exist here.
14. **Any README, docs, notebook, or git history.**
15. **Any per-class / per-domain breakdown of the leakage probe.** Leakage is only ever a single scalar over the pooled ISIC-test+PAD set.
16. **Any subgroup metadata join.** `ISIC_2019_Training_Metadata.csv` has `age_approx`, `anatom_site_general`, `sex`, `lesion_id` — none of it is merged into `master_metadata.csv` (which is only `path,label,domain`). No fairness/subgroup analysis is possible without rebuilding the metadata.
17. **A writer for `RESULTS_TABLES.csv` and `results/effb3_control/leakage_seed42_best39.json`** — both exist on disk with no producing script.

---

## 9. Potential technical debt

Ordered by impact on Paper 3.

### 9.1 The OOD test set is 100 % contained in the CSG training set — blocking

`SkinDataModule.setup` passes **all 2,298 PAD rows** (`self.pad_df_all`) into `CombinedTrainDataset` as the auxiliary domain stream, and `splits.build_id_ood_test_dataloaders` returns **the same 2,298 PAD rows** as the OOD test set. Every CSG model has seen every OOD test image ~16,211/2,298 ≈ 7 times per epoch × 40 epochs.

The ResNet50 baseline never sees PAD during training. So the headline OOD comparison (baseline Maha 0.8851 vs CSG 0.4084) contrasts a model evaluated on unseen data against models evaluated on memorised data. This alone explains the sub-0.5 AUROCs and makes the current OOD numbers unusable as an OOD-detection claim for CSG. A held-out PAD partition (train-aux vs OOD-test) is required before any OOD result can be reported.

`skin_dataset.py:414` also builds `ood_test = pd.concat([isic_test, pad_df])` — the same PAD rows again — which is what the leakage probe consumes. The probe is unaffected (it measures memorised domain identity, which is the point), but the overlap should be stated in the paper.

### 9.2 Mahalanobis is fitted through a different transform than it is scored on

In `train_csg.py` the fit set is `datamodule.train_dataloader()` — the **paired CSG loader with training augmentation active** (RandomResizedCrop, flips, rotation, ColorJitter, RandomErasing). `collect_z_lesion_labels_csg` sees `len(batch)==5`, takes `images_lesion` (the dataloader's grayscale-**before**-normalisation tensor), and passes it to `encode_z_lesion`, which applies `_rgb_to_gray3` **again** — this time mixing three already-differently-normalised channels.

Meanwhile the ID/OOD scoring loaders emit `len(batch)==2` RGB tensors under the deterministic eval transform, grayscaled **after** normalisation — exactly one gray conversion.

`csg_lightning.py:106-111` contains a comment stating this pre- vs post-normalisation mismatch was deliberately fixed for train/val by forcing `images_lesion = None`. **The same fix was never applied to the Mahalanobis collection path.** So μ_c and Σ are estimated in a feature distribution the scores never live in, plus augmentation noise.

`eval_ood_scores.py` and `eval_ood_benchmarks.py` fit correctly (ISIC-train under `eval_transform`), which is why their numbers are trustworthy — and why `*_n5.json`'s `maha_auroc` and `ood_comparison.csv`'s Mahalanobis AUROC are **different estimators presented under the same name** (0.4084 vs 0.4089 for runB_orth1 — coincidentally close, but not the same quantity).

### 9.3 `aggregate_results.py` silently overwrites summary metrics with leakage metrics

```python
merged.update(s)   # summary.json
merged.update(l)   # leakage.json  ← wins for id_acc / id_balanced_acc / id_ece
```

Both files contain `id_acc`, `id_balanced_acc`, `id_ece` for the same checkpoint, and they disagree — for `baseline_soft_s42`, 0.8030 vs 0.7673 (a 3.6-point gap).

Root cause: `check_leakage.py` constructs `SkinDataModule` **without forwarding `use_robust_transforms`**, so it always evaluates under `Resize(256)+CenterCrop(224)`. The `baseline_soft` runs were *trained and originally evaluated* under the light `Resize((224,224))` path. The probe therefore re-evaluates them under a transform they never saw. Every published baseline accuracy in `results_final_v1.csv`, `table1_*`, and `table_main.csv` is the mismatched-transform value. CSG runs are unaffected (they use robust transforms in both places), so the effect is a systematic downward bias on the baseline only — i.e. it inflates the reported CSG-vs-baseline utility gap.

### 9.4 Ambiguous "newest checkpoint in directory" resolution has already selected wrong checkpoints

`ood_metrics.find_checkpoint` picks `max(mtime)` over `*.ckpt` in a directory. `run_effb3_control._find_ckpt` and `eval_ood_benchmarks._find_ckpt` prefer newest `best-*` then newest `last*`. With the stale duplicates of §3.2 present, this is unstable — and demonstrably fired: comparing `summary.json.best_checkpoint` against `leakage.json.checkpoint`:

| run | trained best ckpt | ckpt actually probed |
|---|---|---|
| runB_s42 | `best-39.ckpt` | **`last-v2.ckpt`** |
| runB_s52 | `best-28.ckpt` | **`last-v2.ckpt`** |
| runB_s62 | `best-39.ckpt` | **`last-v1.ckpt`** |
| runB_orth1_s42 | `best-36.ckpt` | **`last-v1.ckpt`** |
| runB_orth1_s52 | `best-32.ckpt` | **`last.ckpt`** |
| runB_orth1_s62 | `best-34.ckpt` | **`last.ckpt`** |

Six of eighteen runs — including 3 of the 5 `runB_orth1` seeds that carry the paper's headline conclusion — have their leakage measured on the **last-epoch** checkpoint while their utility comes from the **best-val** checkpoint. The two halves of every published `runB_orth1` row are, for those seeds, from different models. `run_full_benchmark.sh` and `run_one_shot_cbm.sh` both pass an explicit `find_best_ckpt` path, so these came from manual invocations passing a directory.

### 9.5 Six near-duplicate feature-extraction implementations

`collect_logits_labels_features_baseline`, `collect_z_lesion_labels_csg` (ood_metrics), `collect_test_features_csg`, `collect_test_features_baseline` (check_leakage), `collect_logits_z_labels` (eval_ood_scores), `_collect_csg_features`, `_collect_baseline_features` (eval_ood_benchmarks), `_collect_logits_features` (run_effb3_control), `collect_latents_and_domains` (generate_paper_figures). They differ in which latents they return, which batch arities they handle, and whether they mask `label<0`. §9.2 exists precisely because of this duplication.

Similarly, `_compute_ece` is copy-pasted verbatim into 4 files, `_find_ckpt` into 3, and the ISIC-train-loader reconstruction into 3.

### 9.6 Metric conventions inconsistent across scripts

- `reg_eps`: 1e-5 in training scripts vs 1e-3 in evaluation scripts.
- Trade-off figures: `generate_paper_figures.plot_tradeoff` plots **accuracy** vs leakage; `make_effb3_publication_files.plot_tradeoff` plots **balanced accuracy** vs leakage. Both are captioned "utility-leakage trade-off".
- `plot_tradeoff` in `generate_paper_figures.py` hardcodes axis limits and per-method text offsets and silently `return`s if any of the 4 aggregate JSONs is absent — a figure can quietly fail to regenerate.
- `aggregate_results.py --append_csv` **appends**, while `run_full_benchmark.sh` truncates `results_final_v1.csv` at start. Running the aggregator standalone twice silently duplicates rows.

### 9.7 Statistical reporting at n=5

Wilcoxon signed-rank cannot produce p < 0.0625 with n=5; all 8 rows in `stat_tests.csv` report exactly that value and are marked `significant = no`, next to Welch p-values as low as 4e-07 marked `yes`. Cohen's d values of −37 and −16 are arithmetically correct but reflect near-zero within-method variance (leakage std 0.0009 for baseline), not a meaningful standardised effect. Reporting these as-is invites reviewer objections.

### 9.8 Performance

`mahalanobis_min_squared_distances` is a Python `for` loop over samples × classes in float64. For the 16,211-row fit + 5,067 ID + 2,298 OOD sets, × 8 classes, × 5 seeds, × 3 methods, this dominates `eval_ood_benchmarks.py` runtime. A vectorised form is a one-line change with identical numerics.

`_supervised_contrastive_cross_domain_loss` also loops over all 64 batch rows in Python per step — but `λ_supcon = 0.0` in every final run, so it computes a loss that is then multiplied by zero, ~16,211 times per epoch.

The CSG dataloader computes `images_lesion` for all 64 images every step, which `training_step` immediately discards.

### 9.9 Configuration and dependency drift

- `configs/baseline.yaml` and `configs/csg.yaml` are **never loaded** — no `yaml.load` / `OmegaConf` anywhere. They advertise `lr 1e-4`, `wd 1e-2`, `max_epochs 50`, `batch_size 32`, `precision 32`, `metadata_csv: data/master_metadata.csv`, and `variant: csg_full` — **none of which matches any executed run**. They are actively misleading.
- `requirements.txt` lists `opencv-python` (never imported) and omits `torchmetrics`, `scipy`, `seaborn` (all imported and required). No version pins at all, against a stack that includes numpy 2.4 / pandas 3.0 / torch 2.11.
- `plot_reliability_diagram.py` defaults `--baseline_ckpt` to `checkpoints/baseline_soft`, **which does not exist** (the real path is `checkpoints/baseline/baseline_soft_s42`). It survives only via a keyword-based recursive fallback scan that picks the newest matching ckpt anywhere under `checkpoints/` — non-deterministic given §3.2/§3.3.
- `generate_paper_figures.py` and `eval_ood_scores.py` default to `checkpoints/csg_lite/runB_s42/best-31.ckpt`, which is a **stale duplicate**, not that run's best (`best-39.ckpt`). The published t-SNE figure was generated from it unless overridden.
- `checkpoints/` and `lightning_logs/` are in `.gitignore` but the repo is not under git, so 16 GB of checkpoints have no provenance tracking at all.
- Two metadata CSVs (`..._lesion_only.csv`, `..._lesion_only_soft.csv`) are byte-identical because the `aggressive` and `soft` presets share `--out_dir`; the aggressive crops were overwritten. No aggressive-preset data survives.

---

## 10. Summary for Paper 3 planning

**What is solid:** the split logic, the leakage-probe design (with shuffle control and majority baseline recorded), 23 loadable checkpoints across 5 methods at n=5 (n=3 for `runB` orth=5), complete per-epoch optimization traces for all 13 CSG runs, and a properly-executed backbone control that isolates capacity from disentanglement.

**What must be resolved before reuse:**
1. PAD-UFES is simultaneously the CSG auxiliary training set and the entire OOD test set (§9.1) — all CSG OOD numbers are invalid as stated.
2. Six of eighteen leakage measurements were taken on a different checkpoint than the matching utility measurement (§9.4).
3. Baseline utility metrics are systematically depressed by an eval-transform mismatch that only affects the baseline (§9.3).
4. Two different Mahalanobis estimators are reported under one name (§9.2, §6.2).

**What does not exist and would have to be built:** any embedding cache, a held-out PAD partition, `csg_full`, subgroup metadata, and a persisted record of the `z_context` OOD scores.
