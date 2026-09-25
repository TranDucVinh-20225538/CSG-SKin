# CSG-Skin — Repository Guide

**Scope of this document:** a from-scratch audit of exactly what is present in this git working copy, built by reading every source file in the repository (not by reusing conclusions from `docs/repository_audit.md`, though that file is cross-referenced where useful — see "Relationship to the prior audit" below). No code was modified to produce this guide.

**What this checkout actually contains:** the source code (`src/`, `scripts/`, `cbm_revision/scripts/`, `configs/`, `requirements.txt`, two root CSVs, an empty `notebooks/`), plus, as of this revision, a `checkpoints/` directory. `checkpoints` is a symlink (`checkpoints -> /Users/cubo/ResearchArtifacts/checkpoints`, untracked by git) pointing at 16 GB / 67 `.ckpt` files across `baseline/`, `csg_lite/`, and `effb3_single/`. **`data/`, `results/`, and `lightning_logs/` are still absent** — all three remain listed in `.gitignore` and were never populated in this checkout, so dataset statistics, per-run `summary.json`/`leakage.json`/`config.yaml`/CSV-log contents, and figures still cannot be verified here. No Python environment with `torch` is available in this session either, so checkpoint *contents* (state-dict keys, `hyper_parameters`, tensor shapes) could not be inspected directly — everything below about the checkpoints is derived from **filesystem-level facts** (paths, file sizes, counts, modification times) verified directly against `/Users/cubo/ResearchArtifacts/checkpoints`, cross-checked against the architecture as read from `src/models/*.py`. Every claim that would require `data/`/`results/`/checkpoint-internals is explicitly marked **[not present in this checkout]**.

---

## 1. Repository architecture

```
CSG-Skin/
├── requirements.txt              # 7 lines, incomplete relative to actual imports (see §9)
├── results_final_v1.csv          # frozen n=3 headline table (4 methods) — present, hand/script-produced
├── RESULTS_TABLES.csv            # n=3 aggregate + per-seed table, richer columns — present
├── configs/                      # baseline.yaml, csg.yaml — not loaded by any script (see §9)
├── notebooks/                    # .gitkeep only, no notebooks
├── docs/
│   └── repository_audit.md       # prior audit written on a machine with data/checkpoints/results present
│
├── src/                                  # importable package (no setup.py; used via PYTHONPATH-style sys.path insert)
│   ├── __init__.py
│   ├── datasets/
│   │   ├── constants.py                  # LABELS, LABEL_TO_INDEX, ImageNet mean/std
│   │   ├── preprocess_metadata.py        # builds data/master_metadata.csv from ISIC+PAD source CSVs
│   │   ├── skin_dataset.py               # SkinDataset, CombinedTrainDataset, SkinDataModule (the core datamodule)
│   │   ├── splits.py                     # load_filtered_master, build_id_ood_test_dataloaders (split logic, duplicated)
│   │   ├── csg_train_dataset.py          # CSGTrainDataset — dead code, imported nowhere
│   │   ├── isic.py, pad_ufes.py          # 1-line pointer-comment stubs
│   │   └── __init__.py                   # 1-line stub
│   ├── models/
│   │   ├── baseline.py                   # BaselineNet (ResNet50) + BaselineResNet50 LightningModule
│   │   ├── csg_lite.py                   # CSGLite dual-encoder nn.Module + GradientReversalLayer
│   │   ├── csg_lightning.py              # CSGLiteLightning — 5-term loss, GRL schedule, SupCon
│   │   ├── effb3_single.py               # EffB3SingleNet / EffB3SingleLightning (single-encoder control)
│   │   ├── backbone.py, csg_full.py      # 1-line stubs; csg_full is NOT implemented
│   │   └── __init__.py                   # 1-line stub
│   ├── losses/
│   │   ├── csg_losses.py                 # ClassificationLoss, ContextLoss, AdvLoss, IndependenceLoss
│   │   ├── indep_loss.py, adv_loss.py     # 1-line pointer stubs (point back to csg_losses.py)
│   │   └── __init__.py                   # 1-line stub
│   └── utils/
│       ├── ood_metrics.py                # Mahalanobis fit/score, FPR@95, feature-collection helpers, find_checkpoint
│       ├── paths.py                      # PROJECT_ROOT / DATA_ROOT (derived from this file's location)
│       ├── seed.py                       # seed_everything()
│       ├── metrics.py, logging_utils.py, visualization.py  # 1-line stubs, unimplemented
│       └── __init__.py                   # 1-line stub
│
├── scripts/                       # main pipeline entry points (all standalone, sys.path-inject ROOT)
│   ├── train_baseline.py          # train ResNet50 baseline + OOD table + summary.json
│   ├── train_csg.py               # train CSG-Lite + Mahalanobis OOD + summary.json
│   ├── check_leakage.py           # domain linear-probe on trained checkpoint → leakage.json
│   ├── eval_ood_scores.py         # Energy/MSP/Cosine/Mahalanobis on existing ckpts — stdout only, no artifact
│   ├── aggregate_results.py       # merge summary.json + leakage.json across seeds → mean±std table
│   ├── build_lesion_only_metadata.py  # Otsu-based lesion crop → lesion-only metadata CSV + cropped JPEGs
│   ├── verify_data_layout.py      # sanity-check expected data/ paths exist — stdout only
│   ├── verify_isic_labels.py      # master_metadata vs official ISIC GroundTruth agreement check
│   ├── run_full_benchmark.sh      # orchestrates 3-seed baseline_soft/runA_grl/runB run + aggregate
│   ├── generate_paper_figures.py  # t-SNE, trade-off, leakage bar, training-stability figures
│   ├── plot_confusion_matrix_runb.py   # 8×8 confusion matrix figure
│   ├── plot_reliability_diagram.py     # calibration/reliability diagram figure
│   ├── visualize_soft_cropping_examples.py  # original/mask/crop triptych figure
│   └── generate_workflow_framework.py  # hand-drawn Figure-1 framework diagram (matplotlib patches, no data dependency)
│
└── cbm_revision/scripts/           # a second, later "journal revision" layer on top of scripts/
    ├── run_one_shot_cbm.sh        # idempotent master runner: 5-seed baseline/runA_grl/runB_orth1 → aggregate → OOD → stats → figures → summary → EffB3 control
    ├── run_effb3_control.py / .sh # EfficientNet-B3 single-encoder ablation, 5 seeds
    ├── eval_ood_benchmarks.py     # MSP/Energy/Cosine/Mahalanobis × AUROC/AUPR_IN/AUPR_OUT/FPR95, 3 methods × 5 seeds
    ├── stat_tests_cbm.py          # Welch t-test + Wilcoxon signed-rank + Cohen's d vs baseline
    ├── plot_cbm_figures.py        # trade-off, seed-stability, OOD-comparison figures for the n=5 tables
    ├── write_final_summary.py     # narrative FINAL_SUMMARY.md (n=3 vs n=5 comparison)
    └── make_effb3_publication_files.py  # publication table/figure combining EffB3 control with n=5 CBM tables

checkpoints/ -> /Users/cubo/ResearchArtifacts/checkpoints   # symlink, untracked; 16 GB / 67 .ckpt files — see §7
data/, results/, lightning_logs/                            # still absent — .gitignored, never populated in this checkout
```

`scripts/` and `cbm_revision/scripts/` are two generations of the same experiment: `scripts/run_full_benchmark.sh` is the original 3-seed benchmark, and `cbm_revision/scripts/run_one_shot_cbm.sh` is a later, idempotent 5-seed re-run added for a journal revision, layered on top of the exact same `scripts/train_*.py` / `check_leakage.py` / `aggregate_results.py` building blocks. There is no `setup.py`/`pyproject.toml`; every entry-point script inserts the repo root onto `sys.path` manually (`ROOT = Path(__file__).resolve().parents[1]`) so `from src...` imports resolve.

---

## 2. Training pipeline

### 2.1 Data preprocessing chain

```
ISIC_2019_Training_Metadata.csv  +  ISIC_2019_Training_GroundTruth.csv
                                          pad_ufes20/metadata.csv
        └──────────► src/datasets/preprocess_metadata.py ◄─────────
                          PAD_DIAG_MAP: BCC→BCC, MEL→MEL, NEV→NV,
                                        ACK→AK, SEK→BKL, SCC→SCC
                                   ▼
                     data/master_metadata.csv   (columns: path, label, domain)
                                   ▼   scripts/build_lesion_only_metadata.py (preset=soft)
                                   │   border-median diff → Otsu threshold → tight bbox
                                   │   → margin 0.30 → min_crop_ratio 0.70 → resize 224 → JPEG q95
                                   ▼
        data/master_metadata_lesion_only_soft.csv   ← metadata path used by every training script's default
```

- `preprocess_metadata.py` accepts either an ISIC metadata CSV that already has one-hot lesion columns, or falls back to a separate GroundTruth CSV (tries several filename variants, case-insensitively). PAD-UFES diagnoses are restricted to the six keys in `PAD_DIAG_MAP`; rows with any other diagnosis are dropped. Rows whose image file doesn't resolve on disk are dropped (`filter_existing_paths`).
- `build_lesion_only_metadata.py` implements a dependency-free Otsu threshold (`_otsu_threshold`, a from-scratch NumPy histogram/argmax, no `cv2`/`skimage` despite `opencv-python` being in `requirements.txt`) applied to `|gray - border_median|`, then expands and clamps the resulting bounding box before resizing to a square. `--preset soft` (margin 0.30, min_crop_ratio 0.70, autocontrast off) is the preset actually used for training; `--preset aggressive` (margin 0.10, min_crop_ratio 0, autocontrast on) exists as a CLI option.
- `verify_data_layout.py` and `verify_isic_labels.py` are read-only sanity scripts (stdout only, no artifact); the latter exits with code 3 if any label mismatch is found, intended as a CI-style gate against ISIC image/label misalignment.

### 2.2 Splits (`SkinDataModule.setup` and `splits.build_id_ood_test_dataloaders`)

Both implement **the same two-stage stratified split independently** (not shared via a common call), always with `random_state=42` regardless of the `--seed` CLI flag:

| Split | Source | Stratified by |
|---|---|---|
| ISIC train / val | 80% / 20% of the 80% ISIC train+val | `label_idx` |
| ISIC test (= ID test) | 20% of ISIC | `label_idx` |
| PAD-UFES (= OOD test) | 100% of PAD-UFES | — never split |

Because the split seed (`SplitConfig.random_state=42`) is decoupled from `--seed`, every seeded run (42/52/62/72/82) sees an identical train/val/test partition and only model init / data-loading order vary — this is deliberate (comment in `skin_dataset.py`) and lets checkpoints from different seeds be probed with row-aligned dataloaders.

**Structural point to flag:** `SkinDataModule.setup()` builds the CSG training stream as `CombinedTrainDataset(isic_train, self.pad_df_all, ...)` where `pad_df_all` is *all* PAD-UFES rows, and the same `pad_df` (all rows) is what `test_dataloader()` / `build_id_ood_test_dataloaders` return as the "OOD test set." So the entire OOD test set is also the CSG auxiliary training domain stream — every CSG model sees every one of its own "OOD" test images repeatedly during training. This is true by construction in the code (`skin_dataset.py:412-432`, `splits.py:63-96`) regardless of which specific numbers are reported in any results file. See §8 for the consequence.

### 2.3 Transforms (`src/datasets/skin_dataset.py`)

| Name | Pipeline |
|---|---|
| `build_train_transform_robust` | Resize 256 → RandomResizedCrop 224 (scale 0.7–1.0) → HFlip → VFlip → Rotation(20°) → ColorJitter(0.3,0.3,0.3,0.08) → ToTensor → ImageNet normalize → RandomErasing(p=0.25) |
| `build_val_transform_robust` | Resize 256 → CenterCrop 224 → ToTensor → normalize |
| `build_lesion_branch_transform_gray` | Resize 256 → CenterCrop 224 → Grayscale(3-channel) → ToTensor → normalize |
| "light" path (`use_robust_transforms=False`) | train: Resize(224,224) + mild ColorJitter/HFlip; eval: Resize(224,224) — no crop, no erasing |

`SkinDataModule(use_robust_transforms=...)` picks between these two families wholesale. `train_baseline.py`'s `--no_robust_transforms` flag selects the light path — this is the `baseline_soft` preset used in the actual benchmark runner scripts. All CSG (`train_csg.py`) and EffB3-control runs always use the robust path (no CLI flag to disable it there). `check_leakage.py` constructs its own `SkinDataModule` **without ever forwarding `use_robust_transforms`**, so it always evaluates every checkpoint (baseline included) under the robust eval transform — a transform mismatch for any baseline checkpoint that was trained under the light path (see §9).

### 2.4 Baseline training — `scripts/train_baseline.py`

- Model: `BaselineResNet50` → `BaselineNet` = ResNet-50 (`IMAGENET1K_V1` weights) with `fc` replaced by `Identity()` plus a separate `Linear(2048, num_classes)` classifier head; `forward(x, return_features=True)` exposes the 2048-d pooled feature.
- Loss: `CrossEntropyLoss(label_smoothing=0.03)`. Optimizer: AdamW, default lr 2e-4, wd 1e-4. LR schedule: linear warmup over `warmup_epochs` (default 5) then cosine decay to `max_epochs` (default 40), implemented as a hand-written `LambdaLR`.
- Trainer: batch size 96, 12 workers, `precision="16-mixed"` on GPU, single device, `ModelCheckpoint(monitor="val/acc", mode="max", save_top_k=1, save_last=False, filename="best-{epoch:02d}")`.
- Extra instrumentation: first-batch NaN/all-zero sanity check in `training_step`; epoch-0 val hook that predicts 5 hardcoded ISIC images from the training folder and prints pred-vs-true; per-class validation accuracy logged and printed every epoch (`on_validation_epoch_end`).
- Post-fit: reloads the best checkpoint, fits Mahalanobis parameters on `datamodule.train_dataloader()` (i.e. **under whichever transform the datamodule was built with** — light or robust, matching how the model was trained), prints an MSP/Energy/Mahalanobis OOD table, and writes `summary.json` (`id_acc`, `id_balanced_acc`, `id_ece`, `ood_maha_auroc`, `ood_maha_fpr95`, `best_checkpoint`) plus a simple hand-rolled `config.yaml` (JSON-per-line, not real YAML) into `results/baseline/<run_name>/`.

### 2.5 CSG-Lite training — `scripts/train_csg.py`

Architecture (`src/models/csg_lite.py`):

```
x_ctx ──► context_backbone (EfficientNet-B3) ──► context_projector Linear(1536→64) ──► z_context (64)
                                                          ├─► context_predictor Linear(64→2) → d_ctx
                                                          └─► context_orth_projector Linear(64→16, no bias)
x_les ──► lesion_backbone  (EfficientNet-B3) ──► lesion_projector  Linear(1536→16)
   (= gray3(x_ctx) if x_lesion not given)                └─► lesion_bn BatchNorm1d(16) ──► z_lesion (16)
                                                                  ├─► lesion_classifier Linear(16→num_classes) → y_hat
                                                                  └─► GRL(α) ─► domain_classifier_adv
                                                                        [16→32→BN→ReLU→Drop(.5)→16→BN→ReLU→Drop(.5)→2]
```

Two fully independent EfficientNet-B3 encoders (no shared weights), both ImageNet-pretrained by default. `backbone_variant="b0"` is also supported as a lighter option via the same `_build_efficientnet_backbone` helper.

Loss (`CSGLiteLightning.training_step`):
`L = L_cls + λ_ctx·L_ctx + λ_adv·L_adv + λ_orth·L_orth + λ_supcon·L_supcon`

| Term | Definition | Notes |
|---|---|---|
| `L_cls` | `F.cross_entropy(y_hat, y, ignore_index=-1)` | PAD rows carry `y=-1` (from `csg_lite_paired_collate`) and are excluded from the classification loss |
| `L_ctx` | CE(`d_ctx`, domain) | context branch is trained **to predict** domain |
| `L_adv` | CE(`d_adv`, domain) through a `GradientReversalLayer` | lesion branch is trained **to fool** the domain adversary |
| `L_orth` | `mean(cosine_similarity(z_lesion, W·z_context)²)` | `W = context_orth_projector`; drives lesion/context representations toward orthogonality |
| `L_supcon` | cross-domain supervised contrastive on L2-normalized `z_lesion`; positive pairs = same class **and** different domain; implemented with a Python `for` loop over batch rows | script default weight 1.0, but every actual benchmark preset in `run_full_benchmark.sh`/`run_one_shot_cbm.sh` passes `--lambda_supcon 0.0` |

GRL schedule (`_compute_grl_alpha`, Ganin & Lempitsky style): `progress = global_step / (estimated_stepping_batches - 1)`, `α = max(α_min, 2/(1+exp(-γ·progress^power)) - 1)`, with script defaults `γ=20`, `power=0.5`, `α_min=0.2`.

Optimizer: single AdamW instance with **three parameter groups** carved out by `id(p)` set membership — main params at `lr`, `domain_classifier_adv` at `lr × adv_lr_multiplier` (default 30), `lesion_classifier` at `lr × lesion_cls_lr_multiplier` (default 0.2). No LR scheduler.

Batching: `CombinedTrainDataset` + `csg_lite_paired_collate` emit `batch_size` ISIC + `batch_size` PAD images concatenated into one batch (so an effective batch of `2 × batch_size`); dataset length is `max(n_isic, n_pad)` with modulo-indexed cycling of the shorter side, `drop_last=True`.

A deliberate design choice recorded in-line: `training_step` always sets `images_lesion = None` and lets the model derive the lesion input in-model via `CSGLite._rgb_to_gray3(images_ctx)`, so the lesion branch always sees "grayscale after normalization" in both train and val — the dataloader still computes an externally-collated `images_lesion` tensor every step (via `lesion_branch_transform`, "grayscale before normalization") but it is discarded (see §9 for the corresponding Mahalanobis-collection code path that does *not* apply this fix).

Post-fit: reload the best checkpoint, run `ood_metrics.collect_z_lesion_labels_csg` on the **paired CSG train loader** (still includes training-time augmentation) to fit Mahalanobis params, then AUROC/FPR95 the (ISIC test) vs (PAD) split, and write the same `summary.json` shape as the baseline script.

### 2.6 EfficientNet-B3 single-encoder control — `cbm_revision/scripts/run_effb3_control.py`

`EffB3SingleNet`: `gray3(x)` → EfficientNet-B3 → `Linear(1536→16)` → `BatchNorm1d(16)` → `Linear(16→num_classes)`. Plain cross-entropy, AdamW lr 1e-4 wd 1e-4, no LR scheduler, 40 epochs, batch 32, robust transforms always on. This is deliberately built to mirror the CSG lesion branch exactly (same gray-conversion, same projector→BN→classifier shape) with the context branch and every disentanglement loss removed — an ablation designed to answer "is the leakage reduction just a bigger backbone, or does disentanglement do real work?" This script is self-contained: it trains missing seeds (`--run_train`), evaluates a domain linear probe and 4-detector OOD scores itself (duplicating logic also found in `eval_ood_benchmarks.py` and `check_leakage.py`), and writes its own `metrics.csv` / `leakage_probe.csv` / `ood_scores.csv` / `comparison_table.csv` / `summary.txt` into `results/effb3_control/`. Its final comparison table reads the n=5 CBM JSONs (`results/cbm_revision/*_n5.json`) directly, so it must be run *after* `run_one_shot_cbm.sh` has produced those.

### 2.7 Orchestration

| Runner | What it does | Seeds | Methods |
|---|---|---|---|
| `scripts/run_full_benchmark.sh` | Trains + probes + aggregates, **overwrites `results_final_v1.csv`** at start | 42, 52, 62 | `baseline_soft`, `runA_grl`, `runB` (λ_orth=5.0) |
| `cbm_revision/scripts/run_one_shot_cbm.sh` | Idempotent — skips any seed whose `summary.json`/`leakage.json` already exists; also runs the extended OOD benchmarks, stats, figures, narrative summary, and the EffB3 control at the end | 42, 52, 62, 72, 82 | `baseline_soft`, `runA_grl`, `runB_orth1` (λ_orth=1.0) |
| `cbm_revision/scripts/run_effb3_control.sh` | EffB3 control only | 42, 52, 62, 72, 82 | `effb3_single` |

`run_full_benchmark.sh`'s `runB` and `run_one_shot_cbm.sh`'s `runB_orth1` are **different method configurations under overlapping-but-distinct names** — `runB` uses `--lambda_orth 5.0`, `runB_orth1` uses `--lambda_orth 1.0`; the 5-seed runner never touches the λ=5.0 configuration at all. Any downstream table needs to be read with this naming in mind.

### 2.8 Method presets (from script/runner defaults; per-checkpoint `hparams` cannot be verified in this checkout — see intro)

| Method | script | λ_ctx | λ_adv | λ_orth | λ_supcon | lr | epochs (as run) | transforms |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `baseline_soft` | train_baseline.py | – | – | – | – | 2e-4 | 40 | light (`--no_robust_transforms`) |
| `runA_grl` | train_csg.py | 1.0 | 2.0 | 0.0 | 0.0 | 1e-4 | 40 | robust |
| `runB` (orth=5) | train_csg.py | 1.0 | 2.0 | 5.0 | 0.0 | 1e-4 | 40 | robust |
| `runB_orth1` | train_csg.py | 1.0 | 2.0 | 1.0 | 0.0 | 1e-4 | 40 | robust |
| `effb3_single` | run_effb3_control.py | – | – | – | – | 1e-4 | 40 | robust |

All CSG runs (as launched by the runner scripts): `lesion_latent_dim=16`, `context_latent_dim=64`, `backbone_variant=b3`, `adv_lr_multiplier=30`, `lesion_cls_lr_multiplier=0.2`, `grl_gamma=20`, `grl_progress_power=0.5`, `grl_alpha_min=0.2`, `weight_decay=1e-4`. (Note: `train_csg.py`'s own module-level defaults are `MAX_EPOCHS=20`, `λ_adv=5.0`, `λ_orth=10.0`, `λ_supcon=1.0` — these are only the argparse fallbacks for direct invocation; every actual benchmark run overrides them via CLI flags as shown in the table.)

---

## 3. Datasets

Two source datasets, merged by `preprocess_metadata.py` into a single `path,label,domain` schema over the shared 8-class label space `["MEL","NV","BCC","AK","BKL","DF","VASC","SCC"]` (`src/datasets/constants.py`):

- **ISIC 2019** (`domain="isic"`) — expects `ISIC_2019_Training_Metadata.csv` (demographics) and a one-hot GroundTruth CSV (several filename variants tried, case-insensitive glob fallback); an `AKIEC→AK` legacy rename handles ISIC-2018-style column names. Images are resolved under `ISIC_2019_Training_Input/` with a nested-folder fallback and a `*_downsampled.jpg` fallback.
- **PAD-UFES-20** (`domain="pad_ufes"`) — expects `pad_ufes20/metadata.csv` with `img_id` and `diagnostic` columns; only rows whose `diagnostic` is one of `{BCC, MEL, NEV, ACK, SEK, SCC}` survive the `PAD_DIAG_MAP` filter (i.e. PAD's `NEV`→`NV`, `ACK`→`AK`, `SEK`→`BKL`; any other PAD diagnosis, if present in the raw data, is silently dropped by the `.isin()` filter before mapping).

`load_filtered_master` (in both `splits.py` and re-implemented inline in `SkinDataModule.setup`) re-filters the merged CSV to rows whose `label` is a known class and whose `path` exists on disk — this filtering runs identically in at least three places (`skin_dataset.py`, `splits.py`, and inline in several `cbm_revision` scripts building an "ISIC-train-under-eval-transform" loader), which is a duplication point worth being aware of if the label set or filtering logic ever needs to change (see §9.6).

`build_lesion_only_metadata.py` produces the *actual* metadata CSV used for training (`master_metadata_lesion_only_soft.csv`) by re-writing every image path to point at an Otsu-cropped copy; the original `master_metadata.csv` (uncropped) is only the intermediate artifact.

Because none of `data/master_metadata*.csv` are present in this checkout, row counts, per-class distribution, and the PAD-UFES DF/VASC-class asymmetry reported in `docs/repository_audit.md` §1.1 **cannot be independently re-verified here** — they are properties of the actual dataset files, not of the code.

---

## 4. Models

| File | Class | Role |
|---|---|---|
| `src/models/baseline.py` | `BaselineNet` / `BaselineResNet50` | ResNet-50 backbone, single classifier head, the "no disentanglement" reference point |
| `src/models/csg_lite.py` | `CSGLite` | The paper's proposed architecture: two independent EfficientNet-B3 encoders (lesion + context), GRL-based adversarial domain unlearning on the lesion branch, and a cosine-orthogonality term between branches |
| `src/models/csg_lightning.py` | `CSGLiteLightning` | Lightning training wrapper around `CSGLite`: composes the 5-term loss, the GRL alpha schedule, the 3-group optimizer, and per-epoch debug/print instrumentation |
| `src/models/effb3_single.py` | `EffB3SingleNet` / `EffB3SingleLightning` | Single-encoder ablation matching the CSG lesion branch exactly (no context branch, no adversary) |
| `src/models/backbone.py` | — | 1-line stub; no shared backbone factory exists — each model file builds its own backbone inline |
| `src/models/csg_full.py` | — | 1-line stub; **not implemented**. `configs/csg.yaml` references `variant: csg_lite # or csg_full`, but `csg_full` does not exist anywhere in the codebase — that config option is aspirational only |

`GradientReversalLayer` (in `csg_lite.py`) is a standard `torch.autograd.Function`-based implementation (identity forward, `-λ·grad` backward) with a mutable `lambd` set per-step from the Lightning module's schedule.

---

## 5. Feature / latent extraction

There is no single canonical feature-extraction utility — instead there are **several independently-written collection functions**, each returning a different subset of latents and handling a different batch arity:

| Function | Location | Returns |
|---|---|---|
| `collect_logits_labels_features_baseline` | `src/utils/ood_metrics.py` | `(logits, labels, features)` for a baseline checkpoint (2048-d backbone) |
| `collect_z_lesion_labels_csg` | `src/utils/ood_metrics.py` | `(z_lesion, labels)`, drops rows with `label<0` (i.e. PAD rows in the paired loader) |
| `collect_test_features_csg` / `collect_test_features_baseline` | `scripts/check_leakage.py` | `(z_lesion, z_context, backbone_raw, logits, labels)` — richest single collector; for a baseline checkpoint all three "latent" slots are filled with the same 2048-d feature (documented in-code) |
| `collect_logits_z_labels` | `scripts/eval_ood_scores.py` | `(logits, z_lesion, z_context, labels)` |
| `_collect_csg_features` / `_collect_baseline_features` | `cbm_revision/scripts/eval_ood_benchmarks.py` | `(logits, z, labels)` |
| `_collect_logits_features` | `cbm_revision/scripts/run_effb3_control.py` | `(logits, z, labels)` for the single-encoder model |
| `collect_latents_and_domains` | `scripts/generate_paper_figures.py` | `(z_lesion, z_context, domains)`, for t-SNE plotting |

Latent dimensions, as defined by the model classes (verifiable from `src/models/*.py` regardless of whether any checkpoint exists):

| Logical name | Dim | Source |
|---|---:|---|
| `z_lesion` | 16 (`lesion_latent_dim` default) | `CSGLite.encode_z_lesion` / `forward(return_latents=True)` |
| `z_context` | 64 (`context_latent_dim` default) | `CSGLite.context_projector(extract_context_features(x))` |
| `backbone_raw` (context) | 1536 | `CSGLite.extract_context_features` (EfficientNet-B3 pooled) |
| baseline features | 2048 | `BaselineNet.forward(return_features=True)` (ResNet-50 pooled) |
| EffB3-control `z` | 16 (`latent_dim` default) | `EffB3SingleNet.extract_embedding` |
| logits | `num_classes` (8) | all models |

**No script anywhere persists these latents to disk.** A grep across `src/`, `scripts/`, `cbm_revision/` for `np.save`, `np.savez`, `torch.save`, `to_pickle`, literal `.npy`/`.npz` turns up nothing but the argparse flag name `--ckpt_for_embedding` and the method name `extract_embedding` — no actual write call. Every representation above is recomputed in memory on every script invocation and discarded on exit; there is no embedding-cache artifact type in this codebase at all, by design or omission.

---

## 6. Experiment scripts

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `src/datasets/preprocess_metadata.py` | build master metadata | ISIC GroundTruth + PAD metadata | `data/master_metadata.csv` |
| `scripts/build_lesion_only_metadata.py` | Otsu-based lesion crop | master metadata | cropped JPEGs + `*_lesion_only*.csv` |
| `scripts/verify_data_layout.py` | path/layout check | `data/` | stdout only |
| `scripts/verify_isic_labels.py` | master vs. GroundTruth agreement | master + GroundTruth | stdout only (exit code 3 on mismatch) |
| `scripts/train_baseline.py` | train ResNet50 + OOD table | metadata | ckpt, `summary.json`, `config.yaml`, CSV logs |
| `scripts/train_csg.py` | train CSG-Lite + Mahalanobis OOD | metadata | ckpt, `summary.json`, `config.yaml`, CSV logs |
| `scripts/check_leakage.py` | domain linear probe (the paper's central metric) | ckpt + metadata | `leakage.json` (optional, via `--output_json`) |
| `scripts/eval_ood_scores.py` | Energy/MSP/Cosine/Mahalanobis on `z_lesion` & `z_context` | ckpts | **stdout only — no artifact** |
| `scripts/aggregate_results.py` | merge `summary.json` + `leakage.json` across seeds → mean±std | run JSONs | `*_aggregate.{md,json}`, optional CSV append |
| `scripts/generate_paper_figures.py` | t-SNE, trade-off, leakage bar, training-stability figures | ckpt + aggregate JSONs + CSV logs | figure PNG/PDF pairs |
| `scripts/plot_confusion_matrix_runb.py` | confusion matrix figure | ckpt | `fig_confusion_matrix.{png,pdf}` |
| `scripts/plot_reliability_diagram.py` | calibration diagram | 2 ckpts | `fig_reliability_baseline_vs_runB.{png,pdf}` |
| `scripts/visualize_soft_cropping_examples.py` | preprocessing visualization | master metadata | `fig_soft_crop_examples.{png,pdf}` |
| `scripts/generate_workflow_framework.py` | Figure-1 diagram (no data dependency) | — | `figure1_framework.{png,pdf}` |
| `cbm_revision/scripts/eval_ood_benchmarks.py` | 4 detectors × 4 metrics × 5 seeds | ckpt dirs | `ood_comparison.{csv,json}` |
| `cbm_revision/scripts/stat_tests_cbm.py` | Welch + Wilcoxon + Cohen's d | n=5 JSONs | `stat_tests.{csv,json}` |
| `cbm_revision/scripts/plot_cbm_figures.py` | 3 revision figures | n=5 JSONs + OOD CSV | `results/cbm_revision/figures/*` |
| `cbm_revision/scripts/write_final_summary.py` | narrative n=3-vs-n=5 summary | n=3 + n=5 JSONs | `FINAL_SUMMARY.md` |
| `cbm_revision/scripts/run_effb3_control.py` | train + evaluate EffB3 control | metadata / ckpts | 5 CSVs + per-seed JSONs + `summary.txt` |
| `cbm_revision/scripts/make_effb3_publication_files.py` | publication tables/figure for control | control CSVs + n=5 JSONs | `aggregate_stats.csv`, `significance_effb3.csv`, `backbone_control_tradeoff.png`, `concise_results.md` |

### 6.1 The leakage probe — exact specification (`scripts/check_leakage.py`)

This is the paper's central metric, so it's worth stating precisely:

1. Load a checkpoint; `--model_type auto` infers `csg` vs `baseline` from state-dict key prefixes (`model.lesion_backbone.*` / `model.context_backbone.*` → csg; `net.backbone.*` → baseline; defaults to `csg` if neither matches).
2. Build a `SkinDataModule` (**always with the default `use_robust_transforms=True`** — the `--no_robust_transforms` CLI flag that exists on `train_baseline.py` has no equivalent here) and take `test_dataloader()` = ISIC test + all PAD, `shuffle=False`.
3. Domain targets come from the dataframe's `domain` column: `isic→0`, `pad_ufes→1`.
4. Extract, in loader order, `z_lesion`, `z_context`, `backbone_raw` (for baseline checkpoints all three are the same 2048-d feature).
5. For each probe seed (`--probe_seeds`, default `42,52,62`): stratified 70/30 train/test split → `StandardScaler` → `LogisticRegression(max_iter=2000)` → held-out accuracy; report mean ± std across seeds.
6. Sanity control: same probe on `z_lesion` with domain labels shuffled (`--shuffle_test_seed`) — this should reduce to the majority-class baseline if the probe pipeline itself is correctly implemented.
7. Also recomputes ID accuracy / balanced accuracy / 15-bin ECE on the ISIC-test-only loader.
8. Writes `leakage.json` if `--output_json` is given.

### 6.2 Mahalanobis — exact specification (`src/utils/ood_metrics.py`)

- `compute_mahalanobis_params_from_arrays`: per-class means; a single **pooled within-class covariance** `Σ = Σᵢ (xᵢ − μ_{yᵢ})(xᵢ − μ_{yᵢ})ᵀ / (N − K)`, regularized by `+ reg_eps·I`, inverted once for a shared precision matrix. Raises if any class has zero samples in the fit set.
- `mahalanobis_min_squared_distances`: a **pure Python double loop over samples × classes** in float64 — no vectorization; this is a real performance bottleneck for large fit sets (see §8).
- `reg_eps` differs by caller: `1e-5` (the function default, used by `train_baseline.py`/`train_csg.py`) vs. `1e-3` (explicitly passed by `eval_ood_scores.py`, `eval_ood_benchmarks.py`, `run_effb3_control.py`).
- The fitting set also differs by caller: `train_csg.py` fits on the **augmented paired CSG train loader** (training-time augmentation active); `eval_ood_scores.py` / `eval_ood_benchmarks.py` build a separate ISIC-train loader **under the eval transform** specifically to avoid that. These are two different estimators that happen to share the name "Mahalanobis AUROC" in different result tables — see §8.

---

## 7. Saved artifacts, checkpoints, and outputs

`checkpoints/`, `results/`, `lightning_logs/`, and `data/` are all real directories the code is written to produce and consume — every training script writes `ckpt_dir / "best-{epoch:02d}.ckpt"` via `pl.callbacks.ModelCheckpoint`, and every training/eval script writes `run_dir / "summary.json"` and (for `check_leakage.py`) `leakage.json`. Of these, **`checkpoints/` is now present** (symlinked, 16 GB, 67 `.ckpt` files); `results/`, `lightning_logs/`, and `data/` remain absent. Two root-level CSVs are also present and give a frozen snapshot of what a completed run once produced:

- **`results_final_v1.csv`** — 4 rows (`Baseline Soft`, `Run A / GRL`, `Run B (orth=5.0)`, `Run B (orth=1.0)`), each `Method,Acc,Bal Acc,ECE,Leakage` at n=3, produced by `scripts/aggregate_results.py --append_csv` as called from `run_full_benchmark.sh`.
- **`RESULTS_TABLES.csv`** — the same 4 aggregate rows plus 12 per-seed rows, with extra `OOD_Maha_AUROC` / `OOD_Maha_FPR95` / `Notes` columns naming specific checkpoint files (e.g. `best-31.ckpt`). No script in this repository produces this exact file/schema — it does not match `aggregate_results.py`'s output shape (`Method,Acc,Bal Acc,ECE,Leakage` only, no OOD or per-seed columns), so this file must have been hand-assembled or produced by a script that isn't part of this checkout.

### 7.1 Checkpoint inventory (verified directly against `/Users/cubo/ResearchArtifacts/checkpoints`)

67 `.ckpt` files, ~16 GB total. Per-architecture file size is constant and matches the model definitions in `src/models/*.py`: ResNet-50 baseline 282,756,535 B, CSG-Lite dual-EfficientNet-B3 259,974,997 B, EffB3-single 129,515,273 B.

**Named run directories** (`<method>_s<seed>/`, matching the `run_name` convention in `train_baseline.py`/`train_csg.py`/`run_effb3_control.py`):

| Method | Seeds present | ckpts per dir |
|---|---|---|
| `baseline_soft` | 42, 52, 62, 72, 82 (all 5) | 1 (`best-*.ckpt` only, `save_last=False` as coded in `train_baseline.py`) |
| `runA_grl` | 42, 52, 62, 72, 82 (all 5) | 2 (`best-*.ckpt` + `last.ckpt`) |
| `runB_orth1` | 42, 52, 62, 72, 82 (all 5) | 2 each, **except s42 has 4** (stale duplicate, see below) |
| `runB` (orth=5.0) | **only 42, 52, 62 — no s72/s82** | 6, 6, 4 respectively (all stale duplicates, see below) |
| `effb3_single` | 42, 52, 62, 72, 82 (all 5) | 2 (`best-*.ckpt` + `last.ckpt`) |

This reproduces exactly the checkpoint availability described in `docs/repository_audit.md` §3.1: 23 "final" checkpoints across 5 methods, with `runB` (λ_orth=5.0) present for only 3 of 5 seeds.

**Stale duplicate checkpoints** — directories that accumulated extra `best-*`/`last-v*` files from repeated `trainer.fit()` invocations into the same `dirpath`:

| Run dir | Files present |
|---|---|
| `csg_lite/runB_s42/` | `best-31, best-37, best-39, last, last-v1, last-v2` |
| `csg_lite/runB_s52/` | `best-01, best-28, best-34, last, last-v1, last-v2` |
| `csg_lite/runB_s62/` | `best-28, best-39, last, last-v1` |
| `csg_lite/runB_orth1_s42/` | `best-36, best-37, last, last-v1` |

**Legacy/exploratory loose checkpoints** (not inside any named run directory — pre-benchmark tuning artifacts): `checkpoints/baseline/*.ckpt` — 6 files (`best-05`, `best-06`, `best-08`, `best-27`, `best-30`, `best-34`, each 269.7 MB); `checkpoints/csg_lite/*.ckpt` — 8 files (`best-14`, `best-18`, `best-25`, `best-31_best`, `last`, `last-v1`, `last-v2`, `last-v3`, each ~247.9–248.9 MB). Their specific validation metrics (e.g. which epoch reached which `val/acc`) cannot be confirmed without loading the checkpoint's `hyper_parameters`/logged metrics, which requires `torch` — not available in this session.

### 7.2 Confirmed: the "newest checkpoint in directory" resolution bug actually fires here

§8 below flags `ood_metrics.find_checkpoint` (picks `max(mtime)` over **all** `*.ckpt` in a directory, best and last mixed together) as fragile whenever stale duplicates exist. With real timestamps now available, this was checked directly rather than left as a theoretical risk — for every stale directory listed above, the checkpoint `find_checkpoint` would resolve is **not** the one `ls -1t best-*.ckpt | head -1` (the shell runners' `find_best_ckpt`, used right after training) would have picked:

| Run dir | Newest `best-*` (what training considered "best") | What `find_checkpoint` (max mtime over *all* `*.ckpt`) actually resolves |
|---|---|---|
| `csg_lite/runB_s42/` | `best-39.ckpt` | `last-v2.ckpt` |
| `csg_lite/runB_s52/` | `best-28.ckpt` | `last-v2.ckpt` |
| `csg_lite/runB_s62/` | `best-39.ckpt` | `last-v1.ckpt` |
| `csg_lite/runB_orth1_s42/` | `best-36.ckpt` | `last-v1.ckpt` |
| `csg_lite/runB_orth1_s52/` | `best-32.ckpt` | `last.ckpt` |
| `csg_lite/runB_orth1_s62/` | `best-34.ckpt` | `last.ckpt` |

(Verified by running the same `max(..., key=mtime)` logic `find_checkpoint` uses, via `pathlib.Path.glob`, directly against these directories.) In every one of these six cases, any script that resolves a checkpoint from the run *directory* rather than being handed the specific `best-*.ckpt` path would silently evaluate a different (later-epoch, not best-val) model than the one the training run actually selected — and 3 of these 6 are `runB_orth1` seeds, the method with the paper's headline conclusion. `check_leakage.py --ckpt <path-or-dir>` and `plot_confusion_matrix_runb.py`/`plot_reliability_diagram.py` (whose `--ckpt`/`--*_ckpt` args also accept a directory) are all exposed to this if invoked with a directory instead of an explicit file path.

Anything beyond filesystem-level facts (per-seed metric values inside `summary.json`/`leakage.json`, `data/` row counts, exact hyperparameters recorded in each checkpoint's `hyper_parameters`) is **still not verifiable in this checkout** — `data/` and `results/` remain absent, and no `torch` install was available to open the `.ckpt` files. `docs/repository_audit.md` contains such an inventory from a session that had that additional state (and a working `torch`) available; its checkpoint-existence claims are now independently corroborated by the filesystem check above, but its specific *metric values* remain unverified here.

---

## 8. Assumptions and structural risks baked into the code

These are properties of the code itself (verified by reading it), independent of any particular training run:

1. **The CSG "OOD test set" is, by construction, the CSG training script's own auxiliary domain stream.** `SkinDataModule.setup()` passes *all* PAD-UFES rows into `CombinedTrainDataset` for CSG training, and the same all-PAD dataframe is what both `test_dataloader()` and `build_id_ood_test_dataloaders()` hand back as the OOD test set (`skin_dataset.py:412-432`, `splits.py:63-96`). The ResNet-50 baseline never trains on PAD, so any baseline-vs-CSG OOD comparison contrasts a model evaluated on genuinely unseen data against a model evaluated on data it trained on ~7× per epoch for 40 epochs. This is a property of the current split logic, not of any specific result — it would need a held-out PAD partition (train-aux vs. OOD-test) before any CSG OOD-detection number is meaningful.
2. **Mahalanobis fit/score transform mismatch in `train_csg.py`'s post-fit path.** The fit set is the paired CSG train loader with training-time augmentation active; `collect_z_lesion_labels_csg` receives the loader's `images_lesion` tensor (grayscale computed *before* normalization) and passes it through `encode_z_lesion`, which applies `_rgb_to_gray3` *again* on already-processed channels — while `training_step` deliberately avoids exactly this double-conversion by forcing `images_lesion=None` (a comment in `csg_lightning.py` documents that this fix was applied to train/val but not to the Mahalanobis-collection path). Meanwhile `eval_ood_scores.py`/`eval_ood_benchmarks.py` fit on a clean, single-conversion ISIC-train-under-eval-transform loader. These two paths are legitimately different estimators; any table that reports "Mahalanobis AUROC" needs to say which pipeline produced it.
3. **`aggregate_results.py` merges `summary.json` and `leakage.json` with `leakage.json` silently winning.** Both files carry `id_acc`/`id_balanced_acc`/`id_ece` for (nominally) the same checkpoint; `merged.update(l)` after `merged.update(s)` means the leakage-side value always wins in any merged/aggregated table. Because `check_leakage.py` doesn't forward `use_robust_transforms`, this specifically means baseline runs trained under the light transform get re-scored under the robust transform for every published aggregate — a systematic, one-directional bias against the baseline in any table built by `aggregate_results.py`.
4. **"Newest checkpoint in directory" resolution is fragile — and confirmed to actually misfire in the checkpoints now present in this checkout.** `ood_metrics.find_checkpoint` and several `_find_ckpt` re-implementations (in `run_effb3_control.py`, `eval_ood_benchmarks.py`) pick `max(mtime)` over `*.ckpt`, preferring `best-*` then falling back to `last*`. Several `csg_lite` run directories accumulated stale checkpoints from repeated `trainer.fit()` invocations into the same `dirpath` (`ModelCheckpoint(save_last=True)` writes `last-v1.ckpt`, `last-v2.ckpt`, etc. on each re-run); for six such directories this resolution provably picks a different (later-epoch, non-best-val) checkpoint than the one training selected as best — see §7.2 for the verified per-directory table.
5. **Six-plus near-duplicate feature-extraction implementations** (§5) differ subtly in batch-arity handling and label masking; a bug fixed in one is not automatically fixed in the others (item 2 above is a direct instance of this).
6. **`_compute_ece` / `compute_ece_from_logits` / `_ece` is copy-pasted (not shared) across `train_baseline.py`, `train_csg.py`, `check_leakage.py`, and `run_effb3_control.py`** — all four implementations are algorithmically identical 15-bin ECE, but any future fix has to be applied four times.
7. **Performance:** `mahalanobis_min_squared_distances` is an O(N·K) pure-Python loop; for realistic ISIC-scale fit sets this dominates the runtime of every OOD-evaluation script. `_supervised_contrastive_cross_domain_loss` loops over all batch rows in Python per training step even though every benchmark preset sets `λ_supcon=0.0`, so the (zero-weighted) loss is still computed in full on every step of every CSG run.
8. **Configuration/dependency drift:**
   - `configs/baseline.yaml` / `configs/csg.yaml` are never loaded by any script (`yaml`/`OmegaConf` is not imported anywhere in `src/` or `scripts/`) — their contents (`lr 1e-4`, `wd 1e-2`, `max_epochs 50`, `precision 32`, `variant: csg_full`) do not match any actual run preset in §2.8 and would be actively misleading if used as a reference.
   - `requirements.txt` lists `opencv-python` (never imported anywhere — the Otsu implementation is hand-rolled NumPy) and omits `torchmetrics`, `scipy`, and `seaborn`, all of which are directly imported (`torchmetrics` in `baseline.py`/`csg_lightning.py`/`effb3_single.py`; `scipy.stats` in `stat_tests_cbm.py`/`make_effb3_publication_files.py`; `seaborn` in `plot_confusion_matrix_runb.py`). No version pins at all.
   - `scripts/generate_workflow_framework.py --output_dir` defaults to the hardcoded absolute path `/mnt/data2/Vinh/CSG-Skin/results/figures` — a machine-specific path baked into the script rather than derived from `PROJECT_ROOT`.
   - `scripts/plot_reliability_diagram.py --baseline_ckpt` defaults to `checkpoints/baseline_soft`, which does not match the run-directory naming convention used elsewhere (`checkpoints/baseline/<run_name>`); it survives only via a keyword-based recursive fallback scan (`_resolve_checkpoint_with_fallback`) that picks the newest matching `.ckpt` anywhere under `checkpoints/` — non-deterministic if stale checkpoints are present.
   - `generate_paper_figures.py` and `eval_ood_scores.py` both hardcode a default checkpoint path of `checkpoints/csg_lite/runB_s42/best-31.ckpt` — a specific epoch number baked in as a script default rather than resolved via `find_checkpoint`.
9. **`src/models/csg_full.py`, `src/losses/indep_loss.py`, `src/losses/adv_loss.py`, `src/models/backbone.py`, `src/datasets/isic.py`, `src/datasets/pad_ufes.py`, `src/utils/metrics.py`, `src/utils/logging_utils.py`, `src/utils/visualization.py`** are all 1–2 line stub files containing only a pointer comment — none has an implementation. `IndependenceLoss` (in `csg_losses.py`, cross-covariance formulation) is fully implemented but never instantiated by any script — the actual orthogonality objective used in training is the inline cosine-similarity term in `csg_lightning.py`, not this class. `CSGTrainDataset` and `splits.build_csg_train_dataframe` are fully implemented but imported nowhere — dead alternatives to `CombinedTrainDataset`.
10. **`generate_paper_figures.py`'s figure is named `figure_umap_tsne_disentanglement.png`** but the function that produces it (`plot_tsne_disentanglement`) computes t-SNE only — `umap-learn` is not imported anywhere in the codebase, so the "umap" in the filename does not correspond to a UMAP computation.
11. Both `train_baseline.py` and `train_csg.py` add a `TensorBoardLogger` conditionally on `importlib.util.find_spec("tensorboard")` — if TensorBoard isn't installed in the environment, this silently degrades to CSV-only logging with no error.

---

## 9. Limitations of this guide

- This checkout now has `checkpoints/` (verified directly, §7.1–7.2) but still has no `data/`, `results/`, or `lightning_logs/` — dataset statistics, per-run `summary.json`/`leakage.json`/`config.yaml` contents, CSV training logs, and figures are still about the *code's logic* only, not verified against actual run output. Anyone using this guide to plan new work needs `data/` and `results/` populated (via `preprocess_metadata.py` → `build_lesion_only_metadata.py` → the training/eval scripts) before metrics can be reproduced.
- No Python environment with `torch` was available in this session, so checkpoint **contents** (state-dict keys, `hyper_parameters`, logged metrics, exact tensor shapes) could not be opened and inspected directly — the architecture facts in this guide come from reading `src/models/*.py`, and the checkpoint-inventory facts in §7 come from filesystem metadata (paths/sizes/mtimes) only.
- `configs/baseline.yaml` / `configs/csg.yaml` cannot be used as a source of truth for hyperparameters (§8, item 8) — always read the actual `train_*.py` argparse defaults and the runner shell scripts' explicit flags instead.
- Two CSVs at the repo root (`results_final_v1.csv`, `RESULTS_TABLES.csv`) are the only concrete experimental *metric* numbers available in this checkout; they represent a specific n=3 run on a specific (unavailable-here) dataset copy and should not be treated as reproducible without `data/` and a working `torch` environment to actually re-run the pipeline against the checkpoints.
- This guide deliberately does not restate `docs/repository_audit.md`'s dataset- or metric-specific findings (row counts, per-seed `val/acc`, OOD AUROC values) as fact, since `data/` and `results/` are still absent here. Where the two documents describe the same *checkpoint-existence* facts (§7.1–7.2), they are now independently corroborated on this filesystem; where the prior audit describes `data/`/`results/` contents or metric values, that remains out of scope here.

---

## 10. Reusable components for future work

**Directly reusable as-is:**
1. **The checkpoints under `checkpoints/`** (§7.1) — 5 fully-seeded methods (`baseline_soft`, `runA_grl`, `runB_orth1`, `effb3_single`, all ×5 seeds) plus `runB` (orth=5.0) at 3/5 seeds, all loadable in principle via `BaselineResNet50.load_from_checkpoint` / `CSGLiteLightning.load_from_checkpoint` / `EffB3SingleLightning.load_from_checkpoint` given a `torch`/`pytorch-lightning` environment (not available in this session, so loading itself is untested here). **Before reusing any `runB` or `runB_orth1` checkpoint by pointing a script at its run *directory* rather than an explicit file**, check it against the stale-duplicate table in §7.2 first — for `runB_s{42,52,62}` and `runB_orth1_s42`, directory-based resolution (`find_checkpoint`) does not return the trained-best checkpoint.
2. **`src/datasets/skin_dataset.py` / `splits.py`** — a deterministic, stratified ISIC/PAD split (`random_state=42`, decoupled from `--seed`) with a paired-domain training dataset (`CombinedTrainDataset`) and a matching ID/OOD dataloader builder. Any new model can plug into `SkinDataModule` directly.
3. **`src/utils/ood_metrics.py`** — self-contained Mahalanobis fit/score, FPR@95, and MSP/Energy table printing; correct as written, just slow (O(N·K) Python loop) and should be vectorized before scaling up sample counts.
4. **`src/models/csg_lite.py` + `csg_lightning.py`** — the full dual-encoder architecture and training loop, including the GRL schedule and 3-group optimizer, is a clean reference implementation to extend (e.g. toward the never-built `csg_full` variant) or to swap backbones (`b0`/`b3` already parameterized).
5. **`scripts/check_leakage.py`'s linear-probe methodology** — domain-leakage-via-logistic-probe with a shuffle-label sanity control and multi-seed stability reporting is a solid, reusable evaluation pattern for any future disentanglement claim, independent of this specific dataset.
6. **`scripts/build_lesion_only_metadata.py`'s Otsu-crop pipeline** — dependency-free (no cv2/skimage), parameterized by margin/min-crop-ratio/autocontrast, easily reused for other lesion-image preprocessing needs.
7. **The figure-generation scripts** (`generate_paper_figures.py`, `plot_confusion_matrix_runb.py`, `plot_reliability_diagram.py`) share a consistent style helper (`configure_plot_style`) and PNG@300dpi+PDF dual-export pattern — worth copying forward as the house style for future figures.

**Reusable with modification (fix the known issue first):**
- `aggregate_results.py` — generic mean±std aggregator; fix the `leakage.json`-wins merge order (§8 item 3) before trusting any aggregate it produces.
- `scripts/eval_ood_scores.py` / `cbm_revision/scripts/eval_ood_benchmarks.py` — correct Mahalanobis fitting methodology (clean ISIC-train-under-eval-transform); worth consolidating into a single shared feature-extraction module rather than the current 6-way duplication (§8 item 5) before extending further.
- `cbm_revision/scripts/stat_tests_cbm.py` / `make_effb3_publication_files.py` — Cohen's d + Welch scaffolding is straightforward to extend to more methods, but keep in mind Wilcoxon signed-rank cannot reach p<0.05 at n=5 (floor is 0.0625) — this is a property of the test at that sample size, not a bug, and any future write-up should lead with Cohen's d rather than the Wilcoxon p-value at this n.

**What would need to be built, not just reused:**
- A held-out PAD partition (train-aux split vs. genuine OOD-test split) — required before any CSG OOD-detection claim is meaningful (§8 item 1).
- Any embedding-cache mechanism — currently nothing is ever persisted to disk (§5); every latent is recomputed per-script-invocation.
- `src/models/csg_full.py` — referenced by `configs/csg.yaml` but never implemented.
- A single consolidated feature-extraction module to replace the 6+ near-duplicate collectors (§8 item 5), which would also remove the specific train/eval transform mismatch bug in item 2 as a side effect of having one code path instead of two.
