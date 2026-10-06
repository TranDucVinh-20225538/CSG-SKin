# Paper B — Phase 0 verification gates

**Date:** 2026-09-20
**Repo (read-only):** `/data2/hpcshared/Vinh/CSG-Skin`
**Writable tree:** `/data2/cmdir/home/toandq/CSG-Skin-paperB` (see §Deviation)
**Verdict: all five gates PASS. Phase 1 is authorised. No Phase 0.2 re-eval.**

| Gate | Verdict | Blocking? |
|---|---|---|
| 0.1 Inventory + checkpoint policy | PASS (runB orth=5 missing s72/s82; not needed for Phase 1's 4-method grid) | No |
| 0.2 Train/eval transform parity | **PASS — matched.** Both train and eval derive lesion input via in-model `_rgb_to_gray3` after ImageNet norm. The METHODS-documented pre-norm `Grayscale` transform is built by the dataloader and then **discarded**. | Would have been yes if mismatched |
| 0.3 Label → domain probe | PASS (computed). Label-only accuracy **0.7968** vs majority **0.6880** | No |
| 0.4 Split hygiene | PASS. Explicit **YES**: the PAD images used in the adversarial/context branches at train time are the same 2,298 images later used as the OOD set. | No (expected YES; this is P2's mechanism) |
| 0.5 Environment | PASS. Written to `ENVIRONMENT.md`. Scheduler is SLURM. | No |

---

## 0.1 Inventory

Seeds requested: 42 / 52 / 62 / 72 / 82.
Policy (applied everywhere from here on): **best-by-val/acc = the most recently written `best-*.ckpt` in the run directory.** Fallback to newest `last*.ckpt` only if no `best-*.ckpt` exists. Never use `find_checkpoint()` (max mtime over *all* `*.ckpt`), which is how several `leakage.json` files ended up pointing at `last-v1.ckpt`.

This matches `ModelCheckpoint(monitor="val/acc", mode="max", save_top_k=1)` and `cbm_revision/scripts/eval_ood_benchmarks.py::_find_ckpt`. When a run was relaunched, older `best-*.ckpt` files were left behind; newest-mtime `best-*` is the later run's val-acc winner.

Machine-readable table: `phase0_checkpoint_inventory.csv`.

| method | seed | exists | policy ckpt | kind | also present | leakage.json pointed at |
|---|---:|---|---|---|---|---|
| baseline_soft | 42 | yes | `checkpoints/baseline/baseline_soft_s42/best-20.ckpt` | best | — | best-20.ckpt |
| baseline_soft | 52 | yes | `.../baseline_soft_s52/best-24.ckpt` | best | — | best-24.ckpt |
| baseline_soft | 62 | yes | `.../baseline_soft_s62/best-29.ckpt` | best | — | best-29.ckpt |
| baseline_soft | 72 | yes | `.../baseline_soft_s72/best-26.ckpt` | best | — | best-26.ckpt |
| baseline_soft | 82 | yes | `.../baseline_soft_s82/best-17.ckpt` | best | — | best-17.ckpt |
| effb3_control | 42 | yes | `checkpoints/effb3_single/effb3_single_s42/best-39.ckpt` | best | last.ckpt | best-39.ckpt |
| effb3_control | 52 | yes | `.../effb3_single_s52/best-30.ckpt` | best | last.ckpt | best-30.ckpt |
| effb3_control | 62 | yes | `.../effb3_single_s62/best-29.ckpt` | best | last.ckpt | best-29.ckpt |
| effb3_control | 72 | yes | `.../effb3_single_s72/best-39.ckpt` | best | last.ckpt | best-39.ckpt |
| effb3_control | 82 | yes | `.../effb3_single_s82/best-25.ckpt` | best | last.ckpt | best-25.ckpt |
| runA_grl | 42 | yes | `checkpoints/csg_lite/runA_grl_s42/best-38.ckpt` | best | last.ckpt | best-38.ckpt |
| runA_grl | 52 | yes | `.../runA_grl_s52/best-37.ckpt` | best | last.ckpt | best-37.ckpt |
| runA_grl | 62 | yes | `.../runA_grl_s62/best-34.ckpt` | best | last.ckpt | best-34.ckpt |
| runA_grl | 72 | yes | `.../runA_grl_s72/best-35.ckpt` | best | last.ckpt | best-35.ckpt |
| runA_grl | 82 | yes | `.../runA_grl_s82/best-23.ckpt` | best | last.ckpt | best-23.ckpt |
| runB_orth1 | 42 | yes | `checkpoints/csg_lite/runB_orth1_s42/best-36.ckpt` | best | best-37.ckpt, last.ckpt, last-v1.ckpt | **last-v1.ckpt** (policy disagreement) |
| runB_orth1 | 52 | yes | `.../runB_orth1_s52/best-32.ckpt` | best | last.ckpt | **last.ckpt** |
| runB_orth1 | 62 | yes | `.../runB_orth1_s62/best-34.ckpt` | best | last.ckpt | **last.ckpt** |
| runB_orth1 | 72 | yes | `.../runB_orth1_s72/best-35.ckpt` | best | last.ckpt | best-35.ckpt |
| runB_orth1 | 82 | yes | `.../runB_orth1_s82/best-37.ckpt` | best | last.ckpt | best-37.ckpt |
| runB (orth=5) | 42 | yes | `checkpoints/csg_lite/runB_s42/best-39.ckpt` | best | best-37, best-31, last* | last-v2.ckpt |
| runB (orth=5) | 52 | yes | `.../runB_s52/best-28.ckpt` | best | best-01, best-34, last* | last-v2.ckpt |
| runB (orth=5) | 62 | yes | `.../runB_s62/best-39.ckpt` | best | best-28, last* | last-v1.ckpt |
| runB (orth=5) | 72 | **NO** | — | MISSING | — | — |
| runB (orth=5) | 82 | **NO** | — | MISSING | — | — |

Notes:

- Phase 1's 4-method grid is `baseline_soft`, `effb3_control`, `runA_grl`, `runB_orth1` — all 5 seeds present.
- `runB` (`λ_orth=5.0`) is inventoried only. It was never part of the n=5 CBM tables.
- For `runB_orth1_s42`, `best-36.ckpt` (2026-04-24 04:24) is newer than `best-37.ckpt` (2026-04-23 22:13). The later training run's val-acc winner is epoch 36. `leakage.json` used `last-v1.ckpt` from that same later run (same timestamp as `best-36`). Published leakage for s42 is therefore last-epoch, not best-val. Paper B uses `best-36.ckpt`.
- Orphan ckpts at `checkpoints/baseline/best-*.ckpt` and `checkpoints/csg_lite/best-*.ckpt` (no seed dir) are pre-benchmark exploratory runs and are ignored.

---

## 0.2 Train / eval transform parity — CRITICAL

### What `x_lesion=None` does

```124:126:src/models/csg_lite.py
    def forward(self, x_context, x_lesion=None, return_latents=False):
        if x_lesion is None:
            x_lesion = self._rgb_to_gray3(x_context)
```

`_rgb_to_gray3` is a **post-normalisation** linear combo on the tensor already in ImageNet-normalised RGB:

`gray = 0.2989 R + 0.5870 G + 0.1140 B`, then repeated to 3 channels.

### What training actually feeds each branch

`CombinedTrainDataset` *does* build a separate lesion tensor via `build_lesion_branch_transform_gray` (Resize 256 → CenterCrop 224 → **Grayscale(3)** → ToTensor → ImageNet norm). That is the METHODS-documented path.

`CSGLiteLightning.training_step` then **throws it away**:

```106:116:src/models/csg_lightning.py
        # Use the same lesion path in train/val: derive lesion input inside model from images_ctx.
        # This avoids train/val mismatch (pre-normalization grayscale vs post-normalization conversion).
        ...
        images_lesion = None
        ...
        y_hat, d_ctx, d_adv, z_l, z_c = self.model(images_ctx, x_lesion=images_lesion, return_latents=True)
```

So during training:

| Branch | Tensor | Transform |
|---|---|---|
| context | `images_ctx` | `build_train_transform_robust` (colour jitter, flips, RandomErasing, ImageNet norm) |
| lesion | `_rgb_to_gray3(images_ctx)` | same colour-augmented, **already-normalised** RGB, grayed in-model |

### What eval does

`eval_ood_benchmarks.py`, `check_leakage.py`, `eval_ood_scores.py`, and `CSGLiteLightning.validation_step` all call `model(images)` / `model(images, x_lesion=None)`. `images` come from `build_val_transform_robust` (Resize 256 → CenterCrop 224 → ToTensor → ImageNet norm). The lesion branch then gets `_rgb_to_gray3(images)`.

### Verdict

**Train and eval are matched on the grayscale conversion:** both apply `_rgb_to_gray3` *after* ImageNet normalisation. They do **not** route a raw colour tensor to the lesion branch, and they do **not** use the pre-norm `Grayscale` transform at either train or eval.

The residual train/eval difference is ordinary augmentation (jitter / crop / erase on train vs center-crop on eval). That is not the bug the work order described.

**No Phase 0.2 re-eval. `results/paperB/phase0_reeval/` is left empty on purpose.**

### Related bugs that are *not* this gate

1. **`build_lesion_branch_transform_gray` is dead code at train time.** The dataloader still spends the work; `training_step` ignores the result. Documentation that cites this transform as the trained lesion path is wrong.
2. **`train_csg.py` post-fit Mahalanobis** uses `collect_z_lesion_labels_csg` on the *paired train loader*. That helper takes `images_lesion` (pre-norm gray) and then `encode_z_lesion` applies `_rgb_to_gray3` again — the documented double-conversion. Published OOD numbers in `results/cbm_revision/ood_comparison.csv` come from `eval_ood_benchmarks.py`, which uses the single-conversion eval path. Paper B will not reuse the `train_csg.py` post-fit AUROCs.
3. **`check_leakage.py` `backbone_raw` for CSG is the *context* backbone**, not the lesion backbone (`extract_context_features`). Flagged; Phase 1 extracts both.

---

## 0.3 Domain / label entanglement

PAD-UFES has **no DF and no VASC** (0 / 239 and 0 / 253 in the full master CSV). On the leakage-probe set (ISIC test + all PAD, n=7365, 5067 / 2298, majority **0.6880**) the label × domain table is:

| label | ISIC test | PAD |
|---|---:|---:|
| AK | 173 | 730 |
| BCC | 665 | 845 |
| BKL | 525 | 235 |
| DF | 48 | 0 |
| MEL | 904 | 52 |
| NV | 2575 | 244 |
| SCC | 126 | 192 |
| VASC | 51 | 0 |

Protocol: one-hot ground-truth label → logistic regression → domain, 70/30 stratified split, 5 seeds (42/52/62/72/82), same as `check_leakage.py`.

| Estimator | Accuracy | Balanced acc | AUROC |
|---|---|---|---|
| Majority (always ISIC) | 0.6880 | — | — |
| **Logistic on one-hot label** | **0.7968 ± 0.0056** | 0.7884 ± 0.0071 | 0.8424 ± 0.0076 |
| In-sample majority-per-label (Bayes) | 0.7970 | — | — |

**0.797 is the upper bound on how much of a domain probe is just label–domain correlation.** Context:

- ResNet-50 leakage 0.979 ≫ 0.797 → most of the baseline's probe accuracy is true acquisition-domain signal, not labels.
- EffNet-B3 single-encoder leakage **0.8017 ≈ 0.797** → the backbone-control "leakage" is statistically indistinguishable from the label-only ceiling. A large part of the 0.979 → 0.802 drop is backbone + the fact that a 16-d class code cannot exceed the label–domain map.
- CSG `z_lesion` leakage **0.72 < 0.797** → below the label-only ceiling, which is possible because ID balanced accuracy is only ~0.70 (the representation is not a perfect class code). Phase 5's per-class conditional probe is required before anyone treats 0.72 as residual domain leakage.

Raw JSON: `phase0_label_domain_probe.json`.

---

## 0.4 Split hygiene

**Question:** are the PAD images used in the adversarial/context branches during training the same images later used as the OOD set?

**Answer: YES.**

Code path:

1. `SkinDataModule.setup` (`src/datasets/skin_dataset.py`):
   - `self.pad_df_all = pad_df` — **all** `domain == "pad_ufes"` rows (2,298).
   - If `csg_lite_train=True`: `CombinedTrainDataset(isic_train, self.pad_df_all, ...)`.
2. `splits.build_csg_train_dataframe`: concatenates ISIC train with **all** PAD (`label_idx = -1`, `domain_idx = 1`).
3. `splits.build_id_ood_test_dataloaders`: `ood_test_ds = SkinDataset(pad_df)` — again **all** PAD rows. ID test is the held-out ISIC 20% (`random_state=42`, stratify `label_idx`).

ISIC split (recomputed, matches `leakage.json` counts): train 16,211 / val 4,053 / test 5,067. Split seed is **fixed at 42** in `SplitConfig` and is not tied to `--seed`. Per-seed runs vary init and shuffle only.

PAD labels are never used for `L_cls` (`ignore_index=-1`). They **are** used for `L_ctx` and `L_adv`. Every OOD test image has been seen ~`16211/2298 ≈ 7` times per epoch × 40 epochs in the context and adversarial objectives. This is the mechanism behind P2.

ISIC metadata has `lesion_id` (`ISIC_2019_Training_Metadata.csv`). PAD `patient_id` / `lesion_id` / `fitspatrick` live in `pad_ufes20/metadata.csv`, which is a **dangling symlink** (`/mnt/data2/Vinh/Ban_sao_datn/...`). The file exists at `/data2/hpcshared/Vinh/Ban_sao_datn/data/raw/pad_ufes20/metadata.csv` but is mode `600` / owner `anhnv` — unreadable by this user. Phase 2/7/9 will need that file (or a copy). Not a Phase 0 blocker.

---

## 0.5 Environment

See `ENVIRONMENT.md`. Scheduler confirmed: **SLURM**, partition `defq`. Login node has no GPU driver. Original training: 1× A100-SXM4-80GB, torch 2.11.0+cu130, lightning 2.6.1. CSG wall time ≈ 2 min/epoch, 1.3–1.5 h / 40-epoch run.

---

## Deviation from the work order (Phase 0)

1. **Cannot write `results/paperB/` inside the repo.** `CSG-Skin/` is `anhnv:toandq` mode 755. Outputs are under `/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/`. Existing `results/` is untouched.
2. **No `phase0_reeval/` numbers.** Gate 0.2 passed; recomputing published tables would be a different experiment (and would need GPU).
3. PAD official metadata is currently unreadable (permissions). Split-hygiene answer is from code + `master_metadata_lesion_only_soft.csv`, which is sufficient for 0.4.

---

## Gate decision

Phase 0.2 is **not** a train/eval mismatch. **Proceed to Phase 1** (inference only, existing checkpoints, no retraining). Do not start Phase 2 until P1 is scored.
