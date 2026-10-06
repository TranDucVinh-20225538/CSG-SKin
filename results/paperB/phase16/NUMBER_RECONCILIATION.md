# Phase 16 — Number reconciliation (Part A)

Generated from code inspection and on-disk `summary.json` / aggregate files. No retraining was performed.  
`paper/paper_b.tex` was **not present** in the Paper B repo or under `/data2/hpcshared/Vinh` at audit time; provenance for A4 uses `results/paperB/MASTER_REPORT.md` (writing package aligned with the manuscript) plus phase scripts. Re-run A4 against the `.tex` once that file is on disk.

Artifacts:
- `phase3_ood_partitions.csv` — per-run PAD/Fitz Mahalanobis for Phase 3
- `table1_ood_pad_dual_column.json` — Table 1 PAD OOD column, both partitions
- `PREREGISTER_B.json` — Part B locked before any BCN↔HAM training

---

## A1 — Phase 3 PAD partition (Table 1 vs Table 4 / tails)

### What the code does

`scripts/train_phase3_sweep.py` loads the Phase 2 patient split (`pad_adv` / `pad_heldout`), trains (or reuses Phase 2 ckpt at λ=2), then **scores four OOD packs in one pass**:

| Key in `summary.json` | Images | Role |
|---|---:|---|
| `pad_adv` | 1,582 | Seen in L_ctx / L_adv during training |
| `pad_heldout` | 716 | Never in training branches (412 patients) |
| `pad_full` | 2,298 | adv + heldout (includes images the model saw) |
| `fitzpatrick17k` | 3,887 | External OOD |

Mahalanobis for the main-text lesion branch is computed in `detector_block()` with **class-conditional Mahalanobis on `z_lesion`**, statistics fit on **ISIC train only** (`train_pack`), ID = ISIC test, OOD = each pack above (lines 321–373).

### Which partition the manuscript used

| Claim / location | Value at λ=2 | Partition | Source file |
|---|---:|---|---|
| Table 1 / dial “OOD AUROC PAD” | **0.413 ± 0.021** | **`pad_full`** | `MASTER_REPORT.md` Table 1; aggregate `pad_full` mean **0.4128 ± 0.0214** from `phase3_sweep/*/summary.json` |
| Table 4 / Phase 2 / tail headline | **0.427 ± 0.025** | **`pad_heldout`** | `phase2_pad_holdout/PHASE2_REPORT.md`; Phase 3 heldout aggregate **0.4265 ± 0.0247** |
| Phase 14 tail (λ=2) | 0.427 → 0.412 after p90 | **`pad_heldout`** | `phase14/PHASE14_REPORT.md` (Phase 13 feature cache, split `pad_heldout`) |

**Conclusion:** Phase 3 **did score both partitions**; the internal contradiction is **which column was pasted into Table 1**, not missing heldout eval. Table 1’s PAD OOD column matches **`pad_full`** exactly (e.g. λ=2: 0.413 vs 0.4128). That set **includes `pad_adv`**, which violates the Methods claim that main OOD numbers use **`pad_heldout` only**. `MASTER_REPORT.md` line 105 even states Table 1 PAD OOD is “Phase 3 `pad_full` … matching the locked 0.86 → 0.51 number,” while the dial prose above it says transfer is heldout-only.

### Manuscript should carry (PAD OOD, λ grid)

Use **`pad_heldout`** for all main-text PAD Mahalanobis AUROC. No retraining: values already in each `results/paperB/phase3_sweep/runB_orth1_ladv*_s*/summary.json` → `ood.pad_heldout.z_lesion.mahalanobis_classcond.unrestricted`.

**Dual column (z_lesion Maha, unrestricted):**

| λ_adv | n | pad_full (legacy Table 1) | pad_heldout (Methods-aligned) |
|---:|---:|---|---|
| 0 | 5 | 0.863 ± 0.025 | 0.843 ± 0.023 |
| 0.25 | 3 | 0.508 ± 0.022 | 0.513 ± 0.018 |
| 0.5 | 3 | 0.449 ± 0.008 | 0.465 ± 0.015 |
| 1 | 3 | 0.439 ± 0.040 | 0.455 ± 0.036 |
| 2 | 5 | **0.413 ± 0.021** | **0.427 ± 0.025** |
| 4 | 3 | 0.443 ± 0.038 | 0.455 ± 0.039 |
| 8 | 5 | 0.481 ± 0.026 | 0.494 ± 0.027 |

**One sentence for the paper:** At λ=2, unrestricted lesion Mahalanobis AUROC is **0.413 on the full PAD pool (2,298 images, including adversarial-training PAD)** versus **0.427 on the held-out PAD patients (716 images)**; Table 1 used the former while tail analyses and Phase 2 reporting used the latter.

**Cliff contrast (λ=0 vs 0.25):** Recompute bootstrap/Holm using **heldout** column: OOD drop ≈ 0.843 → 0.513 (Δ ≈ 0.33), still a cliff; exact CIs should be regenerated from `phase3_ood_partitions.csv` (not done in this pass).

---

## A2 — Fitzpatrick17k: 0.391 vs 0.399

### Where each number lives

| Location | λ=2 Fitz Maha | n seeds | Provenance |
|---|---:|---:|---|
| Table 1 / Phase 3 sweep | **0.391 ± 0.068** | 5 | `phase3_sweep` → `ood.fitzpatrick17k.z_lesion.mahalanobis_classcond.unrestricted` |
| Abstract / Table 4 / Phase 2.5a | **0.399 ± 0.039** | 5 | `phase2_5/phase25_aggregate.json` → `methods.runB_orth1.fitz_maha` |

Same checkpoint family at λ=2 (Phase 3 reuses `checkpoints/phase2/runB_orth1_padhold_s{seed}/`). Same Fitz folder (`data/fitzpatrick17k/images`, **n=3887** in both pipelines).

### What differs

1. **Evaluator code path:** Phase 3 uses `train_phase3_sweep.detector_block()` inline; Phase 2.5a uses `eval_phase25.py` → `detectors_from_packs()` (shared with dual-branch tooling, extra metrics, slightly different Mahalanobis wiring for 16-d `z_lesion`).
2. **Per-seed disagreement on identical ckpt** (same seed, both claim λ=2 Phase 2 ckpt):

   | seed | Phase 3 Fitz Maha | Phase 2.5a Fitz Maha |
   |---:|---:|---:|
   | 42 | 0.435 | 0.407 |
   | 52 | 0.419 | 0.344 |
   | 62 | 0.427 | 0.434 |
   | 72 | 0.401 | 0.377 |
   | 82 | **0.272** | **0.434** |

   Seed **82** is an outlier in Phase 3 (pulls mean to 0.391); Phase 2.5a stays ~0.43. This is not seed noise — it indicates **non-equivalent eval implementations or a stale/corrupt Phase 3 eval row**, not a different image subset.

3. **Not** explained by: image count (both 3887), PAD partition (Fitz is separate), or λ=2 checkpoint path (both `reused_phase2_ckpt: true` in Phase 3 summaries).

### Canonical protocol (pick one)

**Adopt Phase 2.5a evaluator** (`scripts/eval_phase25.py` on `z_lesion`, ISIC-train fit, unrestricted AUROC) for **all** Fitz numbers, because it was the pre-registered third-domain test and powers Table 4 / structural (~0.41) narrative.

**Manuscript should carry at λ=2:** **0.399 ± 0.039** from `results/paperB/phase2_5/phase25_aggregate.json` (until a unified re-eval job refreshes Phase 3 rows).

**Recompute both places:** Queue inference-only job: load each Phase 3 / Phase 2 checkpoint, run **`eval_phase25.py` Fitz block only**, write `results/paperB/phase16/fitz_unified/*.json`, then replace Table 1 Fitz column and abstract. **Do not** mix Phase 3 inline Fitz with Phase 2.5a abstract.

**Why the other number differed:** Phase 3’s inline Fitz scorer disagrees with the locked Phase 2.5 pipeline (largest at seed 82); the ±0.068 vs ±0.039 spread is driven by that mismatch, not by a different Fitz subset.

---

## A3 — ERM 0.838 vs single-encoder DANN λ=0 at 0.700

### Reported values

| Table | Condition | PAD Mahalanobis | Source |
|---|---|---:|---|
| Table 2 (Phase 15.2) | ERM, objective weight 0 | **0.838 ± 0.024** (n=3) | `phase15/objectives/erm_w0_s{42,52,62}/summary.json` → **`ood_pad_full`** |
| Table 3 (Phase 15.1) | Single-encoder DANN λ_adv=0 | **0.700 ± 0.030** (n=5) | `phase15/single_dann/single_dann_ladv0_s*/summary.json` → **`ood.pad_heldout`** (pad_full ≈ same) |

Aggregates match `PHASE15_1_REPORT.md` / `PHASE15_2_REPORT.md`.

### Line-by-line comparison (same seed 42 example)

| Factor | DANN λ=0 (15.1) | ERM (15.2) |
|---|---|---|
| Backbone | SingleDannNet EffNet-B3 | **Same** `SingleDannNet` |
| Domain head | Instantiated; λ_adv=0 → **no adv gradient** | **Same** module; never in loss |
| Train batches | `CombinedTrainDataset` ISIC+`pad_adv`, paired collate | **Same** |
| Classification loss | CE with `ignore_index=-1` on PAD | **ISIC-only** CE (`loss_cls_isic`) |
| Optimizer | AdamW **3 groups**: backbone 1e-4, classifier **0.2×**, adv head 30× | AdamW **single group** 1e-4 all params |
| BN on features | `feat_bn` sees **mixed ISIC+PAD** batches | **Same** |
| Mahalanobis fit | ISIC train embeddings only | **Same** |
| OOD scored | pad_full / pad_heldout (both ~0.66 at s42) | **pad_full only** in summary (0.85 at s42) |
| Seeds in table | 5 | **3** |

**Batch composition / BN:** Both train with PAD in the forward pass; λ=0 does **not** make DANN identical to ERM.

**Mahalanobis fit:** Both fit on ISIC train only — the 0.14 gap is **not** from mixed train statistics.

**Dominant causes of 0.14 gap:**
1. **Different optimizers** (tri-group LR with 5× slower classifier vs uniform LR) → different representations at the same epoch budget.
2. **Different classification loss masking** (ignore_index vs ISIC-only rows).
3. **Table mismatch:** comparing **heldout-flavoured DANN aggregate** (0.700) to **pad_full ERM** (0.838) — even at seed 42, DANN ~0.66 vs ERM ~0.85 on pad_full.

**Manuscript action (choose one):**
- **Relabel (minimal):** Table 2 row = “ERM (uniform LR, no GRL logging)”; Table 3 λ=0 = “DANN recipe with λ_adv=0 (tri-group LR, domain head present)”; **do not** describe both as the same condition.
- **Harmonise (experiment):** Add ERM with **identical optimizer groups** or DANN λ=0 with **uniform AdamW**, 5 seeds, score **pad_heldout** for both — out of scope for Part A.

**Manuscript should carry until harmonised:** Keep both numbers but **state the protocol difference**; do not imply equivalence.

---

## A4 — Provenance audit (Phase 3 sweep → manuscript)

Legend: **PF** = `pad_full` Mahalanobis in Phase 3 JSON; **PH** = `pad_heldout`; **F3** = Phase 3 Fitz inline; **F25** = Phase 2.5a; **—** = not Phase 3 sweep.

| Manuscript element (MASTER_REPORT / expected tex) | Partition / source | Tag |
|---|---|---|
| Dial / Table 1 — OOD AUROC PAD column | PF | **PF** |
| Dial / Table 1 — OOD AUROC Fitz column | F3 inline | **F3** |
| Dial / Table 1 — 6-class OOD AUROC | PF restricted | **PF** |
| Dial / Table 1 — leakage bal acc | ISIC test + **PF** probe pool | **PF** (probe protocol) |
| Dial / Table 1 — ID bal acc, ECE | ISIC test | — |
| Dial / Table 1 — PAD xfer 6-class | Phase 6 xfer, **PH** | **PH** |
| Dial / Table 1 — z_context OOD | PH (context Maha) | **PH** |
| Cliff § (λ=0 vs 0.25) bootstrap on PAD OOD | PF | **PF** |
| Causal claim “0.508 → 0.413” (λ=0.25→2) | PF | **PF** |
| Phase 14 tail truncation headline (0.427→0.412) | PH (Phase 13 features) | **PH** |
| Phase 13.3 OOD ECE on PAD | PH | **PH** |
| Abstract Fitz “structural ~0.41” | F25 | **F25** |
| Table 4 Fitz + PAD rows | F25 + PH (Phase 2/2.5) | **F25 / PH** |
| Phase 13.2 backbone Mahalanobis dial | Phase 13 extract (PH) | **PH** |
| Figure: λ vs OOD (main dial figure) | PF if sourced from Table 1 | **PF** |
| Cross-domain cost table (xfer) | Phase 6, PH | **PH** |
| ResNet/EffB3 reference rows Table 1 | Phase 4/5 baselines | — |
| Table 2 ERM 0.838 | Phase 15.2 ERM | — |
| Table 3 single-encoder λ sweep | Phase 15.1 | — |

**If Methods assert “all main OOD numbers use pad_heldout”:** every **PF** row above must switch to **PH** (values in `table1_ood_pad_dual_column.json`), and Fitz must move to **F25** or unified re-eval.

---

## Part B status (blocking preregistration done)

- `PREREGISTER_B.json` written **before** any BCN↔HAM training.
- BCN/HAM class counts **verified** against `ISIC_2019_Training_Metadata.csv` (matches work-order table).
- Next GPU steps (not started): B0 ERM + site probe, B0b controls, then B2 sweep per preregistration. iWildCam eval jobs **62264_2/_3** still running; no new WILDS jobs queued.

---

## Recommended immediate edits to `MASTER_REPORT.md` (optional)

1. Replace Table 1 PAD OOD column with **pad_heldout** aggregates; keep pad_full as supplementary sentence or appendix row.
2. Align Fitz column to **Phase 2.5a** or post–phase16 unified eval.
3. Split “λ=0 control” labels between Phase 15.1 DANN and Phase 15.2 ERM.
