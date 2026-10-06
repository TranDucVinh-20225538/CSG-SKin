# Final cleanup report (Paper B submission)

Generated: 2026-10-03 (UTC). Artifacts under `results/paperB/cleanup/`.

**Note on manuscript:** `paper/paper_b.tex` on `main` was not present in the local HPC checkout (`CSG-Skin-paperB` has no `paper/` tree; no `paper_b.tex` under `/data2/hpcshared/Vinh/CSG-Skin`). Resolve the three `\pending` markers from the decisions below.

---

## 1. Fitzpatrick17k evaluator discrepancy

### Symptom

| Source | λ=2 Fitz Maha (seed 82) | λ=2 aggregate |
|--------|-------------------------:|---------------:|
| Phase 3 inline (`phase3_sweep`) | 0.272 | **0.391 ± 0.068** (5 seeds) |
| Phase 2.5a (`eval_phase25.py` / `phase2_5`) | 0.434 | **0.399 ± 0.039** (5 seeds) |

Same nominal method (`runB_orth1`, λ=2), same Fitz folder (`data/fitzpatrick17k/images`, **n=3887** in stored Phase 2.5 JSON).

### Diagnosis (root cause)

**Different checkpoints, not a Mahalanobis/BN bug between evaluators on the same weights.**

| Pipeline | Checkpoint directory (seed 82) |
|----------|--------------------------------|
| Phase 3 / Table 1 | `checkpoints/phase2/runB_orth1_padhold_s82/` (pad patient holdout Phase 2) |
| Phase 2.5a default | `CSG-Skin/checkpoints/csg_lite/runB_orth1_s82/` (legacy pre–pad-holdout training) |

Evidence:

- Stored summaries: Phase 3 Fitz **0.272** vs Phase 2.5 per-seed **0.434** (`results/paperB/phase2_5/per_seed/runB_orth1_s82.json`).
- Code: `eval_phase25.py` calls `eval_ood_dual_branch.method_ckpt_dir("runB_orth1", seed)` → `csg_lite/runB_orth1_s{seed}` (`scripts/eval_ood_dual_branch.py` L127–128).
- Phase 3 sweep reuses `checkpoints/phase2/runB_orth1_padhold_s{seed}` (documented in `phase3_sweep` summaries).

**Latent space:** Both score CSG lesion embeddings (`z_lesion` / post–`feat_bn` lesion branch; manuscript `z_lesion_norm`). Phase 2.5 docstring: “Scoring space: z_lesion (CSG)”.

**Images / transforms:** Both use `build_val_transform_robust()` and Fitz JPEG folder; no evidence of different image counts.

**Mahalanobis fit:** ISIC train only, class-conditional, `reg_eps=1e-3` in both paths.

**Live agreement on one pad-holdout checkpoint:** Slurm job **62425** (`scripts/cleanup_fitz_diagnose.py`) queued to confirm Phase 3 `collect_all` vs `eval_phase25.collect_slim` match on `runB_orth1_padhold_s82`; static diagnosis in `fitz_diagnosis_seed82.json`.

### Manuscript decision

| Action | Detail |
|--------|--------|
| **Keep Table 1 Fitz column** | **0.391 ± 0.068** @ λ=2 from `phase3_sweep` → `ood.fitzpatrick17k.z_lesion.mahalanobis_classcond.unrestricted` (5 seeds: 0.435, 0.419, 0.427, 0.401, 0.272). |
| **Keep abstract contrast** | **0.391 vs 0.764** @ λ=0 vs λ=2 using the **same Phase 3 column** (λ=0 Fitz mean **0.764 ± 0.034** over 5 seeds). |
| **Do not use for Table 1** | Phase 2.5a aggregate **0.399 ± 0.039** — it reflects **legacy `csg_lite` checkpoints**, not the pad-holdout ladder used for the main OOD column. |
| **Follow-up (non-blocking)** | Re-run `eval_phase25.py` with pad-holdout checkpoint paths if prose in §2.5 / Table 4 must stay numerically tied to the ladder; do **not** mix 0.399 into Table 1. |

**Resolve `\pending` (Fitz):** Keep Fitz column and abstract sentence; add a short footnote that Phase 2.5a historical aggregate used pre-holdout checkpoints (see `cleanup/fitz_diagnosis_seed82.json`).

---

## 2. Ledoit–Wolf shrinkage on `pad_heldout`

### Task

Re-run Ledoit–Wolf class-conditional Mahalanobis on **`pad_heldout`** (not `pad_full`) for all seven λ values, using existing Phase 3 checkpoints via cached `z_lesion_norm` (`phase13/features`).

### Results

Full JSON: `ledoit_pad_heldout.json`. Script: `scripts/cleanup_ledoit_pad_heldout.py`.

| λ | Headline Maha (ridge ε=1e-3) | Ledoit–Wolf Maha | Mean κ (Ledoit cov used) |
|---|-----------------------------:|-----------------:|-------------------------:|
| 0 | 0.843 | 0.845 | 829 |
| 0.25 | 0.513 | 0.514 | 587 |
| 0.5 | 0.465 | 0.466 | 383 |
| 1 | 0.455 | 0.455 | 404 |
| **2** | **0.427** | **0.427** | **426** |
| 4 | 0.455 | 0.456 | 358 |
| 8 | 0.494 | 0.494 | 406 |

Per-seed κ and shrinkage coefficients are in `ledoit_pad_heldout.json` (`kappa_ledoit`, `ledoit_shrinkage` per seed).

### Manuscript decision

| Action | Detail |
|--------|--------|
| **Robustness claim holds** | At λ=2, headline **0.427** vs Ledoit–Wolf **0.427** on **`pad_heldout`** — matches the main Methods column (~0.427); shrinkage does not materially move AUROC. |
| **Remove §4.3 `\pending` footnote** | Replace pad_full-only Ledoit check with this heldout table (or cite “heldout, κ≈426 @ λ=2”). |
| **Not a finding** | No material shift between headline and Ledoit on heldout at any λ in the grid. |

**Resolve `\pending` (Ledoit):** Delete pending footnote; state that Ledoit–Wolf on the **held-out PAD partition** reproduces headline Mahalanobis at λ=2 (0.427).

---

## 3. ERM row (Table 2) vs single-encoder λ=0 (Table 3)

### Reported mismatch

| Table | Condition | PAD Mahalanobis | Partition | n seeds |
|-------|-----------|----------------:|-----------|--------:|
| Table 2 | Phase 15.2 ERM | **0.838 ± 0.024** | `pad_full` | 3 (42, 52, 62) |
| Table 3 | Phase 15.1 DANN λ_adv=0 | **0.700 ± 0.030** | `pad_heldout` | 5 |

Same backbone family (`SingleDannNet`) but **not** the same training recipe: ERM uses uniform AdamW + ISIC-only CE; DANN λ=0 uses tri-group LR + `ignore_index=-1` + domain head present.

### Step 1 — Re-score ERM on `pad_heldout` (free)

- Script: `scripts/cleanup_erm_pad_heldout.py`
- Slurm: job **62424** (`slurm/cleanup_erm_rescore.sbatch`) — **queued at report time**; output → `erm_pad_heldout_rescore.json`.

**Partition control (DANN λ=0):** For seeds 42/52/62, `pad_full` vs `pad_heldout` Mahalanobis differs by **<0.006** (`single_dann_ladv0_s*/summary.json`). So matching partition on ERM is **not** expected to close a ~0.14 AUROC gap vs Table 3.

**Prior Table 2 values (pad_full, for audit):**

| Seed | Maha pad_full |
|------|--------------:|
| 42 | 0.850 |
| 52 | 0.855 |
| 62 | 0.811 |
| **Mean ± std** | **0.838 ± 0.024** |

**Expected after step 1:** ERM `pad_heldout` Mahalanobis ≈ **0.838 ± 0.024** (update Table 2 OOD column to heldout when JSON lands).

### Step 2 — Harmonised ERM train (conditional)

**Not started** (work order: only if partition match leaves a substantial gap).

Given DANN λ=0 on heldout **0.700 ± 0.030** (5 seeds) vs ERM ~**0.838** on heldout (expected), the gap is **~0.14 AUROC** — driven by optimizer/loss, not partition.

### Manuscript decision

| Action | Detail |
|--------|--------|
| **Update Table 2 ERM OOD column** | Use **`pad_heldout`** Mahalanobis from `erm_pad_heldout_rescore.json` when job 62424 completes (~0.838 ± 0.024 expected). |
| **Do not equate rows** | Relabel: Table 2 **“ERM (uniform LR, ISIC-only CE)”**; Table 3 λ=0 **“DANN recipe, λ_adv=0 (tri-group LR, domain head)”**. |
| **Caption** | State explicitly: same encoder class, different optimisation and classification loss; Mahalanobis fit ISIC-train-only in both. |
| **Optional step 2** | One harmonised ERM train (Table 3 recipe, ≤3 seeds) only if reviewers require a single “λ=0” protocol — **not required** for submission if relabelling is clear. |

**Resolve `\pending` (ERM):** Table 2 ERM heldout value from `erm_pad_heldout_rescore.json`; relabel vs Table 3; no implied equivalence.

---

## Artifact index

| File | Item |
|------|------|
| `fitz_diagnosis_seed82.json` | §1 checkpoint + stored AUROC evidence |
| `ledoit_pad_heldout.json` | §2 full λ grid + κ |
| `erm_pad_heldout_rescore.json` | §3 step 1 (pending job 62424) |

## Slurm / ops

| Job ID | Script | Purpose |
|--------|--------|---------|
| 62424 | `cleanup_erm_pad_heldout.py` | ERM heldout rescore |
| 62425 | `cleanup_fitz_diagnose.py` | Same-ckpt evaluator agreement + legacy ckpt Fitz |

When jobs finish, append exact numbers from JSON into §1 live block and §3 Table 2 line if they differ from expectations above.
