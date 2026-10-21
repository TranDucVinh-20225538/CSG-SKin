# Paper B R6 — split stability: does the Mahalanobis inversion depend on the ISIC lesion split?

Written and committed **2026-10-21 13:26 (UTC+7)**, **before any split-stability training, feature extraction or
scoring run**. Fixed now; not changed after results are seen. Deviations (if any) are appended at the end with timestamp
(UTC+7) and reason. Does **not** edit `paper/midl/paper_b.tex`.

Branch for this protocol: `paperb/split-stability` (forked from `origin/paperb/writing-r6-fixes`). Step 1 of the work
order is this decision only; Step 2 (18 GPU runs + `REPORT.md`) starts only after explicit user approval of this commit.

## Question

The Table 1 / Table 2 inversion (Mahalanobis AUROC on `pad_heldout` and clinical sets falling below 0.5 at λ = 2 while
ID balanced accuracy stays intact) is measured on one lesion-level ISIC split produced by
`isic_splits_by_lesion()` in `scripts/train_r2_item4_lesion_split.py`: two nested
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` calls (test fold, then val fold). Does that inversion
depend on this particular split, or does it appear under other folds of the same splitter?

## Design (fixed)

### Code change (to implement only after approval; behaviour locked here)

- Add CLI / API parameter **`--isic_split_seed`** (default **42**) to the lesion-split path used by the Table 1 dual
  encoder (`isic_splits_by_lesion` and callers that wire `p2.isic_splits`).
- Pass the **same** integer into **both** `StratifiedGroupKFold(..., random_state=isic_split_seed)` steps (test split and
  val split). Default 42 reproduces today's published split.
- **PAD patient split stays fixed:** `split_seed=42` in `train_phase2_pad_holdout` / Phase 2 split JSON
  (`pad_adv` / `pad_heldout`). Training seed and `--isic_split_seed` do not retarget PAD.
- **Regression test (CPU, no GPU):** for `--isic_split_seed 42`, hash (sha256 of sorted image path lists, or an
  equivalent canonical listing of train / val / test image stems) must match the frozen reference from the current
  Table 1 split (`results/paperB/r2/item4/split_record.json` and the path lists implied by that recipe). The test fails
  if seed 42 drifts. New seeds write their own `split_record.json` under the split-stability tree (below).

### Splits under study (fixed from now)

| `isic_split_seed` | role |
|---:|---|
| 42 | reference / regression only (not re-trained in the 18-run grid unless a control is added later) |
| **7** | new split A |
| **123** | new split B |
| **2027** | new split C |

Only seeds **{7, 123, 2027}** enter the GO / NO-GO grid below.

### Training grid (18 runs)

- **Recipe:** main / Table 1 dual EfficientNet-B3 lesion-level path (same as R2 item 4 / manuscript Table 1): two-stream
  DANN, λ_ctx = λ_orth = 1, 40 epochs, same LR / adversary / GRL ramp, **class-mix shift 100%** (ISIC's own mix for the
  adversary), checkpoint selection as Table 1; **primary numbers at `last.ckpt`** (consistent with the writing-r6 /
  t1last reporting path for the inversion).
- **λ ∈ {0, 2}**, training seeds **{42, 52, 62}**.
- **3 splits × 2 λ × 3 train seeds = 18 runs.**
- Everything else identical to Table 1 except the ISIC fold seed.

### Degraded-run rule (R1, unchanged)

As in `decisions/paperB_r6_r1.md` / `scripts/r6_common.py`: chance = 1/8; a run is **degraded** if primary-checkpoint
ID test balanced accuracy **&lt; 0.25**. Degraded runs are **reported and flagged**, **excluded from means / CIs /
GO–NO-GO aggregates**, and **not re-run**, unless a later R1-style replacement rule is explicitly amended here before
any run (none is registered now).

### Metrics and bootstrap (same as Table 1 / 2)

Per cell (`isic_split_seed` × λ), over non-degraded train seeds:

- ID balanced accuracy (ISIC test)
- Leakage: **held-out linear probe** (Table 1 / P6 V1 style: ISIC test ∪ `pad_heldout`, not the adversary's `pad_adv`
  selection probe)
- Mahalanobis class-conditional AUROC on **`pad_heldout`**, **Fitzpatrick17k**, **MED-NODE**, **SD-198**

Report **mean ± s.d.** across train seeds and **pooled percentile bootstrap CIs** exactly as
`results/paperB/PREREGISTER_BOOTSTRAP.json` (2,000 draws, rng seed 20261005, percentile 95%, draws shared across seeds
within a cell; ID by lesion group; `pad_heldout` by patient; Fitzpatrick / MED-NODE / SD-198 by the same group rules as
R5/R6 reports). Rule A may be reported for description; **GO / NO-GO below do not use rule A** — they use the
pre-registered mean thresholds only.

### Outputs (Step 2)

`results/paperB/r6/split_stability/`:

- `splits/seed_<S>/split_record.json` (+ frozen path-list hashes)
- `runs/ss_s<S>_ladv<λ>_s<seed>/{config.json, summary.json, …}`
- `summary.csv`, `cells.json`, `BOOTSTRAP.json`
- **`REPORT.md`**: tables by split (λ = 0 vs λ = 2), mean ± s.d. and CIs, degraded flags, and a single verdict
  **GO / NO-GO / INCONCLUSIVE** under the criteria below.

Checkpoints under `checkpoints/r6_split_stability/` (or equivalent); features gitignored as usual.

## Pre-registered decision criteria (verbatim; do not change after seeing results)

**GO (ổn định):** ở cả 3 split, AUROC Mahalanobis trên pad_heldout tại λ=2 có mean &lt; 0.5 VÀ thấp hơn λ=0 cùng split;
và ở ≥ 2/3 split, ít nhất 2/3 tập lâm sàng (Fitz, MED-NODE, SD-198) có mean &lt; 0.5.

**NO-GO:** có ≥ 1 split mà pad_heldout λ=2 có mean ≥ 0.5.

**Còn lại: INCONCLUSIVE.**

Means are over non-degraded train seeds within that (`isic_split_seed`, λ) cell. If a cell has zero non-degraded seeds,
that cell cannot support GO; the report flags it and the overall verdict follows the rules above (typically
INCONCLUSIVE or NO-GO if the λ = 2 pad mean is undefined / not &lt; 0.5).

If criteria, thresholds, or the seed set must change after results exist, append a **Deviations** section with
UTC+7 timestamp and reason; do not silently edit this section.

## Predicted outcome (written now, not binding)

If the inversion is driven by label shift under marginal alignment rather than by a lucky ISIC fold, all three new
splits should show pad_heldout Mahalanobis mean &lt; 0.5 at λ = 2 and below the matched λ = 0 cell (GO or at worst
INCONCLUSIVE if clinical sets are mixed). A single split with pad λ = 2 mean ≥ 0.5 is NO-GO and would force a
manuscript caveat that the published fold is not representative.

## GPU-hour estimate and basis

**Basis (measured, Table 1–class dual encoder):** `decisions/paperB_r5_p8.md` records **≈ 1.6 GPU-h per lesion-level
dual-encoder run** (40 epochs, two-stream, including evaluation; wall 83–110 min). The same figure is reused in
`decisions/paperB_r6_r1.md`. Cross-check on later dual runs with `gpu_hours` in summary
(`results/paperB/r5/p11/runs/dual_effb3_*`, λ_adv = 0, 40 epochs): mean ≈ 1.2 GPU-h (no adversary); GRL λ &gt; 0
Table 1 path remains the 1.6 GPU-h figure above.

| item | count | GPU-h |
|---|---:|---:|
| Train + eval (40 ep, two-stream) | 18 | 18 × 1.6 = **28.8** |
| Cap (1.5× first-run re-estimate buffer) | | **≤ 35** |

CPU: split hashes + report bootstrap ≈ 1–2 CPU-h. No GPU in Step 1.

## Stop rules

- Cap **35 GPU-h**. If the first completed run exceeds 1.5× 1.6 GPU-h, re-estimate before launching the rest.
- A run that fails twice is reported with its log and not retried silently.
- No change to λ grid, train seeds, split seeds, or GO/NO-GO thresholds after any cell is scored, except via Deviations.

## Step gating

1. **This commit (pre-run):** decision only on `paperb/split-stability`. Stop for user approval.
2. **After approval:** implement `--isic_split_seed` + seed-42 hash test; launch 18 runs; write `REPORT.md` with
   tables and the pre-registered verdict. No `paper_b.tex` edits in that step.

## Deviations

- **2026-10-22 01:06 (UTC+7).** Slurm report job `69433` (`paperB-ss-rep`, after array `69432`) failed in
  `load_run` with `AttributeError: module 'r5_common' has no attribute 'DEGRADED_ID_BAL'`. The degraded threshold is
  the R1 rule (ID bal acc &lt; 0.25 = 2× chance); it lives on `r6_common`, not `r5_common`. **Change:**
  `scripts/r6_split_stability.py` now compares against the literal `0.25` (same threshold; no criterion change).
  `REPORT.md` / `VERDICT.json` / `summary.csv` / `cells.json` were regenerated locally on the login node from the
  completed 18 runs (verdict **GO**). No GPU re-run; no edit to GO/NO-GO thresholds, splits, λ grid, or train seeds.
