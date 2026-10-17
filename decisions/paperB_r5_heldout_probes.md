# Paper B — held-out domain probes (linear + nonlinear) across comparison runs

**Label: CPU analysis.** Committed before any new probe script runs. Does **not** change the
primary leakage variable committed in `94e2399` (pad_full remains primary; held-out is secondary).
Does not change GPU order (R1 → R2).

## Question

On images the adversary never saw (P6 V1 protocol: ISIC-test ∪ pad_heldout, GroupShuffleSplit
70/30 by lesion/patient), how much domain signal remains for (a) a linear probe (as V1) and
(b) nonlinear probes (small MLP; kNN k = 20), for every run used in manuscript comparisons?
Separately: at λ = 0.25 (Table 1, 5 seeds), do seeds whose held-out probe is near chance invert
on OOD Mahalanobis while seeds with residual domain signal do not?

## Scope (runs)

Probe every cell that already has cached `isic_test.npz` + `pad_heldout.npz` (+ `pad_full` for
side-by-side). Families:

| Family | Cells | Feature root (when present) |
|--------|-------|-----------------------------|
| R1 `cur` | shift ∈ {0, 50, 100} × λ ∈ {0.5, 2}, plus λ = 0 where extracted | `r6/r1/cur/features/` |
| Table 1 old recipe | 7 λ × 5 seeds | `r2/item4/features/` |
| A4 (P8 main) | all λ with features | `r5/p8/main/features/` |
| tab:single | single-encoder DANN λ grid | `r5/p8/erm_ref/features/` (and `features_bestval` listed, not primary) |
| tab:obj | CORAL / MMD / IRM / GroupDRO / ERM / GRL rows | `phase15/objectives/<tag>/` — **no cached features today**; frozen extract required before probe (GPU; queue after R1 s82 / with P12 extract pattern). Probe keys staged empty until extract lands. |
| Later | P9, P11, R2 | same protocol when features exist; not blocking this commit |

Representation for comparisons: `z_lesion_norm` (Table 1 space). Other reps may be logged but
do not enter the primary tables.

## Design (code: `scripts/r5_heldout_probes.py`)

For each run, on ISIC-test ∪ pad_heldout (domain labels 0/1), GroupShuffleSplit test_size 0.3,
groups = ISIC lesion id / PAD heldout patient (same as P6 V1). Probe seeds 42/52/62.

1. **Linear (V1):** StandardScaler + LogisticRegression(max_iter=4000) — identical to P6 V1.
2. **MLP:** StandardScaler + MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, early_stopping=True,
   validation_fraction=0.15 of the V1 *train* split, n_iter_no_change=10, random_state=probe_seed).
3. **kNN:** StandardScaler + KNeighborsClassifier(n_neighbors=20, weights="uniform").

Report balanced accuracy (mean ± s.d. over probe seeds; for multi-seed cells also mean ± s.d. over
training seeds). Bootstrap CI of the training-seed mean where n_seeds ≥ 3 (PREREGISTER-style
2,000 draws, rng 20261017), else report probe-seed mean ± s.d. only.

**Side-by-side column:** pad_full linear probe (P6 V0 / Table 1 `leakage.z_lesion` / stored
pad_full bal_acc) so held-out never replaces pad_full in the primary column.

### λ = 0.25 per-seed test (Table 1 only)

For each of the 5 seeds at λ = 0.25, tabulate:
- held-out linear bal_acc, held-out MLP bal_acc
- Mahalanobis AUROC on pad_heldout, Fitzpatrick17k, MED-NODE, SD-198

Spearman ρ (descriptive, n = 5) between held-out linear bal_acc and each Mahalanobis AUROC
(and separately between held-out MLP and each AUROC). No inferential claim at n = 5.

## Rule (2) — primary vs secondary (unchanged)

- **Primary leakage variable:** pad_full probe (committed `94e2399`). Manuscript comparisons that
  already cite pad_full stay on pad_full.
- **Secondary:** held-out V1 (+ MLP / kNN). If held-out paints a different picture (e.g. chance
  held-out while pad_full still high, or the reverse), report **both** and label that contrast a
  **finding made after this commit** — do not silently switch the primary.

## PREDICTED OUTCOME (written now)

### Cross-run held-out probes

- At λ = 0 (old recipe, A4, R1 λ = 0 if present, single-encoder λ = 0): held-out linear ≈ 0.90–0.96,
  within ~0.05 of pad_full; MLP ≈ linear or slightly higher; kNN ≥ linear.
- At λ ≥ 2 on dual-encoder DANN (Table 1 / A4 / R1): held-out linear falls toward 0.55–0.70;
  MLP and kNN stay **above** linear (residual nonlinear domain signal) but still track the λ dose
  downward. pad_full remains the higher / primary number.
- Single-encoder (tab:single): held-out collapse is earlier / deeper than dual-encoder at the same
  λ (matches Table tab:single leakage story); MLP/kNN do not restore a usable domain probe at λ ≥ 2.
- tab:obj (after extract): CORAL/MMD at high weight look like partial invariance on pad_full; held-out
  linear stays within 0.05 of pad_full (same as P6 story). IRM/GroupDRO: held-out ≈ pad_full; ID
  collapse separate.

### λ = 0.25 per-seed (prediction written first)

**Claim to test:** seeds whose held-out probe is near chance **invert** on OOD Mahalanobis
(AUROC ≪ 0.5 on Fitz/MED/SD or pad_heldout); seeds where the domain is still linearly/MLP-decodable
**do not** invert (AUROC ≳ 0.5).

Expected pattern at n = 5:
- Held-out linear bal_acc spans roughly 0.55–0.85 across seeds (heterogeneous collapse at λ = 0.25).
- Spearman(held-out linear, Maha AUROC) on pad_heldout / Fitz / MED / SD-198 is **positive**
  (descriptive), order-of-magnitude ρ ≈ 0.4–0.9 — higher residual domain → higher (less inverted)
  AUROC.
- MLP held-out tracks linear; Spearman with Maha similar sign. If MLP stays high while Maha
  inverts on a seed, that is a post-commit finding (nonlinear remnant ≠ detector survival).

## Status notes (items 3–4, at this commit)

### (3) Fitzpatrick by skin type (commit `40571d5`)

No skin-type section under `results/paperB/r5/p7a/REPORT.md` by design. Report lives at:

`results/paperB/r5/p7a/fitz_skintype/REPORT.md`

(plus `summary.csv` / JSON in that directory). Not missing — separate subdirectory.

### (4) P2 and P4 CPU parts

Already completed before this commit (do not re-run unless REPORT is incomplete):

- P2 (§4.4 lesion-level: confidence, AK recall, ECE): `results/paperB/r5/p2/REPORT.md` (job 67558).
- P4 CPU (tail truncation, shrinkage on lesion-level): `results/paperB/r5/p4/REPORT.md` (job 67559).

## (5) Verifier manifest

After reports exist, add every cite-able number from R5/R6 reports used in the manuscript draft
(R1, D1, S1/S2, clinical, P6, P8) to `paper/number_manifest.json` under **staged** keys
(`"staged": true` or id prefix `staged_`), pointing at the REPORT/summary JSON paths — so
`verify_manuscript_numbers.py` can lock them before tex edits. Manifest edit is a follow-on
commit after probe + existing report harvest; this decisions file only commits the plan.

## Outputs

`results/paperB/r5/heldout_probes/`:
- `per_run/<family>__<tag>.json`
- `summary.csv`
- `lambda025_per_seed.csv` + Spearman block in REPORT
- `REPORT.md` (tables next to pad_full; explicit primary/secondary reminder; post-commit finding
  banner if held-out ≠ pad_full story)

## Cost

CPU only on cached features. Rough: ~N_runs × 3 probes × 3 seeds; MLP dominates. Estimate
2–6 h on 8 CPUs for Table1+A4+R1+single (~100+ cells). tab:obj blocked on extract.

## Stop rules

Resume from per-run JSON. 12 h sbatch cap. Skip cells missing `pad_heldout.npz` (log in inventory).

## Deviations

(none yet — append after runs)
