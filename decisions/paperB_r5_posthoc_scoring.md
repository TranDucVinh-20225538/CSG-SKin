# Paper B — post hoc CPU: scoring-side mitigation, clinical consequence, nv_frac correlation, P3/P12 status

**Label: post hoc.** Committed before any of the scripts below run. Does not change GPU order (R1/R2).
No retraining. Features = cached `z_lesion_norm` / logits; detectors fit on ISIC train only.

## (1) Scoring-side mitigation (no retraining)

### Question

Can a scoring-only change to Mahalanobis restore OOD detection at λ = 2 on the three clinical
unseen sets without spoiling λ = 0?

### Variants (report every variant, not only the best)

1. **Standard Mahalanobis** (class-conditional, pooled within-class covariance + 1e-3·I) — baseline.
2. **Relative Mahalanobis (RMD)** — Ren et al. 2021; primary. Score = min_c (MD_c(x) − MD_0(x)), where
   MD_0 is one Gaussian on ISIC train ignoring labels (full covariance + 1e-3·I). Same fit as
   `scripts/r2_rmd.py`.
3. **Per-class standardised Mahalanobis** — secondary. Per-class mean and **per-class** covariance
   (+ 1e-3·I); score = min_c MD_c^{std}(x).

### Evaluation diagnostic (separate table; not a deployable fix)

Class-balanced AUROC (needs OOD labels): reweight OOD or ID so class mix matches; report for
labelled sets only (pad_heldout; mapped subset of MED-NODE / SD-198). Labelled **evaluation
diagnostic**.

### Pre-committed criterion (primary = RMD)

At λ = 2: AUROC > 0.5 **and** the cluster/patient CI excludes 0.5 on **≥ 2 of 3** unseen sets
(Fitzpatrick17k, MED-NODE, SD-198). At λ = 0: drop vs standard Mahalanobis ≤ 0.05 on each of those
three (mean over seeds). pad_heldout and SCIN are reported separately and do **not** enter the
criterion.

### Runs

- Table 1 old recipe: all 7 λ ∈ {0, 0.25, 0.5, 1, 2, 4, 8}, 5 seeds.
- A4 (P8 main): λ ∈ {0, 2} (available feature cells), 5 seeds.
- R1 (`cur`): shift 0% and shift 100% at λ = 2 (healthy seeds: ID bal ≥ 0.25); A4 λ = 0 as the
  λ = 0 reference for R1 cells that lack λ = 0 features. Available λ = 0.5 R1 cells are listed but
  do **not** enter the pre-committed (a) dose test.

### CI

PREREGISTER 2,000 draws, rng 20261005. ID by lesion; pad_heldout / pad_full by patient;
Fitzpatrick by P13 `cluster_union`; MED/SD/SCIN by P7a `group`.

### PREDICTED OUTCOME (written now)

- RMD recovers part of the λ = 2 inversion (as in R2 RMD on pad/Fitz) but **fails the criterion**:
  fewer than 2 of {Fitz, MED, SD-198} have RMD AUROC > 0.5 with CI entirely above 0.5.
- Per-class standardised Mahalanobis ≈ standard or slightly worse at λ = 2 (unstable rare-class
  covariances); does not meet the criterion either.
- At λ = 0, RMD drop vs standard ≤ 0.05 on Fitz/MED/SD (passes the λ = 0 limb); pad_heldout may
  drop ~0.02–0.04.
- Class-balanced diagnostic at λ = 2 moves AUROC toward 0.5 on pad_heldout but stays < 0.55.

## (2) Clinical consequence (post hoc)

### Question

If each model sets a 95% ID-acceptance threshold on its own ISIC **validation** Mahalanobis scores
(λ = 0 and λ = 2 separately), what share of malignant OOD images are accepted as ID, and what share
are predicted NV with confidence > 0.9?

### Design

- Score = standard Mahalanobis (higher = more OOD). Threshold τ = 95th percentile of ISIC val scores
  (accept as ID if score ≤ τ). Fit stats on ISIC train; threshold on ISIC val of the **same** run.
- Malignant ISIC-mapped classes: MEL, BCC, SCC, AK.
- Sets: Fitzpatrick17k and SD-198 (primary). pad_heldout: **n only** for MEL (9) — no rate table.
- Per class: n, share accepted as ID, share predicted NV with max softmax > 0.9; cluster/patient CIs
  (same units as (1)).
- **Label maps (stated in the report):**
  - Fitzpatrick17k: dataset `label` → ISIC via
    `{melanoma→MEL, basal cell carcinoma→BCC, squamous cell carcinoma→SCC, actinic keratosis→AK,
    dermatofibroma→DF, congenital nevus→NV}` (same as P7a fitz_skintype); unmapped excluded from
    per-class malignant rates, count reported.
  - SD-198: folder/`source_label` → ISIC via `SD198_MAP` in `scripts/r5_p7a_sets.py` (MEL/BCC/SCC/AK
    and the other five); unmapped excluded from rates, count reported.
  - pad_heldout: native PAD six-class labels (MEL, NV, BCC, AK, BKL, SCC).

### Runs

Old recipe and A4 at λ ∈ {0, 2}; R1 shift 0/100 at λ = 2 (healthy seeds).

### PREDICTED OUTCOME (written now)

- At λ = 0, malignant acceptance on Fitz/SD-198 is low-to-moderate (≤ 0.30 pooled); NV-conf>0.9
  among accepted malignants is uncommon.
- At λ = 2, malignant acceptance rises sharply (≥ 0.50 on at least one of Fitz/SD for MEL or BCC)
  and a non-trivial share of accepted malignants are predicted NV with conf > 0.9 — the operational
  cost of the collapse.

## (3) nv_frac vs AUROC (descriptive; post hoc)

### Question

Across lesion-level runs (Table 1 old, A4, R1 cells with features), what is the Spearman correlation
between OOD `nv_frac` (true-class mapped NV share; nearest-centroid NV share as secondary) and
Mahalanobis AUROC, per unseen set? How much AUROC variance does `nv_frac` vs leakage explain
(linear R², descriptive)?

### Wording

Correlation / variance explained only. **Do not** call this mediation. **Not** the pre-committed
R1 dose test (a), which still waits for λ = 0.5 to finish.

### PREDICTED OUTCOME (written now)

- Spearman(nv_frac, Maha AUROC) negative on Fitz/SD/SCIN (higher NV mix → lower AUROC), |ρ| ≈ 0.4–0.8.
- On pad_heldout, weaker (|ρ| < 0.4) because pad mix moves with the R1 dose by design.
- R²(nv_frac) > R²(leakage) on Fitz/SD at the pooled lesion-level panel; both leave residual variance.

## (4) Status of P3 and P12 (frozen ImageNet / DINOv2 / MONET)

| Item | Status at this commit | Action |
|---|---|---|
| P3 mapped (ImageNet R50 on R2 features re-indexed) | **done** (`results/paperB/r5/p3/`) | none |
| P3 full frames extract | frames CSV **done**; full R50 features **not** under `frozen/features/` | optional; mapped row is the current P3 primary |
| P12 (DINOv2-B/L, MONET + R50 full) | frames **done**; GPU extract **not submitted**; no `results/paperB/r5/p12/` | submit `r5_frozen_extract.sbatch` **after** R1 array 67547 (dependency), then CPU `STEP=p12` |

P12 must not take a GPU while R1 holds both vip2 GPUs.

## Outputs

- `results/paperB/r5/posthoc/scoring_mitigation/` — results.json, summary.csv, REPORT.md
- `results/paperB/r5/posthoc/clinical/` — results.json, summary.csv, REPORT.md
- `results/paperB/r5/posthoc/nvfrac_corr/` — results.json, summary.csv, REPORT.md
- P12 when extract finishes: `results/paperB/r5/p12/`

## Cost

GPU 0 for (1)–(3). CPU ≈ 2–4 h on 8 CPUs each (bootstrap-heavy). P12 extract ≈ 1.4 GPU-h after R1.

## Deviations
