# Paper B R6 / R1-hi — high-λ R1 at shift 0% (held-out invariance without label shift)

Committed before any R1-hi training, feature extraction, or scoring. Fixed now; deviations appended after runs.
Queued on GPU **after held R2 remainder resumes and finishes** (order 2026-10-17: P10 ∥ R2-prior →
release R2 CDAN/λ0.5 → R1-hi). Does not cut ahead of P10.

## Question

At the R1 three-stream design with **shift = 0%** (adversary sees PAD’s class mix; no ISIC→PAD label shift in the
adversary stream) and **RECIPE = current** (same as R1), does raising λ to {4, 8} produce held-out domain
invariance (gate) and, among gated non-degraded seeds, inversion on clinical unseen sets — or is label shift
necessary for inversion?

## Design

Same as R1 (`decisions/paperB_r6_r1.md`, mode `three`, RECIPE `current`): three streams of 32, GRL, λ_ctx = λ_orth = 1,
lesion-level ISIC split, pad_adv / pad_heldout Phase 2 split. Code: `scripts/train_r6_r1_hi.py` (+ `r6_common.py`),
report `scripts/r6_r1_hi_report.py`.

| Knob | Value |
|------|--------|
| shift | **0%** only (`adv_shift = 0` → ISIC adversary stream resampled to π_PAD) |
| RECIPE | **current** (same recipe object as R1 `cur/`) |
| checkpoint | **last.ckpt** (primary; `CKPT_POLICY = last`, same as R1/P8) |
| λ | **{4, 8}** × seeds **42/52/62/72/82** = 10 runs |
| fallback | if **no** seed passes the gate at λ = 8: one wave of **adversary LR ×3** (adv_lr_multiplier 90 instead of 30) at **λ = 4**, 5 seeds; then stop |

Tags: `r1hi_cur_shift0_ladv{4|8}_s{seed}`; fallback `r1hi_cur_shift0_ladv4_advlr3_s{seed}`.
Outputs under `results/paperB/r6/r1_hi/cur/` (never mixed into `r6/r1/cur/` tables).

## Degraded rule (flag, exclude from gated analyses)

ID balanced accuracy on ISIC test **< 0.25** → **degraded**: flagged in REPORT/summary, **excluded** from gate counts
and from prediction/opposite-reading tallies. Still listed with ID bal and probes.

## Gate (per seed)

Held-out **MLP** probe ≤ **0.55** under the **P6 V1** protocol (ISIC-test ∪ pad_heldout, GroupShuffleSplit 70/30 by
lesion/patient, `z_lesion_norm`, probe seeds 42/52/62; same MLP as `r5_heldout_probes.py`: 1×128, early stopping).
Report **n_pass / n_non_degraded** per cell (λ; and the advlr3 cell if run).

Also report held-out **linear / MLP / kNN(k=20)** for every seed/cell (primary leakage column remains pad_full when
cited elsewhere; here held-out is the gate variable).

## Reported (per cell)

- ID bal acc (mean ± s.d. over non-degraded seeds; all seeds listed); **drop vs R1 shift 0% λ = 2**
  (same recipe, `r6/r1/cur`, per-seed and cell mean of id_bal − id_bal_R1_shift0_λ2 for matching seeds).
- Gate: n seeds with held-out MLP ≤ 0.55 among non-degraded; list pass/fail per seed.
- All **5 detectors** (Mahalanobis, kNN-50, cosine, MSP, Energy) on pad_heldout, Fitzpatrick17k, MED-NODE, SD-198
  (and pad_full / SCIN as secondary), with **cluster CIs** (PREREGISTER 2,000 draws, rng 20261005; ID by lesion;
  pad by patient; Fitz by P13 cluster_union when available else image; P7a by group).
- Held-out linear / MLP / kNN next to pad_full linear (secondary vs primary reminder).

## Prediction criterion (gated, non-degraded seeds only)

**Primary prediction (written now):** among seeds that pass the gate and are not degraded, there is **no inversion**
on Fitzpatrick / MED-NODE / SD-198: Mahalanobis cluster-CI **lower bound > 0.5** on **≥ 2 of 3** unseen sets
(cell-level: using the gated seed pool; if only one gated seed, that seed’s CIs).

**Opposite reading (also fixed now):** if among those gated seeds **≥ 2 of 3** unseen sets **invert**
(Mahalanobis point AUROC < 0.5 with cluster-CI upper bound < 0.5, or pooled cell CI entirely below 0.5 — report the
same CI rule as R1 rule-A style for the gated pool), conclude:

> **Held-out invariance without label shift is sufficient for inversion.**

If the primary prediction holds, conclude the opposite: high-λ held-out invariance at shift 0% does **not** produce
clinical inversion; label shift (or something correlated with it in R1) remains necessary in this design.

## Fallback stop rule

1. Run λ ∈ {4, 8} × 5 seeds (last.ckpt, shift 0%, RECIPE current).
2. Score gates. If **≥ 1** non-degraded seed at **λ = 8** has held-out MLP ≤ 0.55 → **do not** run adv-LR×3;
   write the prediction / opposite reading on the gated pool.
3. If **zero** seeds pass the gate at λ = 8 (all fail MLP ≤ 0.55 or all degraded) → run **adversary LR ×3 at λ = 4**
   once (5 seeds), score the same way, then **stop**.
4. If after that wave still no gated seed (at λ = 8 or the advlr3 λ = 4 cell), REPORT the fixed sentence:

> **held-out invariance without label shift not reached**

and do not claim either the primary or the opposite inversion reading.

## PREDICTED OUTCOME (written now)

Applies only to **gated, non-degraded** seeds; if the gate is empty, the fallback / “not reached” path applies
instead of these numbers.

- **λ = 4, shift 0%:** ID bal stays competent for most seeds (≈ 0.40–0.55); drop vs R1 shift0 λ2 within ≈ 0.05–0.10
  absolute. Held-out MLP falls to ≈ 0.50–0.58; **2–4 / 5** seeds pass the ≤ 0.55 gate. Among gated seeds: Mahalanobis
  on Fitz/MED/SD stays **non-inverting** (CI lo > 0.5 on ≥ 2 of 3) — primary prediction holds; opposite reading not
  triggered.
- **λ = 8, shift 0%:** deeper invariance — held-out MLP ≤ 0.55 for **≥ 3 / 5** seeds (gate met; **no** adv-LR×3).
  ID bal may dip further (one seed may degrade). Gated pool still **no inversion** on ≥ 2 of 3 clinical sets
  (high λ without label shift removes domain probe signal but does not flip clinical Mahalanobis below 0.5).
- **Fallback:** predicted **not** needed. If λ = 8 unexpectedly fails the gate for all seeds, adv-LR×3 at λ = 4
  recovers 1–3 gated seeds; still no inversion (or, if gate still empty → “not reached”).

## Outputs

`results/paperB/r6/r1_hi/cur/`:
- `runs/<tag>/{config.json,summary.json}`, `features/<tag>/` (gitignored npz),
- `per_run_probes.json`, `cells.json`, `summary.csv`, `REPORT.md`
- Checkpoints: `checkpoints/r6_r1_hi/cur/<tag>/` (last.ckpt primary)

## Cost

≈ 2.4 GPU-h / three-stream run (R1 basis) × 10 = **24 GPU-h**; +5 × 2.4 = **36 GPU-h** if fallback fires.
Cap 40 GPU-h. CPU report + probes ≈ 2–4 CPU-h after features exist.

## Queue

GPU array after P10 completes (`afterany` on the P10 stage-2 array that ends the P10 chain). Does not hold or
release R2; does not pre-empt remaining R1 s82.

## Deviations

(none yet — append after runs)
