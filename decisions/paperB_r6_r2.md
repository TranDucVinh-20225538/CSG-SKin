# Paper B R6 / R2 — label-shift-aware alignment (CDAN, IWDAN, IWCDAN) on the dual encoder

Committed before any R2 run (no R2 training, feature extraction or scoring has run at this commit).
Fixed now; not changed after results are seen. Deviations are appended at the end.

## Question

Does an alignment method that corrects for label shift keep invariance (leakage falls) while OOD AUROC stays above 0.5
and cross-domain predictions do not collapse to the majority class (AK recall)? And how close are its label-free
estimates of the PAD class proportions to the true ones?

## Methods (code: `scripts/train_r6_iwdan.py`, `scripts/r6_common.py::R6Lightning`)

- **CDAN** (Long et al. 2018), plain (no entropy conditioning): the domain head sees the flattened outer product
  softmax(ŷ) ⊗ GRL(z_lesion_norm); softmax detached as in the reference implementation. Input dimension 8 × latent
  (128 at latent 16); head = the existing adversary template scaled to that input.
- **IWDAN / IWCDAN** (Tachet des Combes et al., NeurIPS 2020): DANN / CDAN domain loss with each source (ISIC) row
  weighted by w[y]; the source classification loss weighted by w as well (the paper's algorithm). **Estimated branch
  only** uses target class weights from source-trained predictions on **unlabeled** pad_adv (no PAD labels in any
  loss or in the estimation): during each epoch accumulate the source confusion matrix
  C[i, j] = P_S(ŷ = i, y = j) (hard predictions, training batches, ISIC rows fed to the adversary) and the target
  prediction histogram μ[i] = P_T(ŷ = i) (PAD rows); at epoch end solve min ||C w − μ||² s.t. w ≥ 0,
  Σ_j w_j P_S(y = j) = 1 (SLSQP); update w ← 0.5·w_old + 0.5·w_new (moving average as in the authors' code);
  w = 1 during epoch 1. Weights are logged per epoch (`iw_log.jsonl`), including the implied target mix w ⊙ p_S.
- **Oracle branch (IWDAN only):** w fixed from the **true** pad_adv class proportions π_T via
  w_j = π_T(j) / P_S(y = j) (renormalised so Σ_j w_j P_S(y = j) = 1). PAD labels set these weights only; they never
  enter a loss. Labelled **“oracle, not deployable”** — an upper bound on what label-shift reweighting can do when
  π_T is known. No QP / no prediction-based estimate.
- PAD labels are never loaded into any loss. For the estimated branch they are used after training only to report
  true PAD class proportions vs estimates. For the oracle branch they set w only.

Everything else is the lesion-level Table 1 path; the recipe comes from `--recipe` (`current`, `p8`, `auto`) with the
recipe's own stream mode (two-stream unless the recipe uses class-balanced sampling).

## Runs, baselines

- Estimated: CDAN, IWDAN, IWCDAN × λ ∈ {0.5, 2} × seeds 42/52/62/72/82 = **30** runs.
- Oracle: IWDAN × λ ∈ {0.5, 2} × seeds 42/52/62/72/82 = **10** runs (λ = 0.5 included; cheap at ~1.6 GPU-h each).
- Total new trainings: **40** runs.
- λ = 0 baseline and plain DANN rows come from the same recipe (no new runs): current recipe → R4 lesion-level runs
  (λ = 0 and 2: seeds 42/52/62/72/82; λ = 0.5: seeds 42–46, labelled); P8 recipe → P8 main runs (seeds 42–82).

## Data, OOD sets

Lesion-level ISIC split; Phase 2 PAD split (target = pad_adv). OOD: pad_heldout (reported separately), pad_full,
Fitzpatrick17k, MED-NODE, SD-198 (primary for the fix reading below), P7a SCIN when available.

## Reported

Table 1 columns; five detectors; rule A bootstrap; per-class AUROC and decomposition check on pad_heldout; AK recall,
NV recall and NV-predicted share on pad_heldout (6-class); NV-nearest fractions; class-matched leakage.

**Estimated vs true PAD proportions (required).** Per class and per seed for IWDAN / IWCDAN (last-epoch IW estimate
and post-hoc BBSE): estimated π vs true pad_adv (and vs pad_heldout for BBSE). Report **L1** and **total-variation**
error of the estimated mix vs true, with mean ± s.d. over seeds. Oracle rows log the true mix used as w (TV = 0 by
construction against pad_adv).

## Pre-committed reading: does label-shift correction “fix” detection?

Primary sets: **Fitzpatrick17k, MED-NODE, SD-198**. pad_heldout is reported separately and does **not** count toward
the reading. “Fixes” = Mahalanobis AUROC **95% CI lower bound > 0.5** and Table 1 pad_full leakage still
**≤ GRL λ = 2 leakage + 0.05** (same recipe; GRL = the plain DANN / item4 λ = 2 cell). Applied at λ = 2 (primary);
λ = 0.5 reported.

| estimated | oracle | Reading |
|---|---|---|
| fixes | fixes | **deployable fix** |
| fails | fixes | **estimation error** (report weight L1/TV error) |
| fails | fails | **label-shift correction isn’t sufficient** |
| fixes | fails | **anomaly — investigate** |

Per primary set; summarise how many of the three sets fall in each cell. Degraded runs (if any) flagged; excluded from
the pooled CI used in “fixes”.

## CI / bootstrap, gates

As R1 (`PREREGISTER_BOOTSTRAP.json` unchanged; ID by lesion, pad_heldout by patient, Fitzpatrick by P13 cluster when
present else image; P7a by group; rule A per cell). Flags: adversary not engaged (leakage not ≥ 0.05 below λ = 0),
no headroom at λ = 0, QP not converged in the final epoch (logged; estimated branch only).

## Predicted outcome (written now)

- CDAN: inversion persists at λ = 2 (Mahalanobis pad_heldout 0.38–0.50; rule A may or may not be met); conditioning on
  predictions does not remove the class-prior mismatch.
- IWDAN and IWCDAN (estimated): Mahalanobis pad_heldout 0.55–0.80 at λ = 2 (rule A not met); leakage at least 0.10
  below the λ = 0 value but above DANN's at the same λ; AK recall above DANN's at the same λ; NV-predicted share on
  pad_heldout below DANN's.
- Estimated PAD NV share 0.08–0.25 (true 0.099 on pad_adv); AK share underestimated (estimate 0.10–0.30 vs 0.334)
  because the ISIC classifier confuses AK with BKL/BCC; total variation 0.10–0.30.
- Oracle IWDAN: on the three primary unseen sets, more often “fixes” than the estimated branch; if estimated fails and
  oracle fixes → reading = estimation error. If both fail on ≥ 2/3 sets → label-shift correction isn’t sufficient.
- ID balanced accuracy of IW methods 0.02–0.08 below the λ = 0 baseline: the estimated weights for DF and VASC (absent
  from PAD) go towards 0, which removes those classes from the weighted classification loss.

## Outputs

`results/paperB/r6/r2/<recipe>/`: `runs/r2_<recipe>_<method>_ladv<λ>_s<seed>/{config.json, summary.json, iw_log.jsonl}`,
`features/...`, `summary.csv`, `cells.json` (includes all label-shift estimates), `REPORT.md`.
Methods tags: `cdan`, `iwdan`, `iwcdan`, `iwdan_oracle`. Checkpoints `checkpoints/r6_r2/<recipe>/`.

## GPU-hour estimate and basis (updated for oracle)

Basis 1.6 GPU-h per two-stream 40-epoch run (measured); CDAN head and IW bookkeeping negligible.

| Block | Runs | GPU-h |
|---|---:|---:|
| Estimated CDAN/IWDAN/IWCDAN | 30 | 48 |
| Oracle IWDAN (λ ∈ {0.5, 2} × 5 seeds) | 10 | **+16** |
| **R2 total** | **40** | **64** |

No GPU for baselines (reused). CPU: report ≈ 2 CPU-h (extra oracle + proportion tables).

**ETAs (wall, 2 GPUs concurrent on vip2, after R1 finishes):** R2 train ≈ 64/2 ≈ **32 h**; R2 report ≈ 2 h;
P10 (next in GPU order) ladder+stage2 ≈ **8–12 h** wall within the 40 GPU-h cap. Cap R2: **70 GPU-h** (was 55;
raised for the oracle branch). P10 hard cap unchanged at **40 GPU-h**.

## Stop rules and budget caps

Cap **70 GPU-h** (current recipe) / 160 GPU-h (P8 worst case). If importance weights diverge (any w > 50 in
`iw_log.jsonl`, or a NaN loss; checked by the parent from the logs), the run is stopped and reported with its log; no
clipping is added after the fact. Oracle runs do not estimate w (fixed); divergence check applies to estimated branch
only.

## GPU queue order (Paper B)

User order (2026-10-16): **R1 → R2 → P10 → P9 → P11 → P7(b) → Q1 → Q3**.

## Deviations

- 2026-10-16 14:30 (UTC+7), before any R2 run. **Oracle IWDAN branch** and **pre-committed fix reading** on
  Fitzpatrick / MED-NODE / SD-198 added (sections above). Estimated branch clarified: target weights from unlabeled
  pad_adv predictions only. Extra **+16 GPU-h** (10 oracle runs); R2 cap 55 → 70. GPU order places P10 immediately
  after R2.
