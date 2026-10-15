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
  weighted by w[y]; the source classification loss weighted by w as well (the paper's algorithm). w estimated without
  target labels: during each epoch accumulate the source confusion matrix C[i, j] = P_S(ŷ = i, y = j) (hard
  predictions, training batches, ISIC rows fed to the adversary) and the target prediction histogram μ[i] = P_T(ŷ = i)
  (PAD rows); at epoch end solve min ||C w − μ||² s.t. w ≥ 0, Σ_j w_j P_S(y = j) = 1 (SLSQP; the authors used cvxopt,
  not installed here); update w ← 0.5·w_old + 0.5·w_new (moving average as in the authors' code); w = 1 during epoch 1.
  Weights are logged per epoch (`iw_log.jsonl`), including the implied target mix w ⊙ p_S.
- PAD labels are never loaded into training (PAD rows carry label −1); they are used after training only to report
  the true PAD class proportions.

Everything else is the lesion-level Table 1 path; the recipe comes from `--recipe` (`current`, `p8`, `auto`) with the
recipe's own stream mode (two-stream unless the recipe uses class-balanced sampling).

## Runs, baselines

CDAN, IWDAN, IWCDAN × λ ∈ {0.5, 2} × seeds 42/52/62/72/82 = 30 runs.
λ = 0 baseline and plain DANN rows come from the same recipe (no new runs): current recipe → R4 lesion-level runs
(λ = 0 and 2: seeds 42/52/62/72/82; λ = 0.5: seeds 42–46, labelled); P8 recipe → P8 main runs (seeds 42–82).

## Data, OOD sets

Lesion-level ISIC split; Phase 2 PAD split (target = pad_adv). OOD: pad_heldout (primary), pad_full,
Fitzpatrick17k, P7a sets when available.

## Reported

Table 1 columns; five detectors; rule A bootstrap; per-class AUROC and decomposition check on pad_heldout; AK recall,
NV recall and NV-predicted share on pad_heldout (6-class); NV-nearest fractions; class-matched leakage; estimated vs true
PAD class proportions: (i) IW training estimate (last epoch) vs pad_adv, (ii) post-hoc BBSE from the final checkpoint
(ISIC val confusion, pad_heldout predictions) vs pad_heldout, both as NV share, AK share and total variation distance.
True mixes: pad_adv NV 0.099, AK 0.334; pad_heldout NV 0.123, AK 0.282.

## CI / bootstrap, gates

As R1 (`PREREGISTER_BOOTSTRAP.json` unchanged; ID by lesion, pad_heldout by patient, Fitzpatrick by image; rule A per
cell). Flags: adversary not engaged (leakage not ≥ 0.05 below λ = 0), no headroom at λ = 0, QP not converged in the
final epoch (logged).

## Predicted outcome (written now)

- CDAN: inversion persists at λ = 2 (Mahalanobis pad_heldout 0.38–0.50; rule A may or may not be met); conditioning on
  predictions does not remove the class-prior mismatch.
- IWDAN and IWCDAN: Mahalanobis pad_heldout 0.55–0.80 at λ = 2 (rule A not met); leakage at least 0.10 below the λ = 0
  value but above DANN's at the same λ; AK recall above DANN's at the same λ; NV-predicted share on pad_heldout below
  DANN's.
- Estimated PAD NV share 0.08–0.25 (true 0.099 on pad_adv); AK share underestimated (estimate 0.10–0.30 vs 0.334)
  because the ISIC classifier confuses AK with BKL/BCC; total variation 0.10–0.30.
- ID balanced accuracy of IW methods 0.02–0.08 below the λ = 0 baseline: the estimated weights for DF and VASC (absent
  from PAD) go towards 0, which removes those classes from the weighted classification loss.

## Outputs

`results/paperB/r6/r2/<recipe>/`: `runs/r2_<recipe>_<method>_ladv<λ>_s<seed>/{config.json, summary.json, iw_log.jsonl}`,
`features/...`, `summary.csv`, `cells.json` (includes all label-shift estimates), `REPORT.md`.
Checkpoints `checkpoints/r6_r2/<recipe>/`.

## GPU-hour estimate and basis

Basis 1.6 GPU-h per two-stream 40-epoch run (measured); CDAN head and IW bookkeeping negligible. 30 runs:
**48 GPU-h** (current recipe); 72 GPU-h with a three-stream P8 recipe (A5); ×2 if P8 selects 80 epochs (96 / 144).
No GPU for baselines (reused). CPU: report ≈ 1.5 CPU-h.

## Stop rules and budget caps

Cap 55 GPU-h (current recipe) / 150 GPU-h (P8 worst case). If importance weights diverge (any w > 50 in `iw_log.jsonl`,
or a NaN loss; checked by the parent from the logs), the run is stopped and reported with its log; no clipping is added
after the fact.

## Deviations

