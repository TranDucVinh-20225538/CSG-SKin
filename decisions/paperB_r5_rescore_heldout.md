# Paper B R5 — rescore tab:single and tab:obj on pad_heldout with the patient-clustered CI

Committed before any job of this item runs. Fixed now; not changed after results are seen. Deviations are appended at
the end.

## Question (user, 2026-10-15)

tab:single (single-encoder DANN, λ sweep) and tab:obj (objectives incl. MMD weights) print OOD AUROCs scored on
pad_full (2,298 PAD images, 1,582 of which the adversary / invariance penalty trained on), unlabelled (audit A6: 21
cells). Rescore both tables on pad_heldout (716 images / 412 patients the models never saw), for all five detectors,
all weights and seeds, with the patient-clustered bootstrap CI, so both tables and the MMD text use one OOD set.

## Runs (76; CPU re-forward of the saved best checkpoints)

- tab:single: `results/paperB/phase15/single_dann/single_dann_ladv*_s*`. λ ∈ {0, 0.25, 0.5, 1, 2, 4, 8}; 5 seeds at
  λ = 0 and λ = 2, 3 seeds elsewhere (25 runs). Loaded with `SingleDannLightning`.
- tab:obj: `results/paperB/phase15/objectives/*` (ERM w0; CORAL w1/10/100; GroupDRO w0.1/1/10; IRM w1/10/100;
  MMD w1/10/100; 3 seeds each; 39 runs) and `results/paperB/phase15b/mmd/*` (MMD w300/1000/3000/10000; 3 seeds;
  12 runs). Loaded with `train_phase15_2_objectives.InvariantLightning`.

Evaluation path unchanged from Phase 15 / R4 item 4 (`r4_mmd_pad_heldout.py`):
- image-level `p2.isic_splits`, `build_val_transform_robust`, `collect_all`;
- detectors fit on ISIC train only (`phase15_common.detector_block` definitions: Mahalanobis min squared distance with
  shared covariance + 1e-3·I, cosine to class means, kNN k = 50 on L2-normalised features, MSP, Energy);
- pad_full is forwarded once and pad_heldout is taken from it by path (the heldout rows are a subset).

## Pipeline checks (flag, never drop)

- pad_full AUROC recomputed vs the stored summary value, per detector; tolerance 2e-3. Beyond that the run is flagged
  and reported.
- For single_dann, the stored pad_heldout values are checked the same way.
- pad_heldout row order vs the patient metadata (paths equal, asserted).

## Bootstrap / CI

PREREGISTER_BOOTSTRAP.json: 2,000 draws, rng seed 20261005. `r5cpu_common.pooled_bootstrap` (the main-results path):
- ID side: resampled by lesion group on the image-level ISIC test split (`lesion_groups`, ISIC metadata; images
  without a lesion id are their own group);
- pad_heldout side: resampled by patient;
- draws shared across seeds within a cell; CI = percentile 95% of the seed mean;
- per-seed CIs and the code's rule A / B / C / above, per detector.

## Also recorded (secondary)

Held-out leakage probe on the same forward pass: P6 V1 protocol (ISIC test ∪ pad_heldout, 70/30 grouped by lesion /
patient, probe seeds 42/52/62, balanced accuracy) on the scoring features. Reported next to the stored pad_full
leakage. No rule.

## Reported cells

- tab:single: one row per λ; per detector mean ± s.d. over seeds and the pooled CI; n seeds; rule.
- tab:obj: one row per objective × weight, same columns.
- The pad_full mean of each cell is shown next to it (stored values) so the change of OOD set is visible.
- LaTeX rows for both tables in the manuscript's column order.
- The MMD text values (weights 1 → 10,000), on pad_heldout with CIs.

## Predictions (written before any rescored number exists)

1. Pipeline: recomputed pad_full within 2e-3 of the stored values for every run. single_dann recomputed pad_heldout
   within 2e-3 of stored.
2. tab:single on pad_heldout:
   - λ = 0: all detectors 0.70–0.94 (stored points: Maha 0.700, cosine 0.875, kNN 0.935, MSP 0.857, Energy 0.901).
   - λ ≥ 0.25: every detector mean within 0.49–0.56; no cell meets rule A. The λ = 4 cosine (0.490) CI includes 0.5
     (rule C).
   - CI widths (seed mean, 3 seeds) about 0.03–0.05.
3. tab:obj on pad_heldout: Mahalanobis within ±0.01 of pad_full for every objective and weight. MMD w1 ≈ 0.81,
   w3000 ≈ 0.75, w10000 ≈ 0.66; IRM ≈ 0.72–0.74; GroupDRO ≈ 0.86–0.87; CORAL and ERM within ±0.01 of their pad_full
   values. No tab:obj cell meets rule A or B for Mahalanobis (no objective inverts on pad_heldout).
4. The other detectors in tab:obj move by ≤ 0.02 between pad_full and pad_heldout in most cells (> 0.02 in at most
   one fifth of the cells).
5. Held-out leakage probe: within ±0.03 of the stored pad_full leakage for the objectives; lower by 0.00–0.05 for
   single_dann at λ > 0.

## Outputs

`results/paperB/r5/rescore_heldout/`:
- `per_run/<run>/{scores.npz, record.json}`;
- `summary.csv`, `BOOTSTRAP.json`, `REPORT.md` with LaTeX rows.

## Compute

CPU only: `scripts/r5_rescore_heldout.py` (worker per run) as a `pB-` array of 76 tasks, 8 CPUs each, at most 3 at a
time. About 20–25 min per run (image-level ISIC train dominates), so about 30 CPU-job-hours and about 9–10 h wall at
3 concurrent. Report job afterwards, about 0.3 h.

## Deviations
