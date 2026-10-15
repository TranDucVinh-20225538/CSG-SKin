# Paper B R5 / P2 — §4.4 (confidence and calibration) on the lesion-level runs

Committed before any P2 run (no P2 output exists at this commit). Fixed now; deviations appended at the end.

## Question

§4.4 reports, on the image-level runs, that mean OOD confidence overtakes ID confidence as λ rises, OOD ECE rises while
ID ECE stays flat, and cross-domain predictions collapse toward NV (AK recall falls). Does this hold on the 35
lesion-level runs?

## Data and design (code: `scripts/r5_p2_confidence.py`)

Cached logits of the 35 lesion-level runs (`results/paperB/r2/item4/features/<tag>/`). Per run:
- mean MSP (max softmax) on ISIC test, pad_heldout, Fitzpatrick17k; gaps OOD − ID;
- ECE, 15 equal-width bins (`phase15_common.ece_from_logits` definition): ID (ISIC test, 8 classes), OOD pad_heldout
  8-class softmax (Table 1 "OOD ECE") and 6-class restricted softmax (sensitivity);
- pad_heldout per-class recall under the 6-class restriction (`six_class_xfer`, Table 1 X-dom), AK recall (6-class and
  8-class argmax), predicted-NV share on pad_heldout, balanced accuracy;
- cross-check: ECE values vs `results/paperB/r3/c/per_run` and the R4 Table 1 numbers (max |Δ| reported).

## Seeds, λ grid, OOD sets

All 7 λ, 35 runs; pad_heldout (labels; ECE, recall) and Fitzpatrick17k (confidence only, no mapped labels used).

## Gates and controls

Table 1 reproduction check: OOD ECE and X-dom bal acc must match R4 Table 1 to 1e-6 → any difference is flagged.
Fitzpatrick has no ECE (labels do not map on most images).

## CI

Mean ± s.d. over seeds per λ; no new bootstrap.

## PREDICTED OUTCOME (written now)

- Mean confidence: ID 0.60–0.70 at all λ; pad_heldout 0.45–0.60 at λ = 0, rising to 0.80–0.92 at λ ≥ 0.5 (above ID);
  gap OOD − ID ≤ −0.05 at λ = 0 and ≥ +0.10 at λ = 2.
- ECE: ID 0.14–0.19 flat; OOD 0.22 at λ = 0 → 0.70–0.73 at λ ≥ 0.5 (Table 1: 0.219 → 0.717).
- AK recall (6-class) falls from 0.20–0.45 at λ = 0 to < 0.10 at λ ≥ 0.5; predicted-NV share on pad_heldout rises from
  0.10–0.20 to ≥ 0.50.
- Fitzpatrick17k confidence follows pad_heldout (≥ ID at λ ≥ 0.5).

## Outputs

`results/paperB/r5/p2/`: `per_run/<tag>.json`, `summary.csv`, `REPORT.md` (LaTeX rows; one paragraph per result).

## Cost

GPU 0. CPU < 0.2 h on 4 CPUs (logits only).

## Stop rules

None needed beyond the 1 h sbatch cap.

## Deviations
