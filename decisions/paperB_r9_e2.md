# Paper B R9 / E2 — stronger λ = 0 recipe (80-epoch schedule)

Committed before any E2 run. Fixed now; not changed after results are seen. Deviations are appended at the end.

## Why

The competence gap is 0.04–0.08 (dual λ = 0: 0.481 main, 0.520 A4; single-encoder ERM 0.558 ± 0.022). P8 searched latent size, classifier LR and class balance at 40 epochs; its 80-epoch stage B was never run. E2 asks whether a longer schedule brings the dual encoder within 0.03 of ERM. If not, the paper says so.

## Stage S (screen, blind to OOD)

λ = 0, two-stream Table 1 path, seeds 42 and 52. Compute ID val/test balanced accuracy and the degraded flag only. No leakage, no OOD score.

- Dual: A4 at 80 epochs, A7 at 80 epochs (4 runs).
- ERM single-encoder EfficientNet-B3 at 80 epochs, seeds 42 and 52 (2 runs), so the comparison stays matched in schedule. ERM reference for E2 = the higher of 0.558 (40 epochs, 5 seeds) and the 80-epoch two-seed mean.
- Selection: higher mean val balanced accuracy over the two seeds; ties within 0.005 → A4.
- Gate S: two-seed mean ID test balanced accuracy of the selected recipe ≥ ERM reference − 0.03. Gate not met → stop E2. Write "competence gap not closed by an 80-epoch schedule" in REPORT.md with the numbers.
- ≈ 13 GPU-h for the 4 dual runs plus the 2 ERM runs.

## Stage C (confirm, only if gate S is met)

Selected recipe, λ = 0, seeds 62, 72, 82 (seeds 42 and 52 reused). Gate C: five-seed mean ID test balanced accuracy ≥ ERM reference − 0.03. Gate C not met → stop and report as above. ≈ 8 GPU-h.

## Stage M (main, only if gate C is met)

Selected recipe, λ = 2 × seeds 42–82 (5 runs, ≈ 13 GPU-h), then full Table 1 protocol on λ = 0 and λ = 2 (all detectors and sets, Rule A, leakage V1, ID balanced accuracy and ECE, FPR@95TPR as in R7b C1, paired λ = 2 − λ = 0 change in ID balanced accuracy with the D4 bootstrap).

## Reading rules for stage M (as R7 T1)

Competent = five-seed λ = 0 gap to the ERM reference ≤ 0.03.

- **Case A:** Competent, Rule A met on pad_heldout and ≥ 2 of the 3 labelled unseen sets at λ = 2: the inversion holds for a dual encoder within 0.03 of ERM.
- **Case B:** Competent, Rule A not met: the inversion does not hold at this competence.
- **Case C:** Leakage V1 at λ = 2 above 0.60: invariance not reached; reported, settles neither.

Report the case first.

## Shared rules (R7 / R8)

Lesion-disjoint ISIC split (seed 42), pad_adv / pad_heldout, last.ckpt, held-out leakage probe V1 as Table 1.
Degraded = ID test balanced accuracy < 0.25: report the run, exclude it from means (as R1).
Report every run, including failed or degraded ones. No pass/fail wording beyond the rules above.

## Not in R9

Fitzpatrick17k as a second target domain (it would remove Fitzpatrick17k from the OOD sets).
More MK-MMD seeds (five-seed mean V1 0.604 already decides "not reached").

## Outputs

`results/paperB/r9/e2/`: stage S table, selection file, gate results; stage C/M tables with LaTeX rows in the Table 1 format if run; folded into `results/paperB/r9/REPORT.md`.

## Deviations

(none yet)
