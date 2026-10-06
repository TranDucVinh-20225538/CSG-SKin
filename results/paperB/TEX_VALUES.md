# Remaining manuscript values (Phase 6 / array 60585)

Extracted by `scripts/emit_tex_values.py` from `results/paperB/phase6_xfer/`.
Statistic: 6-class balanced accuracy on **pad_heldout** (716 images / 412 patients),
mean ± s.d. across seeds. Softmax/argmax restricted to MEL, NV, BCC, AK, BKL, SCC.
No recomputation. No pad_full / pad_adv substitute.

Skipped non-run files: diagnosis_aggregate.json.

## 1. CSG remaining λ points

Already in the manuscript (not re-emitted):
- λ=0: 0.291 ± 0.016 (n=5, files: runB_orth1_ladv0_s42.json, runB_orth1_ladv0_s52.json, runB_orth1_ladv0_s62.json, runB_orth1_ladv0_s72.json, runB_orth1_ladv0_s82.json)
- λ=0.25: 0.291 ± 0.020 (n=3, files: runB_orth1_ladv0p25_s42.json, runB_orth1_ladv0p25_s52.json, runB_orth1_ladv0p25_s62.json)
- λ=2: 0.249 ± 0.030 (n=5, files: runB_orth1_ladv2_s42.json, runB_orth1_ladv2_s52.json, runB_orth1_ladv2_s62.json, runB_orth1_ladv2_s72.json, runB_orth1_ladv2_s82.json)

- λ=0.5 (`\xdomHalf`): 0.270 ± 0.034 (n=3). subset=pad_heldout, 716/412, 6-class bal acc. Files: runB_orth1_ladv0p5_s42.json, runB_orth1_ladv0p5_s52.json, runB_orth1_ladv0p5_s62.json.
- λ=1 (`\xdomOne`): 0.261 ± 0.016 (n=3). subset=pad_heldout, 716/412, 6-class bal acc. Files: runB_orth1_ladv1_s42.json, runB_orth1_ladv1_s52.json, runB_orth1_ladv1_s62.json.
- λ=4 (`\xdomFour`): 0.276 ± 0.025 (n=3). subset=pad_heldout, 716/412, 6-class bal acc. Files: runB_orth1_ladv4_s42.json, runB_orth1_ladv4_s52.json, runB_orth1_ladv4_s62.json.
- λ=8 (`\xdomEight`): 0.269 ± 0.023 (n=5). subset=pad_heldout, 716/412, 6-class bal acc. Files: runB_orth1_ladv8_s42.json, runB_orth1_ladv8_s52.json, runB_orth1_ladv8_s62.json, runB_orth1_ladv8_s72.json, runB_orth1_ladv8_s82.json.

## 2. EffNet-B3 single-encoder control

- `\xdomEffb`: 0.298 ± 0.010 (n=5). same statistic as the CSG rows: 6-class bal acc on pad_heldout 716/412. Files: effb3_control_s42.json, effb3_control_s52.json, effb3_control_s62.json, effb3_control_s72.json, effb3_control_s82.json.

## 3. pad_heldout class counts (AK, SCC)

- `\nAK` = 202, `\nSCC` = 66. Full counts on pad_heldout (716/412): {'MEL': 9, 'NV': 88, 'BCC': 276, 'AK': 202, 'BKL': 75, 'SCC': 66}. MEL=9 matches the manuscript. A proportional 31.2% of full PAD would be AK ≈ 227, SCC ≈ 60; patient-level split does not preserve proportions (MEL 9 vs proportional 16). Source: `class_counts` in every phase6_xfer run JSON (verified identical).
- AK denominator 202 ≥ 100. No flag. SCC n=66 is smaller and is not the gated claim.

## Report only — per-class recall at λ=0 and λ=2 (not for the manuscript block)

Numerator = round(recall × class_count) per seed; denominator is the pad_heldout count. Mean TP is the average of those integers.

### λ=0 (n=5)

- MEL: mean 4.4/9 (recall 0.489 ± 0.149). s42: 5/9; s52: 3/9; s62: 3/9; s72: 5/9; s82: 6/9.
- NV: mean 43.8/88 (recall 0.498 ± 0.026). s42: 42/88; s52: 44/88; s62: 46/88; s72: 46/88; s82: 41/88.
- BCC: mean 54.8/276 (recall 0.199 ± 0.085). s42: 58/276; s52: 36/276; s62: 80/276; s72: 74/276; s82: 26/276.
- AK: mean 49.4/202 (recall 0.245 ± 0.102). s42: 56/202; s52: 78/202; s62: 49/202; s72: 21/202; s82: 43/202.
- BKL: mean 17.2/75 (recall 0.229 ± 0.042). s42: 18/75; s52: 14/75; s62: 21/75; s72: 14/75; s82: 19/75.
- SCC: mean 5.8/66 (recall 0.088 ± 0.025). s42: 5/66; s52: 4/66; s62: 7/66; s72: 5/66; s82: 8/66.

### λ=2 (n=5)

- MEL: mean 2.8/9 (recall 0.311 ± 0.199). s42: 4/9; s52: 4/9; s62: 4/9; s72: 2/9; s82: 0/9.
- NV: mean 77.8/88 (recall 0.884 ± 0.012). s42: 77/88; s52: 79/88; s62: 77/88; s72: 77/88; s82: 79/88.
- BCC: mean 37.2/276 (recall 0.135 ± 0.020). s42: 42/276; s52: 29/276; s62: 34/276; s72: 40/276; s82: 41/276.
- AK: mean 3.2/202 (recall 0.016 ± 0.012). s42: 1/202; s52: 4/202; s62: 7/202; s72: 2/202; s82: 2/202.
- BKL: mean 9.4/75 (recall 0.125 ± 0.028). s42: 8/75; s52: 12/75; s62: 7/75; s72: 9/75; s82: 11/75.
- SCC: mean 1.4/66 (recall 0.021 ± 0.008). s42: 1/66; s52: 2/66; s62: 1/66; s72: 2/66; s82: 1/66.

Summary AK: 49.4/202 → 3.2/202 (λ=0 n=5, λ=2 n=5).
Summary SCC: 5.8/66 → 1.4/66 (λ=0 n=5, λ=2 n=5).

## FILL-IN BLOCK

```
\newcommand{\xdomHalf}{0.270\,$\pm$\,0.034}
\newcommand{\xdomOne}{0.261\,$\pm$\,0.016}
\newcommand{\xdomFour}{0.276\,$\pm$\,0.025}
\newcommand{\xdomEight}{0.269\,$\pm$\,0.023}
\newcommand{\xdomEffb}{0.298\,$\pm$\,0.010}
\newcommand{\nAK}{202}
\newcommand{\nSCC}{66}
```

