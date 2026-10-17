# Paper B — held-out domain probes (linear / MLP / kNN)

Design: `decisions/paperB_r5_heldout_probes.md`. Job 68660.
**On origin/paperb/reviewer-r1:** `df31679` (this REPORT + summary/λ0.25 tables).
Protocol: P6 V1 (ISIC-test ∪ pad_heldout, GroupShuffleSplit 70/30 by lesion/patient).
Representation: `z_lesion_norm`. Probe seeds 42/52/62.

**Primary leakage variable unchanged:** pad_full linear probe (94e2399).
**Secondary:** held-out linear / MLP / kNN. If held-out ≠ pad_full story, both are reported
and labelled a finding made after the decisions commit.

Inventory: 85 runs with cached features. tab:obj / P9 / P11 / R2: deferred until features exist.

## Cell means (mean ± s.d. over training seeds; bootstrap CI of seed-mean, n≥2)

| family | shift | λ | n | pad_full linear (primary) | heldout linear | heldout MLP | heldout kNN20 |
|---|---:|---:|---:|---|---|---|---|
| a4_main |  | 0.0 | 5 | 0.903 ± 0.012 [0.894, 0.912] | 0.842 ± 0.025 [0.822, 0.861] | 0.851 ± 0.024 [0.832, 0.869] | 0.819 ± 0.011 [0.809, 0.827] |
| a4_main |  | 0.5 | 5 | 0.780 ± 0.010 [0.773, 0.788] | 0.663 ± 0.017 [0.651, 0.677] | 0.651 ± 0.057 [0.609, 0.697] | 0.556 ± 0.021 [0.542, 0.573] |
| a4_main |  | 2.0 | 5 | 0.786 ± 0.034 [0.757, 0.811] | 0.670 ± 0.038 [0.639, 0.700] | 0.711 ± 0.042 [0.677, 0.741] | 0.565 ± 0.016 [0.556, 0.579] |
| r1_cur | 0 | 0.5 | 4 | 0.790 ± 0.007 [0.784, 0.795] | 0.612 ± 0.013 [0.601, 0.621] | 0.676 ± 0.013 [0.665, 0.686] | 0.695 ± 0.011 [0.686, 0.705] |
| r1_cur | 0 | 2.0 | 5 | 0.782 ± 0.019 [0.765, 0.795] | 0.629 ± 0.019 [0.614, 0.643] | 0.635 ± 0.052 [0.594, 0.676] | 0.670 ± 0.070 [0.606, 0.705] |
| r1_cur | 50 | 0.5 | 3 | 0.629 ± 0.045 [0.590, 0.679] | 0.531 ± 0.038 [0.504, 0.574] | 0.511 ± 0.018 [0.500, 0.531] | 0.555 ± 0.021 [0.530, 0.568] |
| r1_cur | 50 | 2.0 | 3 | 0.658 ± 0.020 [0.643, 0.681] | 0.547 ± 0.013 [0.532, 0.556] | 0.510 ± 0.014 [0.501, 0.525] | 0.616 ± 0.011 [0.608, 0.628] |
| r1_cur | 100 | 0.0 | 5 | 0.928 ± 0.020 [0.912, 0.944] | 0.841 ± 0.051 [0.804, 0.883] | 0.910 ± 0.021 [0.894, 0.926] | 0.922 ± 0.010 [0.914, 0.931] |
| r1_cur | 100 | 0.5 | 5 | 0.628 ± 0.062 [0.582, 0.677] | 0.510 ± 0.013 [0.502, 0.521] | 0.500 ± 0.000 [0.500, 0.500] | 0.638 ± 0.051 [0.603, 0.684] |
| r1_cur | 100 | 2.0 | 5 | 0.634 ± 0.054 [0.592, 0.675] | 0.510 ± 0.007 [0.505, 0.515] | 0.500 ± 0.000 [0.500, 0.500] | 0.633 ± 0.028 [0.609, 0.651] |
| single_p5 |  | 4.0 | 3 | 0.798 ± 0.010 [0.789, 0.808] | 0.728 ± 0.012 [0.721, 0.742] | 0.705 ± 0.011 [0.695, 0.717] | 0.521 ± 0.027 [0.504, 0.553] |
| single_p8 |  | 0.0 | 2 | 0.975 ± 0.004 [0.972, 0.977] | 0.939 ± 0.005 [0.936, 0.943] | 0.930 ± 0.006 [0.925, 0.934] | 0.879 ± 0.021 [0.865, 0.894] |
| table1 |  | 0.0 | 5 | 0.927 ± 0.018 [0.913, 0.942] | 0.827 ± 0.056 [0.785, 0.872] | 0.921 ± 0.019 [0.904, 0.935] | 0.935 ± 0.009 [0.927, 0.942] |
| table1 |  | 0.25 | 5 | 0.614 ± 0.092 [0.553, 0.696] | 0.538 ± 0.074 [0.502, 0.604] | 0.547 ± 0.106 [0.500, 0.643] | 0.586 ± 0.099 [0.528, 0.675] |
| table1 |  | 0.5 | 5 | 0.581 ± 0.031 [0.555, 0.603] | 0.506 ± 0.007 [0.502, 0.513] | 0.500 ± 0.001 [0.500, 0.501] | 0.548 ± 0.009 [0.542, 0.556] |
| table1 |  | 1.0 | 5 | 0.563 ± 0.045 [0.533, 0.602] | 0.504 ± 0.004 [0.501, 0.506] | 0.500 ± 0.000 [0.500, 0.500] | 0.560 ± 0.029 [0.537, 0.583] |
| table1 |  | 2.0 | 5 | 0.555 ± 0.046 [0.523, 0.592] | 0.502 ± 0.003 [0.500, 0.505] | 0.500 ± 0.000 [0.500, 0.500] | 0.578 ± 0.036 [0.550, 0.607] |
| table1 |  | 4.0 | 5 | 0.636 ± 0.076 [0.577, 0.695] | 0.516 ± 0.019 [0.503, 0.532] | 0.500 ± 0.000 [0.500, 0.500] | 0.570 ± 0.044 [0.538, 0.605] |
| table1 |  | 8.0 | 5 | 0.663 ± 0.068 [0.624, 0.723] | 0.550 ± 0.057 [0.514, 0.603] | 0.523 ± 0.052 [0.500, 0.570] | 0.608 ± 0.033 [0.585, 0.635] |

## λ = 0.25 Table 1 — per seed (prediction: near-chance held-out ⇒ invert)

| seed | heldout linear | heldout MLP | Maha pad_heldout | Maha Fitz | Maha MED-NODE | Maha SD-198 |
|---:|---|---|---|---|---|---|
| 42 | 0.669 | 0.738 | 0.715 | 0.746 | 0.621 | 0.733 |
| 43 | 0.502 | 0.500 | 0.448 | 0.412 | 0.299 | 0.372 |
| 44 | 0.512 | 0.500 | 0.577 | 0.589 | 0.356 | 0.595 |
| 45 | 0.500 | 0.500 | 0.459 | 0.366 | 0.288 | 0.352 |
| 46 | 0.505 | 0.500 | 0.428 | 0.368 | 0.273 | 0.365 |

### Spearman (descriptive, n = 5)

- heldout_linear vs Maha pad_heldout: ρ = 0.600 (p = 0.285, n = 5)
- heldout_linear vs Maha fitzpatrick17k: ρ = 0.900 (p = 0.0374, n = 5)
- heldout_linear vs Maha mednode: ρ = 0.700 (p = 0.188, n = 5)
- heldout_linear vs Maha sd198: ρ = 0.900 (p = 0.0374, n = 5)
- heldout_mlp vs Maha pad_heldout: ρ = 0.205 (p = 0.741, n = 5)
- heldout_mlp vs Maha fitzpatrick17k: ρ = 0.359 (p = 0.553, n = 5)
- heldout_mlp vs Maha mednode: ρ = 0.359 (p = 0.553, n = 5)
- heldout_mlp vs Maha sd198: ρ = 0.359 (p = 0.553, n = 5)

## Post-commit findings (held-out vs pad_full diverge)

12 cell(s) where held-out and pad_full tell a different story (gap ≥ 0.10, or chance vs non-chance). **pad_full stays primary.**
- a4_main shift=nan λ=0.5: heldout=0.663, pad_full=0.780, gap=0.117
- a4_main shift=nan λ=2.0: heldout=0.670, pad_full=0.786, gap=0.117
- r1_cur shift=0.0 λ=0.5: heldout=0.612, pad_full=0.790, gap=0.178
- r1_cur shift=0.0 λ=2.0: heldout=0.629, pad_full=0.782, gap=0.154
- r1_cur shift=50.0 λ=0.5: heldout=0.531, pad_full=0.629, gap=0.097
- r1_cur shift=50.0 λ=2.0: heldout=0.547, pad_full=0.658, gap=0.111
- r1_cur shift=100.0 λ=0.5: heldout=0.510, pad_full=0.628, gap=0.118
- r1_cur shift=100.0 λ=2.0: heldout=0.510, pad_full=0.634, gap=0.124
- table1 shift=nan λ=0.0: heldout=0.827, pad_full=0.927, gap=0.100
- table1 shift=nan λ=0.25: heldout=0.538, pad_full=0.614, gap=0.076
- table1 shift=nan λ=4.0: heldout=0.516, pad_full=0.636, gap=0.120
- table1 shift=nan λ=8.0: heldout=0.550, pad_full=0.663, gap=0.113

## Status notes

- Fitzpatrick by skin type: `results/paperB/r5/p7a/fitz_skintype/REPORT.md` (not under p7a/REPORT.md).
- P2 / P4 CPU: already done (`r5/p2/REPORT.md`, `r5/p4/REPORT.md`).
- tab:obj: no `pad_heldout.npz` under `phase15/objectives/` — needs frozen extract before probe.
