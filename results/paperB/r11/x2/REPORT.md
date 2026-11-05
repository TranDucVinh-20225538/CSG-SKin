# Paper B R11 X2 — OOD monitors M1 / M2 / M3

Design: `decisions/paperB_r11_x2.md`. Commit `d5942f8`.

Monitors: **M1** dual latent; **M2** ERM EffB3 (same seed); **M3** ImageNet EffB3 frozen (MSP/Energy N/A — no skin head). Primary = Mahalanobis. Rule A = pooled CI upper < 0.5 and k≥3.
Reading: avoids inversion at λ=2 if Rule A on no labelled set and AUROC within 0.03 of own λ=0 (M3: first condition only).

## Cells (Mahalanobis)

| monitor | λ | OOD | AUROC mean±sd | pooled CI | k | n | Rule A | FPR@95TPR |
|---|---:|---|---|---|---:|---:|---|---|
| M1 | 0 | pad_heldout | 0.825 ± 0.022 | [0.810, 0.839] | 0 | 5 | False | 0.911 ± 0.060 |
| M1 | 0 | fitzpatrick17k | 0.712 ± 0.013 | [0.698, 0.725] | 0 | 5 | False | 0.952 ± 0.020 |
| M1 | 0 | mednode | 0.559 ± 0.019 | [0.531, 0.587] | 0 | 5 | False | 0.989 ± 0.011 |
| M1 | 0 | sd198 | 0.708 ± 0.019 | [0.695, 0.722] | 0 | 5 | False | 0.960 ± 0.021 |
| M1 | 0 | scin | 0.723 ± 0.018 | [0.710, 0.736] | 0 | 5 | False | 0.949 ± 0.009 |
| M1 | 0.5 | pad_heldout | 0.415 ± 0.016 | [0.390, 0.438] | 5 | 5 | True | 0.961 ± 0.009 |
| M1 | 0.5 | fitzpatrick17k | 0.377 ± 0.046 | [0.364, 0.390] | 5 | 5 | True | 0.978 ± 0.008 |
| M1 | 0.5 | mednode | 0.309 ± 0.031 | [0.282, 0.335] | 5 | 5 | True | 0.999 ± 0.003 |
| M1 | 0.5 | sd198 | 0.351 ± 0.039 | [0.340, 0.362] | 5 | 5 | True | 0.983 ± 0.008 |
| M1 | 0.5 | scin | 0.529 ± 0.038 | [0.516, 0.542] | 0 | 5 | False | 0.970 ± 0.012 |
| M1 | 2 | pad_heldout | 0.381 ± 0.023 | [0.358, 0.404] | 5 | 5 | True | 0.961 ± 0.007 |
| M1 | 2 | fitzpatrick17k | 0.405 ± 0.031 | [0.392, 0.419] | 5 | 5 | True | 0.978 ± 0.005 |
| M1 | 2 | mednode | 0.288 ± 0.027 | [0.262, 0.315] | 5 | 5 | True | 0.999 ± 0.003 |
| M1 | 2 | sd198 | 0.376 ± 0.026 | [0.364, 0.387] | 5 | 5 | True | 0.983 ± 0.005 |
| M1 | 2 | scin | 0.551 ± 0.039 | [0.538, 0.564] | 0 | 5 | False | 0.965 ± 0.015 |
| M2 | 0 | pad_heldout | 0.708 ± 0.038 | [0.688, 0.728] | 0 | 5 | False | 0.946 ± 0.021 |
| M2 | 0 | fitzpatrick17k | 0.802 ± 0.049 | [0.790, 0.814] | 0 | 5 | False | 0.825 ± 0.064 |
| M2 | 0 | mednode | 0.518 ± 0.053 | [0.491, 0.545] | 1 | 5 | False | 0.991 ± 0.005 |
| M2 | 0 | sd198 | 0.785 ± 0.054 | [0.773, 0.797] | 0 | 5 | False | 0.828 ± 0.073 |
| M2 | 0 | scin | 0.834 ± 0.049 | [0.823, 0.844] | 0 | 5 | False | 0.770 ± 0.070 |
| M2 | 0.5 | pad_heldout | 0.654 | [0.631, 0.677] | 0 | 1 | False | 0.978 |
| M2 | 0.5 | fitzpatrick17k | 0.716 | [0.700, 0.732] | 0 | 1 | False | 0.933 |
| M2 | 0.5 | mednode | 0.490 | [0.456, 0.526] | 0 | 1 | False | 0.994 |
| M2 | 0.5 | sd198 | 0.689 | [0.673, 0.705] | 0 | 1 | False | 0.941 |
| M2 | 0.5 | scin | 0.748 | [0.733, 0.762] | 0 | 1 | False | 0.890 |
| M2 | 2 | pad_heldout | 0.708 ± 0.038 | [0.688, 0.728] | 0 | 5 | False | 0.946 ± 0.021 |
| M2 | 2 | fitzpatrick17k | 0.802 ± 0.049 | [0.790, 0.814] | 0 | 5 | False | 0.825 ± 0.064 |
| M2 | 2 | mednode | 0.518 ± 0.053 | [0.491, 0.545] | 1 | 5 | False | 0.991 ± 0.005 |
| M2 | 2 | sd198 | 0.785 ± 0.054 | [0.773, 0.797] | 0 | 5 | False | 0.828 ± 0.073 |
| M2 | 2 | scin | 0.834 ± 0.049 | [0.823, 0.844] | 0 | 5 | False | 0.770 ± 0.070 |
| M3 | 0 | pad_heldout | 0.792 | [0.770, 0.812] | 0 | 1 | False | 0.753 |
| M3 | 0 | fitzpatrick17k | 0.953 | [0.947, 0.959] | 0 | 1 | False | 0.231 |
| M3 | 0 | mednode | 0.740 | [0.704, 0.776] | 0 | 1 | False | 0.800 |
| M3 | 0 | sd198 | 0.943 | [0.935, 0.949] | 0 | 1 | False | 0.254 |
| M3 | 0 | scin | 0.973 | [0.969, 0.977] | 0 | 1 | False | 0.132 |
| M3 | 0.5 | pad_heldout | 0.792 | [0.770, 0.812] | 0 | 1 | False | 0.753 |
| M3 | 0.5 | fitzpatrick17k | 0.953 | [0.947, 0.959] | 0 | 1 | False | 0.231 |
| M3 | 0.5 | mednode | 0.740 | [0.704, 0.776] | 0 | 1 | False | 0.800 |
| M3 | 0.5 | sd198 | 0.943 | [0.935, 0.949] | 0 | 1 | False | 0.254 |
| M3 | 0.5 | scin | 0.973 | [0.969, 0.977] | 0 | 1 | False | 0.132 |
| M3 | 2 | pad_heldout | 0.792 | [0.770, 0.812] | 0 | 1 | False | 0.753 |
| M3 | 2 | fitzpatrick17k | 0.953 | [0.947, 0.959] | 0 | 1 | False | 0.231 |
| M3 | 2 | mednode | 0.740 | [0.704, 0.776] | 0 | 1 | False | 0.800 |
| M3 | 2 | sd198 | 0.943 | [0.935, 0.949] | 0 | 1 | False | 0.254 |
| M3 | 2 | scin | 0.973 | [0.969, 0.977] | 0 | 1 | False | 0.132 |

## Avoids inversion? (λ=2 vs λ=0, labelled sets)

- **M1**: Rule A hits = ['pad_heldout', 'fitzpatrick17k', 'mednode', 'sd198']; |AUROC_λ2−λ0|≤0.03 on labelled = False; avoids_inversion = False. (pad_heldout Δ=0.444, fitzpatrick17k Δ=0.307, mednode Δ=0.271, sd198 Δ=0.333)
- **M2**: Rule A hits = none; |AUROC_λ2−λ0|≤0.03 on labelled = True; avoids_inversion = True. (pad_heldout Δ=0.000, fitzpatrick17k Δ=0.000, mednode Δ=0.000, sd198 Δ=0.000)
- **M3**: Rule A on labelled = none; avoids_inversion = True (λ-independent by construction).

## Clinical: λ=2 classifier, MEL → pred NV conf>0.9, share flagged at 95% TPR

| monitor | OOD | n_nv09 (mean) | share flagged mean±sd | n_seeds |
|---|---|---:|---|---:|
| M1 | fitzpatrick17k | 42.4 | 0.000 ± 0.000 | 5 |
| M1 | sd198 | 92.0 | 0.002 ± 0.005 | 5 |
| M2 | fitzpatrick17k | 42.4 | 0.046 ± 0.047 | 5 |
| M2 | sd198 | 92.0 | 0.037 ± 0.022 | 5 |
| M3 | fitzpatrick17k | 42.4 | 0.382 ± 0.020 | 5 |
| M3 | sd198 | 92.0 | 0.335 ± 0.016 | 5 |

## Degraded ID bal_acc < 0.25

None.

## Notes

- λ=0.5 × M2: only seed 42 pairs with erm_ref.
- M3: single feature bank; n_seeds=1 for AUROC cells; MSP/Energy omitted.
- SCIN included for completeness; X1 = Not label space.

