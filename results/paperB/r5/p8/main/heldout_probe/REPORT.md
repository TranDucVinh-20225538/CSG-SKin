# Held-out leakage (P6 V1, z_lesion_norm): P8 vs current-recipe dual encoder at the same λ

Same probe function and protocol for every row (V1: ISIC test ∪ pad_heldout, 70/30 GroupShuffleSplit by lesion / patient, probe seeds 42/52/62). Chance = 0.5; lower = less domain information.

| λ | model | checkpoint | n | epochs | V1 held-out leakage | P6 reference (Table 1) |
|---:|---|---|---:|---|---|---|
| 0 | P8 (A4) | best-val | 5 | 1–7 | 0.858 ± 0.014 | — |
| 0 | P8 (A4) | last | 5 | 39–39 | 0.842 ± 0.025 | — |
| 0 | Table 1 (current recipe) | best-val | 5 | 15–39 | 0.827 ± 0.056 | 0.827 ± 0.056 |
| 0.5 | P8 (A4) | best-val | 5 | 3–9 | 0.728 ± 0.029 | — |
| 0.5 | P8 (A4) | last | 5 | 39–39 | 0.663 ± 0.017 | — |
| 0.5 | Table 1 (current recipe) | best-val | 5 | 26–33 | 0.506 ± 0.007 | 0.506 ± 0.007 |
| 2 | P8 (A4) | best-val | 5 | 2–19 | 0.696 ± 0.050 | — |
| 2 | P8 (A4) | last | 5 | 39–39 | 0.670 ± 0.038 | — |
| 2 | Table 1 (current recipe) | best-val | 5 | 20–39 | 0.502 ± 0.003 | 0.502 ± 0.003 |

LaTeX rows (λ & model & checkpoint & n & V1):

```
0 & P8 (A4) & best-val & 5 & 0.858\,$\pm$\,0.014 \\
0 & P8 (A4) & last & 5 & 0.842\,$\pm$\,0.025 \\
0 & Table 1 (current recipe) & best-val & 5 & 0.827\,$\pm$\,0.056 \\
0.5 & P8 (A4) & best-val & 5 & 0.728\,$\pm$\,0.029 \\
0.5 & P8 (A4) & last & 5 & 0.663\,$\pm$\,0.017 \\
0.5 & Table 1 (current recipe) & best-val & 5 & 0.506\,$\pm$\,0.007 \\
2 & P8 (A4) & best-val & 5 & 0.696\,$\pm$\,0.050 \\
2 & P8 (A4) & last & 5 & 0.670\,$\pm$\,0.038 \\
2 & Table 1 (current recipe) & best-val & 5 & 0.502\,$\pm$\,0.003 \\
```

