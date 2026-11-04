# Paper B R10b — bootstrap CIs + table fills (CPU)

Deadline 10 Nov 2026. Branch `paperb/writing-r6`. No `.tex` edits. No training.
Commits authored `vinhtranduc` only (no Co-Authored-By). PREREGISTER: 2,000 draws, rng 20261005.
Rule A = pooled CI upper < 0.5 and ≥ 3 seeds with their own CI upper < 0.5 (degraded rule unchanged).
Each of B1–B4 was its own CPU job (wall 12 h ≫ 40 min). Deviation lines in `decisions/paperB_r10.md`.

## Commit hashes

| item | hash | CPU hours (timer) | SLURM wall |
|---|---|---:|---:|
| launch + deviations | `c7b79ef` | — | — |
| B4 | `db14df3` | 0.075 | 0:04:41 |
| B3 | `6a00d2c` | 0.132 | 0:08:05 |
| B1 | `228f32c` | 0.330 | 0:19:58 |
| B2 | `9d8e460` | 0.841 | 0:50:36 |
| REPORT (this file) | see tip of `paperb/writing-r6` | — | — |

**Total CPU hours (sum of item timers):** 1.379

## B1 — R1 shift 50% bootstrap

Restricted boots to shift 50% × λ ∈ {0.5, 2} × {pad_heldout, fitzpatrick17k, mednode, sd198, scin}.
Seed lists = `{42,52,62,72,82}` for both λ (unchanged vs last full run).
Other R1 cells keep CIs from the last full / `--no_boot` report.
**E1:** R9 E1 refs are shift 0%/100% only — no E1 shift-50 boot; E1 `--no_boot` left unchanged.

Means match the R10 `--no_boot` R1 report; **new** are pooled CI, k, Rule A.

| shift | λ | set | mean±sd | pooled 95% CI | k | Rule A |
|---:|---:|---|---:|---|---:|---|
| 50% | 0.5 | pad_heldout | 0.486±0.047 | [0.463, 0.508] | 3/5 | **not met** |
| 50% | 0.5 | Fitzpatrick17k | 0.455±0.115 | [0.443, 0.468] | 4/5 | met |
| 50% | 0.5 | MED-NODE | 0.312±0.051 | [0.282, 0.344] | 5/5 | met |
| 50% | 0.5 | SD-198 | 0.428±0.104 | [0.417, 0.439] | 4/5 | met |
| 50% | 0.5 | SCIN | 0.567±0.080 | [0.555, 0.579] | 1/5 | not met |
| 50% | 2 | pad_heldout | 0.469±0.027 | [0.448, 0.490] | 4/5 | met |
| 50% | 2 | Fitzpatrick17k | 0.457±0.062 | [0.444, 0.470] | 3/5 | met |
| 50% | 2 | MED-NODE | 0.322±0.054 | [0.294, 0.353] | 5/5 | met |
| 50% | 2 | SD-198 | 0.432±0.047 | [0.421, 0.444] | 4/5 | met |
| 50% | 2 | SCIN | 0.583±0.043 | [0.571, 0.595] | 0/5 | not met |

**Old → new (Rule A):** previously `—` under `--no_boot`. Notable: shift 50% λ=0.5 pad_heldout has k=3/5 but pooled CI upper **0.508 ≥ 0.5 → not met** (would look like a pass on means alone). SCIN fails Rule A at both λ.

LaTeX:
```
50\% & 0.5 & pad_heldout & $0.486\pm0.047$ & $[0.463, 0.508]$ & 3/5 & not met \\
50\% & 0.5 & Fitzpatrick17k & $0.455\pm0.115$ & $[0.443, 0.468]$ & 4/5 & met \\
50\% & 0.5 & MED-NODE & $0.312\pm0.051$ & $[0.282, 0.344]$ & 5/5 & met \\
50\% & 0.5 & SD-198 & $0.428\pm0.104$ & $[0.417, 0.439]$ & 4/5 & met \\
50\% & 0.5 & SCIN & $0.567\pm0.080$ & $[0.555, 0.579]$ & 1/5 & not met \\
50\% & 2 & pad_heldout & $0.469\pm0.027$ & $[0.448, 0.490]$ & 4/5 & met \\
50\% & 2 & Fitzpatrick17k & $0.457\pm0.062$ & $[0.444, 0.470]$ & 3/5 & met \\
50\% & 2 & MED-NODE & $0.322\pm0.054$ & $[0.294, 0.353]$ & 5/5 & met \\
50\% & 2 & SD-198 & $0.432\pm0.047$ & $[0.421, 0.444]$ & 4/5 & met \\
50\% & 2 & SCIN & $0.583\pm0.043$ & $[0.571, 0.595]$ & 0/5 & not met \\
```

## B2 — R2 SCIN with bootstrap

Regenerated `results/paperB/r6/r2/cur/REPORT_SCIN.md` without `--no_boot`.
Means unchanged vs `--no_boot` regenerate; **new** = Maha CIs, k, Rule A.
OOD unit = P7a `group` (case). **Only IWCDAN λ=2 meets Rule A** on SCIN.

| cell | λ | Maha mean±sd [CI] | k | Rule A |
|---|---:|---|---:|---|
| λ=0 (R1 shift100) | 0 | 0.723±0.018 [0.710, 0.736] | 0/5 | not met |
| DANN | 0.5 | 0.534±0.015 [0.521, 0.547] | 0/5 | not met |
| CDAN | 0.5 | 0.540±0.028 [0.526, 0.553] | 0/5 | not met |
| IWDAN | 0.5 | 0.671±0.030 [0.658, 0.684] | 0/5 | not met |
| IWCDAN | 0.5 | 0.536±0.035 [0.523, 0.550] | 0/5 | not met |
| IWDAN_ORACLE | 0.5 | 0.737±0.016 [0.726, 0.749] | 0/5 | not met |
| DANN | 2 | 0.519±0.073 [0.507, 0.532] | 1/5 | not met |
| CDAN | 2 | 0.510±0.056 [0.497, 0.522] | 1/5 | not met |
| IWDAN | 2 | 0.586±0.071 [0.573, 0.599] | 0/5 | not met |
| IWCDAN | 2 | 0.478±0.026 [0.464, 0.491] | 3/5 | **met** |
| IWDAN_ORACLE | 2 | 0.727±0.039 [0.716, 0.739] | 0/5 | not met |

**Old → new:** Maha k / rule A were `—` → filled; IWCDAN λ=2 is the sole Rule A pass among these SCIN cells.

## B3 — A4 rows of tab:unseen

λ as in `r10/C3_unseen.json` ({0, 2}; no λ=0.5 A4 arm). Means unchanged; **added** pooled CI, k, Rule A.

| λ | set | mean±sd | pooled 95% CI | k | Rule A |
|---:|---|---:|---|---:|---|
| 0 | pad_heldout | 0.786±0.019 | [0.770, 0.802] | 0/5 | not met |
| 0 | fitzpatrick17k | 0.821±0.023 | [0.811, 0.831] | 0/5 | not met |
| 0 | mednode | 0.560±0.021 | [0.533, 0.588] | 0/5 | not met |
| 0 | sd198 | 0.799±0.026 | [0.789, 0.810] | 0/5 | not met |
| 0 | scin | 0.822±0.022 | [0.812, 0.832] | 0/5 | not met |
| 2 | pad_heldout | 0.452±0.014 | [0.425, 0.478] | 4/5 | met |
| 2 | fitzpatrick17k | 0.458±0.061 | [0.445, 0.471] | 3/5 | met |
| 2 | mednode | 0.298±0.038 | [0.271, 0.326] | 5/5 | met |
| 2 | sd198 | 0.430±0.061 | [0.419, 0.441] | 5/5 | met |
| 2 | scin | 0.659±0.073 | [0.648, 0.670] | 0/5 | not met |

LaTeX:
```
A4 & 0 & pad_heldout & $0.786\pm0.019$ & $[0.770, 0.802]$ & 0/5 & not met \\
A4 & 0 & fitzpatrick17k & $0.821\pm0.023$ & $[0.811, 0.831]$ & 0/5 & not met \\
A4 & 0 & mednode & $0.560\pm0.021$ & $[0.533, 0.588]$ & 0/5 & not met \\
A4 & 0 & sd198 & $0.799\pm0.026$ & $[0.789, 0.810]$ & 0/5 & not met \\
A4 & 0 & scin & $0.822\pm0.022$ & $[0.812, 0.832]$ & 0/5 & not met \\
A4 & 2 & pad_heldout & $0.452\pm0.014$ & $[0.425, 0.478]$ & 4/5 & met \\
A4 & 2 & fitzpatrick17k & $0.458\pm0.061$ & $[0.445, 0.471]$ & 3/5 & met \\
A4 & 2 & mednode & $0.298\pm0.038$ & $[0.271, 0.326]$ & 5/5 & met \\
A4 & 2 & sd198 & $0.430\pm0.061$ & $[0.419, 0.441]$ & 5/5 & met \\
A4 & 2 & scin & $0.659\pm0.073$ & $[0.648, 0.670]$ & 0/5 & not met \\
```

**Old → new:** means-only → +CI/k/Rule A. At λ=2, primary sets meet Rule A; SCIN does not (CI well above 0.5).

## B4 — SCIN in tab:splits

Splits {7, 123, 2027} × λ ∈ {0, 2}; last.ckpt features (3 seeds); OOD unit = P7a `group` (same as SCIN elsewhere / REPORT_SCIN; deviation vs work-order wording “image resampling”).

| split | λ | mean±sd | pooled 95% CI | k | Rule A |
|---:|---:|---:|---|---:|---|
| 7 | 0 | 0.733±0.042 | [0.720, 0.745] | 0/3 | not met |
| 7 | 2 | 0.498±0.032 | [0.486, 0.511] | 1/3 | not met |
| 123 | 0 | 0.700±0.007 | [0.686, 0.712] | 0/3 | not met |
| 123 | 2 | 0.508±0.040 | [0.494, 0.521] | 1/3 | not met |
| 2027 | 0 | 0.734±0.029 | [0.721, 0.746] | 0/3 | not met |
| 2027 | 2 | 0.541±0.081 | [0.529, 0.554] | 1/3 | not met |

LaTeX:
```
7 & 0 & SCIN & $0.733\pm0.042$ & $[0.720, 0.745]$ & 0/3 & not met \\
7 & 2 & SCIN & $0.498\pm0.032$ & $[0.486, 0.511]$ & 1/3 & not met \\
123 & 0 & SCIN & $0.700\pm0.007$ & $[0.686, 0.712]$ & 0/3 & not met \\
123 & 2 & SCIN & $0.508\pm0.040$ & $[0.494, 0.521]$ & 1/3 & not met \\
2027 & 0 & SCIN & $0.734\pm0.029$ & $[0.721, 0.746]$ & 0/3 & not met \\
2027 & 2 & SCIN & $0.541\pm0.081$ & $[0.529, 0.554]$ & 1/3 & not met \\
```

**Old → new:** SCIN absent from these split-stability cells → filled. No cell meets Rule A (with n=3 seeds, Rule A needs all 3 seed CIs below 0.5; split 7 λ=2 mean is near 0.5 but CI upper 0.511).

## Not run

tab:single-dino ERM reference: already in `results/paperB/r7/t1/summary.csv` (ERM DINOv2-S). No run.
