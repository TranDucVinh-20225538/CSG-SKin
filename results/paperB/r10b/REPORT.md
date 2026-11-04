# Paper B R10b — bootstrap CIs + table fills (CPU)

Deadline 10 Nov 2026. Branch `paperb/writing-r6`. No `.tex` edits. No training.
Commits authored `vinhtranduc` only. PREREGISTER: 2,000 draws, rng 20261005.
Rule A = pooled CI upper < 0.5 and ≥ 3 seeds with their own CI upper < 0.5.

## Commit hashes

| item | hash | CPU hours |
|---|---|---:|
| B1 | `228f32c26ca619e61690157613910c20488b2f72` | 0.330 |
| B2 | `9d8e4606ef4107f7c5fa01c21eeb9094558d6c24` | 0.841 |
| B3 | `6a00d2c976dac62deba01bdc2dee84cdeac4211e` | 0.132 |
| B4 | `db14df3be325da4ad5d233c71c8f808be57dd532` | 0.075 |
| REPORT | `4d2224531950ac70997e7bce67acf1ba70f761a1` | — |

**Total CPU hours (sum of item timers):** 1.379

## B1 — R1 shift 50% bootstrap

R9 E1 references are shift 0% and shift 100% only; no shift-50% boots required. E1 --no_boot from R10 left unchanged.

Seed lists checked against `{42,52,62,72,82}`:
```
{
 "shift 50%|0.5": [
  42,
  52,
  62,
  72,
  82
 ],
 "shift 50%|2": [
  42,
  52,
  62,
  72,
  82
 ]
}
```

LaTeX rows (shift 50% cells):
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

Old → new: cells previously had means only / `—` CIs under R10 `--no_boot`; Rule A column newly filled. See `B1_shift50_boots.md` for the table.

## B2 — R2 SCIN with bootstrap

Regenerated `results/paperB/r6/r2/cur/REPORT_SCIN.md` (commit at run: `228f32c`).
Old → new: Maha k<0.5 and rule A columns were `—` under `--no_boot`; now filled.

## B3 — A4 rows of tab:unseen

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

Old → new: means-only A4 rows from R10 C3 gain pooled CI, k, Rule A.

## B4 — SCIN in tab:splits

OOD unit: `P7a group column` (same as SCIN elsewhere / REPORT_SCIN).

LaTeX:
```
7 & 0 & SCIN & $0.733\pm0.042$ & $[0.720, 0.745]$ & 0/3 & not met \\
7 & 2 & SCIN & $0.498\pm0.032$ & $[0.486, 0.511]$ & 1/3 & not met \\
123 & 0 & SCIN & $0.700\pm0.007$ & $[0.686, 0.712]$ & 0/3 & not met \\
123 & 2 & SCIN & $0.508\pm0.040$ & $[0.494, 0.521]$ & 1/3 & not met \\
2027 & 0 & SCIN & $0.734\pm0.029$ & $[0.721, 0.746]$ & 0/3 & not met \\
2027 & 2 & SCIN & $0.541\pm0.081$ & $[0.529, 0.554]$ & 1/3 & not met \\
```

Old → new: SCIN absent from split-stability table cells → filled for splits {7,123,2027} × λ ∈ {0,2}.
