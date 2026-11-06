# Paper B R11 — reproduction (a, b) and arXiv extensions X1–X5

Branch `paperb/writing-r6`. Commits `vinhtranduc` only. Tip at start: `8356383`.

## Level a — rule rebuild

Command: `python scripts/reproduce_rule_a.py`

```
checked 174 cells; 0 mismatch(es)
```

## Level b — bootstrap redrawn from cached features

ok=True. CPU h 2.089. Tolerance 1e-9.

| file | n compared | n differing |
|---|---:|---:|
| `r5/t1last/BOOTSTRAP.json` | 448 | 0 |
| `r5/t1last/UNSEEN_ECE.json` | 210 | 0 |
| `r10b/B1_shift50_boots.json` | 60 | 0 |
| `r10b/B3_a4_ci.json` | 60 | 0 |
| `r10b/B4_splits_scin.json` | 90 | 0 |
| `r6/r2/cur/scin_report.json` | 176 | 0 |

Differing values: **None**. `reproduce_rule_a.py` on redrawn scratch also: `checked 174 cells; 0 mismatch(es)`.

## X1 — OOD split by label space

Decisions: `decisions/paperB_r11_x1.md`. Commit `6a7cb62`. CPU h 0.434.

**Counts:** Fitz in/out/amb = 586 / 3241 / 60 (n=3887). SD-198 = 525 / 5950 / 109 (n=6584).

**Cases (λ = 2 decides):**

- Fitzpatrick17k: **Not label space** (Rule A met on both in-ISIC and out-of-ISIC)
- SD-198: **Not label space** (Rule A met on both subsets)

Implication: inversion is **not** because OOD is outside the ISIC label space — drop “SCIN because out-of-label.” Full tables: `results/paperB/r11/x1/X1.md`.

## X2 — OOD monitor on untouched features

Decisions: `decisions/paperB_r11_x2.md`. Forwards 72821 + score 72926 COMPLETED. Full: `results/paperB/r11/x2/REPORT.md`.

- **M1**: Rule A hits = ['pad_heldout', 'fitzpatrick17k', 'mednode', 'sd198']; |AUROC_λ2−λ0|≤0.03 on labelled = False; avoids_inversion = False. (pad_heldout Δ=0.444, fitzpatrick17k Δ=0.307, mednode Δ=0.271, sd198 Δ=0.333)
- **M2**: Rule A hits = none; |AUROC_λ2−λ0|≤0.03 on labelled = True; avoids_inversion = True. (pad_heldout Δ=0.000, fitzpatrick17k Δ=0.000, mednode Δ=0.000, sd198 Δ=0.000)
- **M3**: Rule A on labelled = none; avoids_inversion = True (λ-independent by construction).

## X3 — adversary on head over frozen ERM

Decisions: `decisions/paperB_r11_x3.md`. Train 72999 + score 73000 COMPLETED. Full: `results/paperB/r11/x3/REPORT.md`.

- **Case C**: invariance not reached; settles nothing. Stop after λ grid..
- competent=None; invariant=False; ID bal=None; V1=None.

## X4 — BCN → HAM

Decisions: `decisions/paperB_r11_x4.md`. GPU train queued.

- Train `73068` (15× %2, ~1.4 GPU-h/run); score `73069` afterok. Dual CSG BCN→HAM.

## X5 — semi-synthetic controlled label shift

Decisions: `decisions/paperB_r11_x5.md`. Jobs 73027–73029 COMPLETED. Full: `results/paperB/r11/x5/REPORT.md`.

- **reproduces_inversion = False** (t=1 hits []; t=0 hits []).

## GPU / CPU hours and commit hashes

| item | hash | CPU h |
|---|---|---:|
| launch + a + X1 decisions + GPU est | `9c16f39` | — |
| X2–X5 decisions | `1b86d07` | — |
| X1 results | `6a7cb62` | 0.434 |
| X1 reading note | `ec1e2f2` | — |
| level b | `7b6ccde` | 2.089 |
| REPORT update | tip after this commit | — |

See `results/paperB/r11/GPU_ESTIMATE.md` (≈26–31 GPU-h total; issued before first GPU job).
