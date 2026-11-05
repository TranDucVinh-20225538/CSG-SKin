# Paper B R11 — reproduction (a, b) and arXiv extensions X1–X5

Branch `paperb/writing-r6`. Commits `vinhtranduc` only. Tip at start: `8356383`.

## Level a — rule rebuild

Command: `python scripts/reproduce_rule_a.py`

```
checked 174 cells; 0 mismatch(es)
```

## Level b — bootstrap redrawn from cached features

_(re-running after JSON `int64` serialize fix; first attempt FAILED at write of `BOOTSTRAP.json`)_

## X1 — OOD split by label space

Decisions: `decisions/paperB_r11_x1.md`. Commit `6a7cb62`. CPU h 0.434.

**Counts:** Fitz in/out/amb = 586 / 3241 / 60 (n=3887). SD-198 = 525 / 5950 / 109 (n=6584).

**Cases (λ = 2 decides):**

- Fitzpatrick17k: **Not label space** (Rule A met on both in-ISIC and out-of-ISIC)
- SD-198: **Not label space** (Rule A met on both subsets)

Implication: inversion is **not** because OOD is outside the ISIC label space — drop “SCIN because out-of-label.” Full tables: `results/paperB/r11/x1/X1.md`.

## X2 — OOD monitor on untouched features

_(pending; decisions locked)_

## X3 — adversary on head over frozen ERM

_(pending; decisions locked)_

## X4 — BCN → HAM

_(pending; decisions locked)_

## X5 — semi-synthetic controlled label shift

_(pending; decisions locked)_

## GPU / CPU hours and commit hashes

| item | hash | note |
|---|---|---|
| launch + a + X1 decisions + GPU est | `9c16f39` | |
| X2–X5 decisions | `1b86d07` | |
| X1 results | `6a7cb62` | |
| level b | _(pending retry)_ | |

See `results/paperB/r11/GPU_ESTIMATE.md` (≈26–31 GPU-h total; issued before first GPU job).
