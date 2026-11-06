# Paper B R11 X3 — adversary on head over frozen ERM

Design: `decisions/paperB_r11_x3.md`. Commit `778b1f1`.

Frozen ERM EffB3 features (no aug). Head: 2-layer MLP → 16-d + classifier + derm adversary/GRL. 40 epochs, last.ckpt. ERM ref ID bal = 0.558; competent ≥ 0.528; invariant V1 ≤ 0.60.

## Cells

| λ | shift | ID bal | V1 | Rule A pad | Rule A Fitz | Rule A med | Rule A SD | n |
|---:|---:|---|---|---|---|---|---|---:|
| 0.5 | 100% | 0.507 ± 0.019 | 0.813 ± 0.043 | False | False | False | False | 5 |
| 0 | 100% | 0.523 ± 0.008 | 0.613 ± 0.043 | False | False | False | False | 5 |
| 2 | 100% | 0.523 ± 0.012 | 0.767 ± 0.033 | False | False | False | False | 5 |
| 8 | 100% | 0.506 ± 0.019 | 0.782 ± 0.028 | False | False | False | False | 5 |
| 2 | 0% | 0.492 ± 0.010 | 0.813 ± 0.077 | False | False | False | False | 5 |

## Verdict (λ = 2, shift 100 %)

- **Case C**: invariance not reached; settles nothing. Stop after λ grid..
- competent = None; invariant = False; ID bal = None; V1 = None.
- Rule A pad = None; unseen hits = None.

## Notes

- Cache: `results/paperB/r11/x3/cache/` (symlinks to erm_ref; no augmentation).
- λ=0.5 written as ladv0p5 in tags.

