# Paper B R11 X4 — BCN → HAM

Design: `decisions/paperB_r11_x4.md`. Commit `28042bc`.

## TV class priors (report first)

- TV(BCN train, HAM adv) = 0.3608264845276012.
- TV(BCN train, HAM heldout) = 0.35191212874567324.
- ham_heldout lesions frac target ≥ 20% (split uses 30%).

## Cells

| λ | shift | ID bal | V1 | RuleA Fitz | med | SD | ham_heldout | n |
|---:|---:|---|---|---|---|---|---|---:|
| 0 | 100% | 0.362 ± 0.007 | 0.809 ± 0.017 | False | False | False | False | 5 |
| 2 | 0% | 0.365 ± 0.027 | 0.671 ± 0.012 | False | False | False | True | 5 |
| 2 | 100% | 0.366 ± 0.017 | 0.599 ± 0.041 | False | False | False | True | 5 |

## Verdict (λ = 2)

- **not_reproduced**: met on fewer than 2 at shift 100%; not reproduced in this pair.
- V1 = 0.5991062221769926; hits@100 = []; hits@0 = [].

ham_heldout and SCIN reported, not counted.

