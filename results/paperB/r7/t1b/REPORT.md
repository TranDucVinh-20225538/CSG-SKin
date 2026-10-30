# Paper B R7 T1b: single-encoder DANN DINOv2-S/14

Design: `decisions/paperB_r7_t1b.md`. Commit `79c7002`.

## Reading-rule outcome (computed first)

invariance not reached (V1=0.884>0.60): test did not take place; W1 stays open

Held-out leakage V1 mean: 0.884. Invariance (≤0.60): False. pad_heldout Rule A: above (k_below=0/5).

| λ | n | Leakage | ID bal acc | ID ECE | OOD ECE (pad_heldout) | OOD AUROC (Maha pad_heldout) | Fitz AUROC (Maha) | X-dom bal acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5 | 0.884 ± 0.019 | 0.555 ± 0.023 | 0.191 ± 0.003 | 0.672 ± 0.034 | 0.546 ± 0.033 | 0.476 ± 0.069 | 0.335 ± 0.025 |

GPU hours: 3.62.

