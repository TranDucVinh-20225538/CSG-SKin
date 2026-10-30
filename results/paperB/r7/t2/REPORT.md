# Paper B R7 T2: pad_adv size

Design: `decisions/paperB_r7_t2.md`. Commit `79c7002`.

pad_adv subsample feeds **both** adversarial and context branches; pad_heldout unchanged. With 3 seeds, Rule A means all 3 seed CI uppers < 0.5.

| arm | λ | n | Leakage | ID bal acc | ID ECE | OOD ECE (pad_heldout) | OOD AUROC (Maha pad_heldout) | Fitz AUROC (Maha) | X-dom bal acc | pad Rule A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pad_adv 25% | 0 | 3 | 0.898 ± 0.032 | 0.472 ± 0.019 | 0.216 ± 0.009 | 0.299 ± 0.009 | 0.819 ± 0.022 | 0.724 ± 0.043 | 0.291 ± 0.034 | above |
| pad_adv 25% | 2 | 3 | 0.632 ± 0.007 | 0.477 ± 0.002 | 0.207 ± 0.006 | 0.720 ± 0.012 | 0.479 ± 0.013 | 0.407 ± 0.011 | 0.264 ± 0.018 | C |
| pad_adv 50% | 2 | 3 | 0.575 ± 0.030 | 0.459 ± 0.022 | 0.207 ± 0.002 | 0.744 ± 0.018 | 0.415 ± 0.043 | 0.397 ± 0.069 | 0.257 ± 0.025 | A |

Class counts per subsample (from first run of each frac):

- 25%: n_images=407, n_patients=240, counts={'MEL': 9, 'NV': 36, 'BCC': 163, 'AK': 128, 'BKL': 41, 'DF': 0, 'VASC': 0, 'SCC': 30}
- 50%: n_images=802, n_patients=481, counts={'MEL': 21, 'NV': 80, 'BCC': 282, 'AK': 268, 'BKL': 86, 'DF': 0, 'VASC': 0, 'SCC': 65}

GPU hours: 10.29.

