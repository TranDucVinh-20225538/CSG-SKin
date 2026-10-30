# Paper B R7 T1: dual DINOv2 + ERM reference

Design: `decisions/paperB_r7_t1.md`. Commit `79c7002`.

## Reading-rule case (computed first)

**Case C.** gap=0.095 (near=False), pad Rule A=True, labelled-unseen Rule A count=3, leakage λ=2=0.576.

Competence gap = ERM-DINOv2 mean ID bal − dual λ=0 mean ID bal = 0.565 − 0.470 = 0.095 (near-ERM iff ≤ 0.03).
EfficientNet-B3 ERM reference (work order): 0.558 ± 0.022; dual λ=0 vs that gap = 0.088.
Leakage at λ=2 (mean): 0.576.

| arm | λ | n | Leakage | ID bal acc | ID ECE | OOD ECE (pad_heldout) | OOD AUROC (Maha pad_heldout) | Fitz AUROC (Maha) | X-dom bal acc | pad rule | Fitz rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| ERM dinov2_vits14 | 0 | 5 | 0.983 ± 0.002 | 0.565 ± 0.015 | 0.195 ± 0.007 | 0.426 ± 0.018 | 0.591 ± 0.060 | 0.704 ± 0.059 | 0.323 ± 0.028 | above | above |
| dual dinov2_vits14 | 0 | 5 | 0.947 ± 0.007 | 0.470 ± 0.015 | 0.203 ± 0.005 | 0.172 ± 0.034 | 0.831 ± 0.029 | 0.778 ± 0.016 | 0.246 ± 0.027 | above | above |
| dual dinov2_vits14 | 2 | 5 | 0.576 ± 0.034 | 0.462 ± 0.022 | 0.204 ± 0.014 | 0.725 ± 0.033 | 0.448 ± 0.045 | 0.472 ± 0.067 | 0.229 ± 0.032 | A | A |

### Table 2 (unseen Mahalanobis)

| arm | λ | mednode | sd198 | scin |
|---|---:|---:|---:|---:|
| ERM dinov2_vits14 | 0 | 0.452 ± 0.033 | 0.702 ± 0.053 | 0.781 ± 0.074 |
| dual dinov2_vits14 | 0 | 0.580 ± 0.030 | 0.790 ± 0.005 | 0.716 ± 0.041 |
| dual dinov2_vits14 | 2 | 0.363 ± 0.045 | 0.461 ± 0.066 | 0.551 ± 0.064 |

```latex
ERM dinov2_vits14 $\lambda=0$ & 0.983\,$\pm$\,0.002 & 0.565\,$\pm$\,0.015 & 0.195\,$\pm$\,0.007 & 0.426\,$\pm$\,0.018 & 0.591\,$\pm$\,0.060 & 0.704\,$\pm$\,0.059 & 0.323\,$\pm$\,0.028 \\
dual dinov2_vits14 $\lambda=0$ & 0.947\,$\pm$\,0.007 & 0.470\,$\pm$\,0.015 & 0.203\,$\pm$\,0.005 & 0.172\,$\pm$\,0.034 & 0.831\,$\pm$\,0.029 & 0.778\,$\pm$\,0.016 & 0.246\,$\pm$\,0.027 \\
dual dinov2_vits14 $\lambda=2$ & 0.576\,$\pm$\,0.034 & 0.462\,$\pm$\,0.022 & 0.204\,$\pm$\,0.014 & 0.725\,$\pm$\,0.033 & 0.448\,$\pm$\,0.045 & 0.472\,$\pm$\,0.067 & 0.229\,$\pm$\,0.032 \\
```

GPU hours (sum of run summaries): 11.44.

## Plain paragraph

Case C: gap=0.095 (near=False), pad Rule A=True, labelled-unseen Rule A count=3, leakage λ=2=0.576. Dual DINOv2 λ=0 ID balanced accuracy 0.470 ± 0.015; ERM-DINOv2 0.565 ± 0.015; competence gap 0.095; EffB3-ERM reference 0.558 ± 0.022.

