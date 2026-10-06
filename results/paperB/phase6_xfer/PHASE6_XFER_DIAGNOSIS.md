# Item 1 — pad_heldout 6-class transfer diagnosis (array 60585)

Source: `results/paperB/phase6_xfer/*.json` (37 files: 27 CSG + 5 baseline_soft + 5 EffB3). Job `60585` (`eval_final_gpu.py`). Written 2026-09-21 20:33–20:38. Numbers as measured. Do not retune.

**Protocol.** pad_heldout only (716 images / 412 patients). Softmax and argmax restricted to MEL, NV, BCC, AK, BKL, SCC. DF and VASC are absent from PAD. PAD labels never entered `L_cls` (`ignore_index=-1`): zero-shot. Phase 3 `pad_acc` was `pad_full` and is contaminated; it is not used.

**Class counts (fixed).** MEL 9, NV 88, BCC 276, AK 202, BKL 75, SCC 66.

**Floors, so this column does not repeat the leakage-scale error.**

| Metric | Floor | Value | Why |
|---|---|---:|---|
| 6-class balanced accuracy | uniform chance | 1/6 ≈ 0.167 | Mean of per-class recalls. A constant predictor scores 1 on one class and 0 on the other five. |
| 6-class plain accuracy | majority (always BCC) | 276/716 = 0.385 | Not the floor for the balanced-accuracy column in the dial. |

---

## Locked cost sentence

Cross-domain diagnosis is **flat across the cliff and then falls**. The abstract may write:

> while cross-domain accuracy does not improve at all

That is the strong form: you pay a safety cost and get nothing. It is not a trade-off.

| Contrast | λ=0 (n=5) | λ=0.25 (n=3) | Δ |
|---|---|---|---|
| 6-class balanced acc (dial column) | 0.291 ± 0.016 | 0.291 ± 0.020 | 0.000 (F_cliff CI [−0.022, 0.022], Holm no) |
| 6-class plain acc | 0.245 ± 0.033 | 0.215 ± 0.009 | −0.030 (falls) |
| Macro AUC (OVR) | 0.605 ± 0.013 | 0.621 ± 0.026 | +0.016 (flat) |

Mean λ>0 balanced acc = 0.267. λ=2 = 0.249 ± 0.030. No λ>0 cell beats λ=0.

---

## Headline table (CSG)

| λ_adv | n | plain acc | bal acc | macro AUC | MEL rec | NV rec | BCC rec | AK rec | BKL rec | SCC rec |
|---:|---:|---|---|---|---|---|---|---|---|---|
| 0 | 5 | 0.245 ± 0.033 | 0.291 ± 0.016 | 0.605 ± 0.013 | 0.489 ± 0.149 | 0.498 ± 0.026 | 0.199 ± 0.085 | 0.245 ± 0.102 | 0.229 ± 0.042 | 0.088 ± 0.025 |
| 0.25 | 3 | 0.215 ± 0.009 | 0.291 ± 0.020 | 0.621 ± 0.026 | 0.444 ± 0.111 | 0.894 ± 0.033 | 0.186 ± 0.018 | 0.028 ± 0.030 | 0.178 ± 0.034 | 0.015 ± 0.015 |
| 0.5 | 3 | 0.187 ± 0.007 | 0.270 ± 0.034 | 0.621 ± 0.012 | 0.407 ± 0.231 | 0.905 ± 0.026 | 0.134 ± 0.016 | 0.007 ± 0.003 | 0.133 ± 0.053 | 0.030 ± 0.015 |
| 1 | 3 | 0.189 ± 0.012 | 0.261 ± 0.016 | 0.592 ± 0.020 | 0.370 ± 0.064 | 0.867 ± 0.013 | 0.145 ± 0.013 | 0.020 ± 0.013 | 0.138 ± 0.031 | 0.025 ± 0.017 |
| 2 | 5 | 0.184 ± 0.002 | 0.249 ± 0.030 | 0.581 ± 0.019 | 0.311 ± 0.199 | 0.884 ± 0.012 | 0.135 ± 0.020 | 0.016 ± 0.012 | 0.125 ± 0.028 | 0.021 ± 0.008 |
| 4 | 3 | 0.191 ± 0.013 | 0.276 ± 0.025 | 0.614 ± 0.010 | 0.407 ± 0.170 | 0.871 ± 0.017 | 0.130 ± 0.029 | 0.017 ± 0.008 | 0.196 ± 0.015 | 0.035 ± 0.017 |
| 8 | 5 | 0.193 ± 0.014 | 0.269 ± 0.023 | 0.603 ± 0.015 | 0.378 ± 0.127 | 0.866 ± 0.033 | 0.149 ± 0.033 | 0.009 ± 0.004 | 0.189 ± 0.032 | 0.021 ± 0.017 |

Balanced accuracy being flat at 0→0.25 **hides a collapse onto NV**. Mean predicted counts:

| λ_adv | n | pred MEL | pred NV | pred BCC | pred AK | pred BKL | pred SCC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 152 | 113 | 106 | 138 | 134 | 72 |
| 0.25 | 3 | 163 | **374** | 86 | **12** | 74 | **7** |
| 2 | 5 | 157 | **426** | 65 | **10** | 50 | **7** |
| truth | — | 9 | 88 | 276 | 202 | 75 | 66 |

At λ≥0.25 the model is a nevus predictor. AK recall 0.245 → 0.028. SCC recall 0.088 → 0.015. NV recall 0.498 → 0.894. That is why balanced accuracy can stay at 0.291 while plain accuracy falls (0.245 → 0.215) and the prediction histogram leaves the two largest PAD classes (BCC, AK).

CSG plain accuracy at every λ is **below** the majority floor 0.385. Balanced accuracy 0.291 is above 1/6 ≈ 0.167 and does not rise with λ.

---

## Melanoma sensitivity at specificity 0.85

pad_heldout contains **9 melanoma images**. No significance claim. Direction is down.

| λ_adv | n | sens@spec 0.85 |
|---:|---:|---|
| 0 | 5 | 0.489 ± 0.099 |
| 0.25 | 3 | 0.333 ± 0.111 |
| 0.5 | 3 | 0.259 ± 0.064 |
| 1 | 3 | 0.296 ± 0.064 |
| 2 | 5 | 0.222 ± 0.176 |
| 4 | 3 | 0.259 ± 0.064 |
| 8 | 5 | 0.289 ± 0.127 |

---

## Controls (same protocol, not on the λ dial)

| Method | n | plain acc | bal acc | macro AUC | MEL rec | NV rec |
|---|---:|---|---|---|---|---|
| baseline_soft (ResNet-50) | 5 | 0.336 ± 0.046 | 0.391 ± 0.028 | 0.692 ± 0.017 | 0.533 ± 0.093 | 0.798 ± 0.022 |
| EffB3 16-d | 5 | 0.234 ± 0.030 | 0.298 ± 0.010 | 0.638 ± 0.008 | 0.356 ± 0.093 | 0.666 ± 0.065 |
| CSG λ=0 | 5 | 0.245 ± 0.033 | 0.291 ± 0.016 | 0.605 ± 0.013 | 0.489 ± 0.149 | 0.498 ± 0.026 |

The entangled baseline transfers better than the invariant model (0.391 vs 0.291). EffB3 matches CSG λ=0. Invariance training does not buy PAD diagnosis.

---

## What MASTER had, and what was missing

MASTER already carried the dial column (bal acc 0.291 → 0.291) and the sentence “flat or falls… pay a safety cost and get nothing.” It did not carry plain accuracy, per-class recall, the NV-collapse histogram, melanoma cells, macro AUC, or the controls. Those are the 60585 numbers that had not been reported.

Machine-readable: `phase6_xfer/diagnosis_aggregate.json`.
