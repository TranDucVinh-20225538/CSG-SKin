# Paper B R7 consolidated report

Generated at commit `79c7002`.

## Item hashes and status

| item | summaries | report |
|---|---:|---|
| t1 | 15/15 | yes |
| t1b | 5/5 | yes |
| t2 | 9/9 | yes |
| c1 | — | yes |
| c2 | — | yes |
| c3 | — | yes |

GPU hours (sum of available run summaries): 25.34.


---

## C1

# Paper B R7 C1: FPR@95TPR (Mahalanobis, Table 1 last.ckpt)

Design: `decisions/paperB_r7_c1.md`. Commit `b0e93c1`.

Score: class-conditional Mahalanobis on `z_lesion_norm`, fit on ISIC train. Higher = more OOD. τ = 95th percentile of ID-test scores. FPR@95TPR = fraction of OOD with score ≥ τ. OOD acceptance at 95% ID coverage = fraction of OOD with score ≤ τ.

| λ | OOD set | FPR@95 mean±sd | pooled CI | OOD accept@95ID mean±sd | n |
|---:|---|---|---|---|---:|
| 0 | fitzpatrick17k | 0.048 ± 0.020 | [0.040, 0.059] | 0.952 ± 0.020 | 5 |
| 0 | mednode | 0.011 ± 0.011 | [0.002, 0.016] | 0.989 ± 0.011 | 5 |
| 0 | pad_heldout | 0.089 ± 0.060 | [0.060, 0.118] | 0.911 ± 0.060 | 5 |
| 0 | scin | 0.051 ± 0.009 | [0.041, 0.062] | 0.949 ± 0.009 | 5 |
| 0 | sd198 | 0.040 ± 0.021 | [0.033, 0.049] | 0.960 ± 0.021 | 5 |
| 0.25 | fitzpatrick17k | 0.024 ± 0.007 | [0.020, 0.028] | 0.976 ± 0.007 | 5 |
| 0.25 | mednode | 0.005 ± 0.005 | [0.000, 0.011] | 0.995 ± 0.005 | 5 |
| 0.25 | pad_heldout | 0.041 ± 0.008 | [0.030, 0.050] | 0.959 ± 0.008 | 5 |
| 0.25 | scin | 0.028 ± 0.004 | [0.023, 0.032] | 0.972 ± 0.004 | 5 |
| 0.25 | sd198 | 0.020 ± 0.007 | [0.017, 0.023] | 0.980 ± 0.007 | 5 |
| 0.5 | fitzpatrick17k | 0.022 ± 0.008 | [0.019, 0.026] | 0.978 ± 0.008 | 5 |
| 0.5 | mednode | 0.001 ± 0.003 | [0.000, 0.004] | 0.999 ± 0.003 | 5 |
| 0.5 | pad_heldout | 0.039 ± 0.009 | [0.030, 0.048] | 0.961 ± 0.009 | 5 |
| 0.5 | scin | 0.030 ± 0.012 | [0.026, 0.035] | 0.970 ± 0.012 | 5 |
| 0.5 | sd198 | 0.017 ± 0.008 | [0.015, 0.020] | 0.983 ± 0.008 | 5 |
| 1 | fitzpatrick17k | 0.024 ± 0.011 | [0.020, 0.027] | 0.976 ± 0.011 | 5 |
| 1 | mednode | 0.005 ± 0.005 | [0.001, 0.011] | 0.995 ± 0.005 | 5 |
| 1 | pad_heldout | 0.034 ± 0.007 | [0.026, 0.044] | 0.966 ± 0.007 | 5 |
| 1 | scin | 0.028 ± 0.005 | [0.023, 0.033] | 0.972 ± 0.005 | 5 |
| 1 | sd198 | 0.018 ± 0.008 | [0.016, 0.021] | 0.982 ± 0.008 | 5 |
| 2 | fitzpatrick17k | 0.022 ± 0.005 | [0.018, 0.025] | 0.978 ± 0.005 | 5 |
| 2 | mednode | 0.001 ± 0.003 | [0.000, 0.005] | 0.999 ± 0.003 | 5 |
| 2 | pad_heldout | 0.039 ± 0.007 | [0.030, 0.049] | 0.961 ± 0.007 | 5 |
| 2 | scin | 0.035 ± 0.015 | [0.031, 0.040] | 0.965 ± 0.015 | 5 |
| 2 | sd198 | 0.017 ± 0.005 | [0.014, 0.019] | 0.983 ± 0.005 | 5 |
| 4 | fitzpatrick17k | 0.021 ± 0.009 | [0.018, 0.025] | 0.979 ± 0.009 | 5 |
| 4 | mednode | 0.002 ± 0.003 | [0.000, 0.006] | 0.998 ± 0.003 | 5 |
| 4 | pad_heldout | 0.048 ± 0.026 | [0.038, 0.058] | 0.952 ± 0.026 | 5 |
| 4 | scin | 0.028 ± 0.008 | [0.024, 0.033] | 0.972 ± 0.008 | 5 |
| 4 | sd198 | 0.016 ± 0.008 | [0.014, 0.018] | 0.984 ± 0.008 | 5 |
| 8 | fitzpatrick17k | 0.021 ± 0.009 | [0.018, 0.024] | 0.979 ± 0.009 | 5 |
| 8 | mednode | 0.002 ± 0.003 | [0.000, 0.006] | 0.998 ± 0.003 | 5 |
| 8 | pad_heldout | 0.053 ± 0.023 | [0.043, 0.064] | 0.947 ± 0.023 | 5 |
| 8 | scin | 0.041 ± 0.017 | [0.036, 0.046] | 0.959 ± 0.017 | 5 |
| 8 | sd198 | 0.014 ± 0.005 | [0.012, 0.016] | 0.986 ± 0.005 | 5 |

## pad_heldout acceptance at λ = 0 and 2

- λ = 0: OOD accept 0.911 ± 0.060; FPR@95 0.089 ± 0.060 CI [0.060, 0.118].
- λ = 2: OOD accept 0.961 ± 0.007; FPR@95 0.039 ± 0.007 CI [0.030, 0.049].

---

## C2

# Paper B R7 C2: figure and clinical run-set audit

Design: `decisions/paperB_r7_c2.md`. Commit `b0e93c1`.

`FIGURES_REPORT.md` is stale. Audit is against `scripts/plot_manuscript_figures.py`.

| Panel | Script | Run set | Checkpoint | Note |
|---|---|---|---|---|
| Fig 1a–d | `fig1_intervention` | Table 1 last.ckpt (r5/t1last/summary.csv) | last.ckpt | load_t1last_by_lambda; OOD ECE from UNSEEN_ECE.json |
| Fig 2 | `fig2_confidence` | phase13/per_run (STALE vs Table 1) | phase13 (best-era) | still load_phase13_runs; regenerate from t1last below |
| Fig 3 | `fig3_detectors` | phase3_sweep summaries | phase3 | load_phase3_summaries |
| Fig 4–6 / S* | `(various)` | see plot_manuscript_figures.py | mixed / legacy | FIGURES_REPORT.md header is stale; trust the script |
| §4.4 / App clinical % | `clinical_t1last` | Table 1 last.ckpt (clinical_t1last) | last.ckpt | /data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/r5/posthoc/clinical_t1last/results.json |

## Fig 2: phase13 (old) vs t1last last.ckpt (new)

Fig 2 still calls `load_phase13_runs()`. Regenerated MSP confidence from `results/paperB/r5/t1last/features_last` (Table 1 last.ckpt).

| λ | set | old mean | new mean | Δ |
|---:|---|---:|---:|---:|
| 0.0 | isic_test | 0.915 | 0.892 | -0.022 |
| 0.0 | pad_heldout | 0.489 | 0.444 | -0.046 |
| 0.0 | fitzpatrick17k | 0.601 | 0.665 | +0.065 |
| 0.25 | isic_test | 0.910 | 0.892 | -0.018 |
| 0.25 | pad_heldout | 0.898 | 0.909 | +0.011 |
| 0.25 | fitzpatrick17k | 0.909 | 0.921 | +0.012 |
| 0.5 | isic_test | 0.908 | 0.892 | -0.016 |
| 0.5 | pad_heldout | 0.911 | 0.924 | +0.013 |
| 0.5 | fitzpatrick17k | 0.914 | 0.935 | +0.022 |
| 1.0 | isic_test | 0.909 | 0.894 | -0.015 |
| 1.0 | pad_heldout | 0.923 | 0.936 | +0.013 |
| 1.0 | fitzpatrick17k | 0.925 | 0.923 | -0.002 |
| 2.0 | isic_test | 0.910 | 0.893 | -0.017 |
| 2.0 | pad_heldout | 0.930 | 0.936 | +0.006 |
| 2.0 | fitzpatrick17k | 0.935 | 0.927 | -0.009 |
| 4.0 | isic_test | 0.908 | 0.887 | -0.021 |
| 4.0 | pad_heldout | 0.916 | 0.927 | +0.011 |
| 4.0 | fitzpatrick17k | 0.924 | 0.937 | +0.013 |
| 8.0 | isic_test | 0.900 | 0.884 | -0.016 |
| 8.0 | pad_heldout | 0.906 | 0.918 | +0.013 |
| 8.0 | fitzpatrick17k | 0.901 | 0.923 | +0.022 |

Clinical percentages in §4.4 / App. H: `clinical_t1last` (Table 1 last.ckpt), already aligned.
No `.tex` edits in this item.


---

## C3

# Paper B R7 C3: Fitzpatrick17k coverage

Design: `decisions/paperB_r7_c3.md`. Commit `91d0da9`.

CSV: `/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase11_third_domain/fitzpatrick17k.csv` (16577 rows). On-disk jpgs under FITZ_DIR: 3887. Matched kept rows: 3887.

## Missing-image reasons (HEAD sample)

| reason | count |
|---|---:|
| http_406 | 150 |

## Class / Fitzpatrick mix: full vs kept (TV)

| variable | n_full | n_kept | TV |
|---|---:|---:|---:|
| raw_label (label) | 16577 | 3887 | 0.3294 |
| isic_mapped_label | 16577 | 3887 | 0.0630 |
| fitzpatrick (fitzpatrick_scale) | 16577 | 3887 | 0.2571 |

---

## T1

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


CASE.json: `{
  "case": "C",
  "reason": "gap=0.095 (near=False), pad Rule A=True, labelled-unseen Rule A count=3, leakage \u03bb=2=0.576",
  "gap": 0.09525779830839765,
  "id_dual_l0": 0.4699365895265982,
  "id_erm": 0.5651943878349959,
  "leakage_l2": 0.5759051614543607
}`

---

## T1B

# Paper B R7 T1b: single-encoder DANN DINOv2-S/14

Design: `decisions/paperB_r7_t1b.md`. Commit `79c7002`.

## Reading-rule outcome (computed first)

invariance not reached (V1=0.884>0.60): test did not take place; W1 stays open

Held-out leakage V1 mean: 0.884. Invariance (≤0.60): False. pad_heldout Rule A: above (k_below=0/5).

| λ | n | Leakage | ID bal acc | ID ECE | OOD ECE (pad_heldout) | OOD AUROC (Maha pad_heldout) | Fitz AUROC (Maha) | X-dom bal acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5 | 0.884 ± 0.019 | 0.555 ± 0.023 | 0.191 ± 0.003 | 0.672 ± 0.034 | 0.546 ± 0.033 | 0.476 ± 0.069 | 0.335 ± 0.025 |

GPU hours: 3.62.


OUTCOME.json: `{
  "outcome": "invariance not reached (V1=0.884>0.60): test did not take place; W1 stays open",
  "leakage": 0.8837843249427918,
  "invariance": false,
  "inverts": false,
  "pad_rule": {
    "mean": 0.5460497490057806,
    "sd": 0.032506069572473754,
    "pooled_ci": [
      0.5187817037111866,
      0.5724106696364775
    ],
    "k_below": 0,
    "n_seeds": 5,
    "rule": "above",
    "per_seed": {
      "42": {
        "point": 0.5739655653351239,
        "ci_lo": 0.5450596104361857,
        "ci_hi": 0.601611756572126,
        "ci_entirely_below_0.5": false
      },
      "52": {
        "point": 0.5521726738795119,
        "ci_lo": 0.52294864952335,
        "ci_hi": 0.5804934513464088,
        "ci_entirely_below_0.5": false
      },
      "62": {
        "point": 0.5752004149976903,
        "ci_lo": 0.5459040534681437,
        "ci_hi": 0.6013519533233326,
        "ci_entirely_below_0.5": false
      },
      "72": {
        "point": 0.49766839435364985,
        "ci_lo": 0.4679849917327486,
        "ci_hi": 0.5271947531350282,
        "ci_entirely_below_0.5": false
      },
      "82": {
        "point": 0.5312416964629275,
        "ci_lo": 0.5024056408179436,
        "ci_hi": 0.5599947015639621,
        "ci_entirely_below_0.5": false
      }
    }
  }
}`

---

## T2

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

