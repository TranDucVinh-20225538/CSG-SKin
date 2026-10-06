# Leakage chance-floor audit

Does not change any conclusion. It is the class of error that makes a reviewer distrust every other number. Done before the abstract.

---

## Two quantities that were mixed

| Quantity | Metric | Chance floor | Where it lives |
|---|---|---|---|
| Phase 0 label-only probe; published Phase 1 “Domain probe acc”; `check_leakage.py` | **plain accuracy** | **0.688** = 5067/7365 (always predict ISIC) | `PHASE0_CHECKS.md`, `phase0_label_domain_probe.json` (`majority_baseline` = 0.6879837067209776), `PHASE1_REPORT.md` |
| Phase 3 / MASTER dial “leakage bal acc”; Phase 13; `leakage_probe()` | **2-class balanced accuracy** | **0.5** | `train_phase3_sweep.py` `balanced_accuracy_score`; tables use `bal_acc_mean` |
| Camelyon17 3-class leakage | **3-class balanced accuracy** | **1/3 ≈ 0.333** | Phase 12. The figure 0.437 = 132k/302k is the **plain-accuracy** majority (always hospital 4). |

A constant “always ISIC” predictor scores plain acc 0.688 and balanced acc 0.5 (recall_ISIC = 1, recall_PAD = 0). Using 0.688 as the floor of a balanced-accuracy column is mixed-scale.

---

## Code (verified)

`scripts/check_leakage.py` `linear_probe_accuracy` returns `accuracy_score` only. Published runB z_lesion **0.724** and ResNet-50 **0.979** are plain accuracy. Comparing them to majority 0.688, or to the label-only ceiling 0.7968, is on-scale.

`scripts/train_phase3_sweep.py` `leakage_probe` records both:

```
acc_mean          <- plain accuracy
bal_acc_mean      <- 2-class balanced accuracy   ← this is the dial column
auroc_mean
```

`aggregate_final_package.py` reads `bal_acc_mean`. The four-column dial is balanced accuracy. Its floor is 0.5, not 0.688.

Phase 12 `leakage_3class` records both `acc` and `balanced_acc`. Reports already use 1/3, not 0.437.

---

## Phase 3 leakage on both scales (same probe, same splits)

| λ_adv | n | plain acc | vs majority 0.688 | bal acc (dial) | vs floor 0.5 |
|---:|---:|---|---|---|---|
| 0 | 5 | 0.930 ± 0.016 | +0.242 | 0.915 ± 0.021 | +0.415 |
| 0.25 | 3 | 0.702 ± 0.004 | +0.014 | 0.553 ± 0.008 | +0.053 |
| 0.5 | 3 | 0.699 ± 0.011 | +0.011 | 0.555 ± 0.029 | +0.055 |
| 1 | 3 | 0.714 ± 0.004 | +0.026 | 0.569 ± 0.010 | +0.069 |
| 2 | 5 | 0.717 ± 0.014 | +0.029 | 0.584 ± 0.027 | +0.084 |
| 4 | 3 | 0.699 ± 0.017 | +0.011 | 0.548 ± 0.035 | +0.048 |
| 8 | 5 | 0.739 ± 0.022 | +0.051 | 0.625 ± 0.045 | +0.125 |

At the cliff, plain acc 0.702 is 1.4 pp above majority; bal acc 0.553 is 5.3 pp above 0.5. Both say “near chance.” The cliff (0.915 → 0.553 on bal; 0.930 → 0.702 on plain) is present on both scales. **No conclusion moves.**

Do not write “0.553 vs majority 0.688.” That subtracts a plain-accuracy floor from a balanced-accuracy number and makes λ=0.25 look 13.5 pp *below* chance. It is 5.3 pp above chance.

---

## File-by-file

| File | Status |
|---|---|
| `MASTER_REPORT.md` L21, L95, L151, L167, L211–212; `phase10/table1.md` | Already correct: dial floor 0.5; 0.688 named as Phase 0 plain majority. |
| `PHASE0_CHECKS.md` | Correct. 0.7968 vs 0.688 is plain accuracy. Also reports label-only **balanced** acc 0.7884 (floor for that cell would be 0.5; not used as a claim). |
| `PHASE1_REPORT.md` “Domain probe acc” 0.979 / 0.802 / 0.724 | Correct metric (plain acc from `check_leakage.py`). Footnote added that this is not the Phase 3 bal-acc column. `id_bal=0.6887` on one EffB3 seed is ID balanced accuracy, coincidence with 0.688. |
| `PHASE1_5_REPORT.md` “logistic 0.81–0.88 vs majority 0.688” | **Was mixed-scale.** 0.81–0.88 is logistic **AUROC**; 0.688 is plain-acc majority. Actual plain acc of those logistics is ~0.81; bal acc is 0.69–0.77. Sentence corrected in both copies. Conclusion (features carry domain, do not reproduce Maha) unchanged. |
| Phase 12 reports | Already correct: 3-class floor 1/3, not 0.437. |
| `PHASE13_REPORT.md` | Already: “chance floor 0.5.” |
| MASTER claims table “λ=0: 0.915; published runB ~0.72; λ=0.25: 0.553” | **Was mixed-scale.** 0.915 / 0.553 are Phase 3 bal acc; ~0.72 is published plain acc on a different checkpoint. Sentence corrected. |

---

## Writing rule (lock)

- Dial / Phase 3 / Phase 13 leakage column: **balanced accuracy, floor 0.5.**
- Phase 0 label-only and published Phase 1 probe: **plain accuracy, floor 0.688.**
- Never put 0.724 and 0.915 in one “leakage” clause without naming the metric.
- Camelyon: **balanced accuracy, floor 1/3**, not 0.437.
- pad_heldout transfer (Item 1): **6-class balanced accuracy, floor 1/6 ≈ 0.167.** Majority plain acc is 0.385 (always BCC). See `phase6_xfer/PHASE6_XFER_DIAGNOSIS.md`.

Conclusions that survive the correction: λ=0.25 leakage is near chance; the 0→0.25 drop is a cliff; leakage is not a safe operating-point knob; Camelyon never left chance under the weak adversary.
