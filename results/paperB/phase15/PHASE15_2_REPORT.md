# Phase 15.2 — non-adversarial objectives

Tier: **main text** if 15.1 port is sound. Plot against **achieved leakage**, not nominal weight.
Leakage floor **0.5**. CORAL/MMD/ERM never use PAD labels. IRM/GroupDRO use pad_adv labels; pad_heldout unseen.

n_summaries = 39 / 39.

| objective | weight | n | leak bal | Maha | ID bal | xfer bal | pad labels in L_cls |
|---|---:|---:|---|---|---|---|---|
| coral | 1 | 3 | 0.979 ± 0.001 | 0.915 ± 0.018 | 0.768 ± 0.008 | 0.397 ± 0.029 | False |
| coral | 10 | 3 | 0.997 ± 0.001 | 0.923 ± 0.043 | 0.775 ± 0.016 | 0.391 ± 0.020 | False |
| coral | 100 | 3 | 0.998 ± 0.000 | 0.999 ± 0.000 | 0.777 ± 0.007 | 0.376 ± 0.012 | False |
| erm | 0 | 3 | 0.973 ± 0.005 | 0.838 ± 0.024 | 0.761 ± 0.008 | 0.399 ± 0.021 | False |
| groupdro | 0.1 | 3 | 0.977 ± 0.003 | 0.869 ± 0.020 | 0.786 ± 0.006 | 0.609 ± 0.025 | True |
| groupdro | 1 | 3 | 0.976 ± 0.002 | 0.873 ± 0.010 | 0.780 ± 0.008 | 0.598 ± 0.018 | True |
| groupdro | 10 | 3 | 0.976 ± 0.002 | 0.868 ± 0.008 | 0.783 ± 0.012 | 0.615 ± 0.021 | True |
| irm | 1 | 3 | 0.983 ± 0.002 | 0.936 ± 0.006 | 0.775 ± 0.013 | 0.618 ± 0.011 | True |
| irm | 10 | 3 | 0.987 ± 0.002 | 0.994 ± 0.001 | 0.692 ± 0.014 | 0.641 ± 0.038 | True |
| irm | 100 | 3 | 0.950 ± 0.016 | 0.729 ± 0.150 | 0.167 ± 0.045 | 0.242 ± 0.071 | True |
| mmd | 1 | 3 | 0.967 ± 0.003 | 0.813 ± 0.028 | 0.779 ± 0.006 | 0.399 ± 0.036 | False |
| mmd | 10 | 3 | 0.947 ± 0.001 | 0.765 ± 0.007 | 0.777 ± 0.010 | 0.399 ± 0.039 | False |
| mmd | 100 | 3 | 0.900 ± 0.017 | 0.716 ± 0.059 | 0.730 ± 0.028 | 0.401 ± 0.023 | False |

If all objectives collapse together vs leakage, the mechanism is invariance. If GRL (15.1) separates, it is adversarial dynamics.

