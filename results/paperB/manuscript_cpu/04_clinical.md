# (4) Clinical table

Source: `r5/posthoc/clinical/REPORT.md`.

## Label maps (dataset → ISIC)

### Fitzpatrick17k

| dataset `label` | ISIC |
|---|---|
| melanoma | MEL |
| basal cell carcinoma | BCC |
| squamous cell carcinoma | SCC |
| actinic keratosis | AK |
| dermatofibroma | DF |
| congenital nevus | NV |

Unmapped Fitzpatrick labels are excluded from per-class rates (counts under each set).

### SD-198

| ISIC | source folders (`source_label`) |
|---|---|
| NV | Blue_Nevus, Compound_Nevus, Congenital_Nevus, Dysplastic_Nevus, Halo_Nevus, Junction_Nevus, Nail_Nevus, Nevus_Incipiens, Nevus_Spilus |
| MEL | Malignant_Melanoma, Lentigo_Maligna_Melanoma |
| BCC | Basal_Cell_Carcinoma |
| AK | Actinic_solar_Damage(Actinic_Keratosis) |
| BKL | Seborrheic_Keratosis, Solar_Lentigo |
| DF | Dermatofibroma |
| VASC | Angioma, Pyogenic_Granuloma, Strawberry_Hemangioma |
| SCC | Bowen's_Disease, Keratoacanthoma |

Unmapped SD-198 folders are excluded from per-class rates.

### pad_heldout

Native PAD six-class labels. **n only** (no acceptance rates): MEL = 9 (full counts repeated per cell below).

## Summary table (n + cluster CI)

| cell | set | class | n | share accepted as ID | 95% CI | share pred NV & conf>0.9 | 95% CI |
|---|---|---|---:|---:|---|---:|---|
| R1 shift 0% λ=2 | fitzpatrick17k | MEL | 83 | 0.913 ± 0.040 | [0.880, 0.943] | 0.214 ± 0.067 | [0.146, 0.287] |
| R1 shift 0% λ=2 | fitzpatrick17k | BCC | 257 | 0.908 ± 0.044 | [0.887, 0.927] | 0.138 ± 0.044 | [0.106, 0.172] |
| R1 shift 0% λ=2 | fitzpatrick17k | SCC | 212 | 0.837 ± 0.060 | [0.811, 0.863] | 0.081 ± 0.038 | [0.054, 0.112] |
| R1 shift 0% λ=2 | fitzpatrick17k | AK | 24 | 0.896 ± 0.072 | [0.833, 0.948] | 0.115 ± 0.021 | [0.010, 0.240] |
| R1 shift 0% λ=2 | sd198 | MEL | 113 | 0.929 ± 0.040 | [0.896, 0.958] | 0.338 ± 0.086 | [0.268, 0.407] |
| R1 shift 0% λ=2 | sd198 | BCC | 60 | 0.892 ± 0.022 | [0.838, 0.942] | 0.175 ± 0.096 | [0.108, 0.246] |
| R1 shift 0% λ=2 | sd198 | SCC | 119 | 0.855 ± 0.053 | [0.821, 0.887] | 0.149 ± 0.058 | [0.107, 0.197] |
| R1 shift 0% λ=2 | sd198 | AK | 60 | 0.946 ± 0.032 | [0.912, 0.975] | 0.100 ± 0.059 | [0.054, 0.154] |
| R1 shift 100% λ=2 | fitzpatrick17k | MEL | 83 | 0.988 ± 0.012 | [0.970, 1.000] | 0.530 ± 0.095 | [0.439, 0.619] |
| R1 shift 100% λ=2 | fitzpatrick17k | BCC | 257 | 0.990 ± 0.009 | [0.984, 0.995] | 0.767 ± 0.080 | [0.729, 0.802] |
| R1 shift 100% λ=2 | fitzpatrick17k | SCC | 212 | 0.956 ± 0.015 | [0.937, 0.972] | 0.567 ± 0.094 | [0.511, 0.622] |
| R1 shift 100% λ=2 | fitzpatrick17k | AK | 24 | 1.000 ± 0.000 | [1.000, 1.000] | 0.775 ± 0.116 | [0.667, 0.883] |
| R1 shift 100% λ=2 | sd198 | MEL | 113 | 0.995 ± 0.005 | [0.988, 1.000] | 0.832 ± 0.077 | [0.782, 0.876] |
| R1 shift 100% λ=2 | sd198 | BCC | 60 | 0.990 ± 0.015 | [0.977, 1.000] | 0.893 ± 0.043 | [0.840, 0.940] |
| R1 shift 100% λ=2 | sd198 | SCC | 119 | 0.982 ± 0.009 | [0.963, 0.995] | 0.780 ± 0.073 | [0.721, 0.832] |
| R1 shift 100% λ=2 | sd198 | AK | 60 | 0.997 ± 0.007 | [0.990, 1.000] | 0.877 ± 0.048 | [0.823, 0.923] |
| a4 λ=0 | fitzpatrick17k | MEL | 83 | 0.851 ± 0.073 | [0.800, 0.899] | 0.077 ± 0.011 | [0.040, 0.119] |
| a4 λ=0 | fitzpatrick17k | BCC | 257 | 0.769 ± 0.092 | [0.739, 0.798] | 0.042 ± 0.021 | [0.028, 0.058] |
| a4 λ=0 | fitzpatrick17k | SCC | 212 | 0.730 ± 0.079 | [0.694, 0.767] | 0.028 ± 0.015 | [0.014, 0.044] |
| a4 λ=0 | fitzpatrick17k | AK | 24 | 0.825 ± 0.054 | [0.750, 0.892] | 0.058 ± 0.037 | [0.008, 0.125] |
| a4 λ=0 | sd198 | MEL | 113 | 0.919 ± 0.059 | [0.887, 0.945] | 0.103 ± 0.055 | [0.069, 0.138] |
| a4 λ=0 | sd198 | BCC | 60 | 0.760 ± 0.108 | [0.697, 0.813] | 0.040 ± 0.045 | [0.017, 0.067] |
| a4 λ=0 | sd198 | SCC | 119 | 0.773 ± 0.048 | [0.724, 0.820] | 0.049 ± 0.052 | [0.029, 0.072] |
| a4 λ=0 | sd198 | AK | 60 | 0.787 ± 0.099 | [0.733, 0.840] | 0.017 ± 0.020 | [0.000, 0.037] |
| a4 λ=2 | fitzpatrick17k | MEL | 83 | 0.954 ± 0.013 | [0.925, 0.978] | 0.398 ± 0.046 | [0.305, 0.486] |
| a4 λ=2 | fitzpatrick17k | BCC | 257 | 0.982 ± 0.011 | [0.974, 0.989] | 0.581 ± 0.124 | [0.537, 0.624] |
| a4 λ=2 | fitzpatrick17k | SCC | 212 | 0.937 ± 0.028 | [0.916, 0.955] | 0.448 ± 0.096 | [0.395, 0.501] |
| a4 λ=2 | fitzpatrick17k | AK | 24 | 0.992 ± 0.019 | [0.975, 1.000] | 0.550 ± 0.116 | [0.392, 0.700] |
| a4 λ=2 | sd198 | MEL | 113 | 0.979 ± 0.018 | [0.956, 0.995] | 0.637 ± 0.192 | [0.581, 0.696] |
| a4 λ=2 | sd198 | BCC | 60 | 1.000 ± 0.000 | [1.000, 1.000] | 0.720 ± 0.107 | [0.647, 0.787] |
| a4 λ=2 | sd198 | SCC | 119 | 0.976 ± 0.018 | [0.960, 0.990] | 0.582 ± 0.113 | [0.521, 0.645] |
| a4 λ=2 | sd198 | AK | 60 | 0.997 ± 0.007 | [0.990, 1.000] | 0.653 ± 0.135 | [0.573, 0.733] |
| old_recipe λ=0 | fitzpatrick17k | MEL | 83 | 0.995 ± 0.007 | [0.988, 1.000] | 0.075 ± 0.043 | [0.039, 0.116] |
| old_recipe λ=0 | fitzpatrick17k | BCC | 257 | 0.980 ± 0.010 | [0.971, 0.988] | 0.044 ± 0.026 | [0.029, 0.062] |
| old_recipe λ=0 | fitzpatrick17k | SCC | 212 | 0.944 ± 0.048 | [0.929, 0.959] | 0.047 ± 0.035 | [0.030, 0.066] |
| old_recipe λ=0 | fitzpatrick17k | AK | 24 | 0.975 ± 0.023 | [0.950, 1.000] | 0.083 ± 0.042 | [0.017, 0.167] |
| old_recipe λ=0 | sd198 | MEL | 113 | 0.977 ± 0.010 | [0.958, 0.991] | 0.081 ± 0.041 | [0.050, 0.117] |
| old_recipe λ=0 | sd198 | BCC | 60 | 0.983 ± 0.012 | [0.963, 0.997] | 0.037 ± 0.022 | [0.010, 0.070] |
| old_recipe λ=0 | sd198 | SCC | 119 | 0.953 ± 0.027 | [0.933, 0.971] | 0.049 ± 0.044 | [0.030, 0.071] |
| old_recipe λ=0 | sd198 | AK | 60 | 0.990 ± 0.015 | [0.977, 1.000] | 0.010 ± 0.009 | [0.000, 0.023] |
| old_recipe λ=2 | fitzpatrick17k | MEL | 83 | 0.983 ± 0.025 | [0.964, 0.998] | 0.487 ± 0.070 | [0.393, 0.580] |
| old_recipe λ=2 | fitzpatrick17k | BCC | 257 | 0.978 ± 0.016 | [0.967, 0.988] | 0.693 ± 0.067 | [0.649, 0.733] |
| old_recipe λ=2 | fitzpatrick17k | SCC | 212 | 0.942 ± 0.036 | [0.920, 0.960] | 0.464 ± 0.053 | [0.409, 0.520] |
| old_recipe λ=2 | fitzpatrick17k | AK | 24 | 0.975 ± 0.037 | [0.942, 1.000] | 0.700 ± 0.068 | [0.558, 0.833] |
| old_recipe λ=2 | sd198 | MEL | 113 | 0.986 ± 0.008 | [0.973, 0.995] | 0.812 ± 0.016 | [0.758, 0.865] |
| old_recipe λ=2 | sd198 | BCC | 60 | 0.990 ± 0.015 | [0.980, 1.000] | 0.820 ± 0.078 | [0.750, 0.883] |
| old_recipe λ=2 | sd198 | SCC | 119 | 0.966 ± 0.034 | [0.948, 0.983] | 0.716 ± 0.045 | [0.652, 0.778] |
| old_recipe λ=2 | sd198 | AK | 60 | 0.990 ± 0.015 | [0.973, 1.000] | 0.833 ± 0.075 | [0.770, 0.893] |

