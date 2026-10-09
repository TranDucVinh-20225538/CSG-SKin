# R2 Item 4 — lesion-level ISIC splits

Commit: `79381fa`. Pre-commit: `results/paperB/r2/PRECOMMIT_ITEM4.json`.

**Verdict:** Pattern changes -> a finding about the main table; takes priority over everything else in R2. Class-mix control (lesion-level test reweighted to the image-level class distribution): pattern changes.

Seeds {42, 52, 62} at λ ∈ {0, 2} for both splits (matched; not the five-seed mean). Everything except the ISIC split is fixed. Detectors on z_lesion^norm, fit on ISIC train.

Null `lesion_id` policy: each such image is its own group (kept). Affected: 2084 of 25331 ISIC images (8.23%); train 8.20%, val 7.77%, test 8.66%. Lesion overlap train/test, val/test, train/val: 0 / 0 / 0.

**Image-level split (the paper's main table): 3041 of 5067 ISIC test images (60.0%) share a `lesion_id` with ISIC train or val; the lesion-disjoint test set is 2026 images (436 of them with null `lesion_id`, each its own group).** This belongs in the manuscript's Methods, not only in Limitations.

## Table 1 columns, mean ± s.d. over 3 seeds

| Split | λ | Leakage bal acc | ID bal acc | ID ECE | OOD ECE (pad_heldout, 6-cls) | Maha pad_heldout | Cosine pad_heldout | kNN pad_heldout | MSP pad_heldout | Energy pad_heldout | Fitz AUROC (Maha) |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| image | 0 | 0.916 ± 0.027 | 0.693 ± 0.020 | 0.106 ± 0.007 | 0.217 ± 0.015 | 0.858 ± 0.014 | 0.884 ± 0.007 | 0.923 ± 0.009 | 0.954 ± 0.008 | 0.972 ± 0.013 | 0.791 ± 0.009 |
| image | 2 | 0.574 ± 0.033 | 0.703 ± 0.007 | 0.100 ± 0.003 | 0.744 ± 0.002 | 0.432 ± 0.024 | 0.457 ± 0.023 | 0.445 ± 0.013 | 0.461 ± 0.027 | 0.457 ± 0.014 | 0.427 ± 0.008 |
| lesion | 0 | 0.925 ± 0.023 | 0.504 ± 0.015 | 0.182 ± 0.021 | 0.198 ± 0.045 | 0.815 ± 0.029 | 0.869 ± 0.025 | 0.915 ± 0.023 | 0.947 ± 0.011 | 0.966 ± 0.009 | 0.727 ± 0.043 |
| lesion | 2 | 0.565 ± 0.056 | 0.464 ± 0.028 | 0.175 ± 0.032 | 0.714 ± 0.027 | 0.400 ± 0.033 | 0.416 ± 0.029 | 0.420 ± 0.028 | 0.450 ± 0.024 | 0.455 ± 0.011 | 0.407 ± 0.016 |

## Class-mix control

Without the reweighted column a difference is ambiguous between "leakage removed" and "class mix changed" (see `results/paperB/r2/disjoint_reweight/REPORT.md`). Reweighting: each ISIC test image of class c weighted p_image-level(c) / p_lesion-level(c); OOD weight 1. Detector scores recomputed per image from cached features (z_lesion^norm, logits), fit on the run's own ISIC train.

| Class | Image-level test n | % | Lesion-level test n | % |
|---|---:|---:|---:|---:|
| MEL | 904 | 17.8 | 904 | 17.8 |
| NV | 2575 | 50.8 | 2575 | 50.8 |
| BCC | 665 | 13.1 | 664 | 13.1 |
| AK | 173 | 3.4 | 174 | 3.4 |
| BKL | 525 | 10.4 | 525 | 10.4 |
| DF | 48 | 0.9 | 48 | 0.9 |
| VASC | 51 | 1.0 | 51 | 1.0 |
| SCC | 126 | 2.5 | 126 | 2.5 |
| total | 5067 | 100 | 5067 | 100 |

| Detector | λ | image-level | lesion-level, unrestricted | lesion-level, reweighted to image-level mix |
|---|---:|---|---|---|
| Maha pad_heldout | 0 | 0.858 ± 0.014 | 0.815 ± 0.029 | 0.815 ± 0.029 |
| Maha pad_heldout | 2 | 0.432 ± 0.023 | 0.400 ± 0.033 | 0.400 ± 0.033 |
| Cosine pad_heldout | 0 | 0.884 ± 0.007 | 0.869 ± 0.025 | 0.869 ± 0.025 |
| Cosine pad_heldout | 2 | 0.457 ± 0.023 | 0.416 ± 0.029 | 0.416 ± 0.029 |
| kNN pad_heldout | 0 | 0.923 ± 0.009 | 0.915 ± 0.023 | 0.915 ± 0.023 |
| kNN pad_heldout | 2 | 0.445 ± 0.013 | 0.420 ± 0.028 | 0.420 ± 0.028 |
| MSP pad_heldout | 0 | 0.954 ± 0.008 | 0.947 ± 0.011 | 0.947 ± 0.011 |
| MSP pad_heldout | 2 | 0.461 ± 0.026 | 0.450 ± 0.024 | 0.450 ± 0.024 |
| Energy pad_heldout | 0 | 0.972 ± 0.013 | 0.966 ± 0.009 | 0.966 ± 0.009 |
| Energy pad_heldout | 2 | 0.457 ± 0.014 | 0.455 ± 0.011 | 0.455 ± 0.011 |
| Fitz AUROC (Maha) | 0 | 0.791 ± 0.009 | 0.727 ± 0.043 | 0.727 ± 0.043 |
| Fitz AUROC (Maha) | 2 | 0.427 ± 0.008 | 0.407 ± 0.016 | 0.407 ± 0.016 |

Recomputed image-level values should match the Table 1 columns above (same features, same definitions).

## Per seed

| Split | λ | seed | Leakage bal acc | ID bal acc | ID ECE | OOD ECE (pad_heldout, 6-cls) | Maha pad_heldout | Cosine pad_heldout | kNN pad_heldout | MSP pad_heldout | Energy pad_heldout | Fitz AUROC (Maha) |
|---|---:|---:|---|---|---|---|---|---|---|---|---|---|
| image | 0 | 42 | 0.917 | 0.707 | 0.114 | 0.224 | 0.852 | 0.880 | 0.921 | 0.955 | 0.980 | 0.787 |
| image | 0 | 52 | 0.942 | 0.670 | 0.100 | 0.200 | 0.874 | 0.892 | 0.933 | 0.963 | 0.980 | 0.801 |
| image | 0 | 62 | 0.888 | 0.703 | 0.105 | 0.228 | 0.848 | 0.879 | 0.915 | 0.946 | 0.957 | 0.784 |
| image | 2 | 42 | 0.593 | 0.695 | 0.103 | 0.746 | 0.431 | 0.454 | 0.439 | 0.446 | 0.459 | 0.435 |
| image | 2 | 52 | 0.536 | 0.706 | 0.099 | 0.744 | 0.409 | 0.436 | 0.436 | 0.444 | 0.442 | 0.419 |
| image | 2 | 62 | 0.592 | 0.709 | 0.097 | 0.743 | 0.456 | 0.482 | 0.461 | 0.491 | 0.470 | 0.427 |
| lesion | 0 | 42 | 0.913 | 0.495 | 0.202 | 0.250 | 0.795 | 0.842 | 0.889 | 0.935 | 0.968 | 0.685 |
| lesion | 0 | 52 | 0.952 | 0.521 | 0.184 | 0.181 | 0.849 | 0.891 | 0.935 | 0.955 | 0.973 | 0.772 |
| lesion | 0 | 62 | 0.910 | 0.497 | 0.161 | 0.164 | 0.802 | 0.874 | 0.921 | 0.952 | 0.956 | 0.724 |
| lesion | 2 | 42 | 0.515 | 0.436 | 0.212 | 0.744 | 0.371 | 0.392 | 0.393 | 0.422 | 0.447 | 0.388 |
| lesion | 2 | 52 | 0.554 | 0.465 | 0.157 | 0.705 | 0.391 | 0.409 | 0.418 | 0.465 | 0.468 | 0.417 |
| lesion | 2 | 62 | 0.625 | 0.491 | 0.157 | 0.693 | 0.436 | 0.448 | 0.449 | 0.462 | 0.451 | 0.416 |

## Pre-committed pattern checks

| Check | image-level | lesion-level | lesion-level, reweighted |
|---|---|---|---|
| leakage_falls | True | True | True |
| id_bal_not_lower_by_0.02 | True | False | False |
| id_ece_within_0.03 | True | True | True |
| ood_ece_rises | True | True | True |
| maha_above_0.5_at_0_below_at_2 | True | True | True |
| all_five_detectors_fall | True | True | True |
| fitz_above_0.5_at_0_below_at_2 | True | True | True |

## Caveats

- Image-level λ=2 seeds reuse the Phase 2 checkpoints (same protocol); lesion-level λ=2 runs train fresh.
- The lesion-level ISIC test set is a different image set from the image-level one; ID columns are not paired by image.
