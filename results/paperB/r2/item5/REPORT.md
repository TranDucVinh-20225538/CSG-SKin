# R2 Item 5 — Camelyon17 restricted to two hospitals

Commit: `ca19dd9`. Pre-commit: `results/paperB/r2/PRECOMMIT_CAMELYON2.json` (with amendments 1–3).

**Verdict:** Gate failed (Leakage moved) under the primary assignment -> negative reported; stop. No third design, no fourth dataset.

**λ-placement hypothesis:** NOT CONFIRMED (mechanistic). `tests/test_grl_placement.py`, one batch: the gradient entering the encoder is identical under both placements (cos = -1.000 to the unreversed gradient, magnitude exactly λα); the only difference is λ on the domain head's own gradient, and over 5 AdamW steps at the head lr that changes the head update norm by 1.5% (SGD: ×8.3). Placement cannot explain the Phase 12.1b adversary sitting at ln 3. The empirical diagnostic (job 65100) was cancelled before running: it could only re-observe the known 12.1b behaviour. The Camelyon17 and iWildCam limitations stay as written; the present runs use the 'grl' placement, so the adversary CE reported above is the direct evidence on this setup.

Assignment: primary (A = 3, B = 0, C = 2). Same-modality contrast: the pre-committed prediction is collapse yes, inversion no. This tests whether the collapse generalises outside dermatology; it does not test the inversion.

## Gate-by-gate (coarse scan, seed 42, assignment primary)

| Gate | Criterion | Value | Pass |
|---|---|---|---|
| Adversary learned | some λ>0 with min window CE < 0.60 | λ=0.1: 0.210; λ=0.3: 0.246; λ=1: 0.365; λ=3: 0.487; λ=10: 0.685 | yes |
| Leakage moved | site probe ≤ λ=0 − 0.10 where the adversary learned (λ=0: 0.952) | λ=0.1: 0.939; λ=0.3: 0.948; λ=1: 0.950; λ=3: 0.948; λ=10: 0.915 | no |
| ID competence | A test acc > 0.85 at λ=0 and λ* | λ=0: 0.995; λ=0.1: 0.995; λ=0.3: 0.994; λ=1: 0.995; λ=3: 0.993; λ=10: 0.989 (no λ*: evaluated at λ=0; λ* part not evaluable) | yes |
| Detector headroom | λ=0 Maha on B_heldout outside [0.45, 0.55] | 0.732 | yes |

## Coarse scan (seed 42) — ID reference = patch-level A test (primary)

| λ | n | Leakage (site probe bal acc) | ID acc | ID bal acc | ID ECE | OOD ECE (B_heldout) | Maha B_heldout | Cosine B_heldout | kNN B_heldout | MSP B_heldout | Energy B_heldout | Hospital C Maha |
|---:|---:|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 0.952 ± 0.000 | 0.995 ± 0.000 | 0.995 ± 0.000 | 0.003 ± 0.000 | 0.247 ± 0.000 | 0.732 ± 0.000 | 0.761 ± 0.000 | 0.724 ± 0.000 | 0.783 ± 0.000 | 0.757 ± 0.000 | 0.949 ± 0.000 |
| 0.1 | 1 | 0.939 ± 0.000 | 0.995 ± 0.000 | 0.995 ± 0.000 | 0.002 ± 0.000 | 0.036 ± 0.000 | 0.645 ± 0.000 | 0.600 ± 0.000 | 0.518 ± 0.000 | 0.532 ± 0.000 | 0.519 ± 0.000 | 0.952 ± 0.000 |
| 0.3 | 1 | 0.948 ± 0.000 | 0.994 ± 0.000 | 0.994 ± 0.000 | 0.002 ± 0.000 | 0.029 ± 0.000 | 0.574 ± 0.000 | 0.589 ± 0.000 | 0.580 ± 0.000 | 0.607 ± 0.000 | 0.585 ± 0.000 | 0.748 ± 0.000 |
| 1 | 1 | 0.950 ± 0.000 | 0.995 ± 0.000 | 0.995 ± 0.000 | 0.002 ± 0.000 | 0.017 ± 0.000 | 0.478 ± 0.000 | 0.555 ± 0.000 | 0.622 ± 0.000 | 0.611 ± 0.000 | 0.599 ± 0.000 | 0.596 ± 0.000 |
| 3 | 1 | 0.948 ± 0.000 | 0.993 ± 0.000 | 0.993 ± 0.000 | 0.001 ± 0.000 | 0.035 ± 0.000 | 0.639 ± 0.000 | 0.656 ± 0.000 | 0.647 ± 0.000 | 0.627 ± 0.000 | 0.622 ± 0.000 | 0.869 ± 0.000 |
| 10 | 1 | 0.915 ± 0.000 | 0.989 ± 0.000 | 0.989 ± 0.000 | 0.002 ± 0.000 | 0.020 ± 0.000 | 0.606 ± 0.000 | 0.647 ± 0.000 | 0.681 ± 0.000 | 0.611 ± 0.000 | 0.577 ± 0.000 | 0.832 ± 0.000 |

## Coarse scan (seed 42) — ID reference = slide-disjoint A slides 30, 32

OOD ECE is not an ID-side number and is repeated unchanged. The slide-disjoint set is 11.3% tumour; compare balanced accuracy.

| λ | n | Leakage (site probe bal acc) | ID acc | ID bal acc | ID ECE | OOD ECE (B_heldout) | Maha B_heldout | Cosine B_heldout | kNN B_heldout | MSP B_heldout | Energy B_heldout | Hospital C Maha |
|---:|---:|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 0.984 ± 0.000 | 0.985 ± 0.000 | 0.976 ± 0.000 | 0.008 ± 0.000 | 0.247 ± 0.000 | 0.730 ± 0.000 | 0.693 ± 0.000 | 0.651 ± 0.000 | 0.689 ± 0.000 | 0.684 ± 0.000 | 0.960 ± 0.000 |
| 0.1 | 1 | 0.976 ± 0.000 | 0.988 ± 0.000 | 0.980 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.000 | 0.468 ± 0.000 | 0.617 ± 0.000 | 0.637 ± 0.000 | 0.620 ± 0.000 | 0.639 ± 0.000 | 0.908 ± 0.000 |
| 0.3 | 1 | 0.980 ± 0.000 | 0.991 ± 0.000 | 0.979 ± 0.000 | 0.004 ± 0.000 | 0.029 ± 0.000 | 0.690 ± 0.000 | 0.678 ± 0.000 | 0.643 ± 0.000 | 0.549 ± 0.000 | 0.563 ± 0.000 | 0.846 ± 0.000 |
| 1 | 1 | 0.973 ± 0.000 | 0.988 ± 0.000 | 0.981 ± 0.000 | 0.003 ± 0.000 | 0.017 ± 0.000 | 0.697 ± 0.000 | 0.682 ± 0.000 | 0.691 ± 0.000 | 0.414 ± 0.000 | 0.453 ± 0.000 | 0.856 ± 0.000 |
| 3 | 1 | 0.984 ± 0.000 | 0.987 ± 0.000 | 0.977 ± 0.000 | 0.005 ± 0.000 | 0.035 ± 0.000 | 0.525 ± 0.000 | 0.600 ± 0.000 | 0.628 ± 0.000 | 0.573 ± 0.000 | 0.595 ± 0.000 | 0.841 ± 0.000 |
| 10 | 1 | 0.958 ± 0.000 | 0.976 ± 0.000 | 0.977 ± 0.000 | 0.004 ± 0.000 | 0.020 ± 0.000 | 0.421 ± 0.000 | 0.530 ± 0.000 | 0.661 ± 0.000 | 0.467 ± 0.000 | 0.490 ± 0.000 | 0.737 ± 0.000 |

## Per-epoch adversary logs

| λ | seed | epoch | adv CE | adv acc | ID select acc | GRL coef |
|---:|---:|---:|---|---|---|---|
| 0 | 42 | 1 | 0.1439 | 0.9390 | 0.9891 | 0.0000 |
| 0 | 42 | 2 | 0.1397 | 0.9406 | 0.9887 | 0.0000 |
| 0 | 42 | 3 | 0.1347 | 0.9433 | 0.9904 | 0.0000 |
| 0 | 42 | 4 | 0.1335 | 0.9438 | 0.9931 | 0.0000 |
| 0 | 42 | 5 | 0.1414 | 0.9406 | 0.9918 | 0.0000 |
| 0 | 42 | 6 | 0.1311 | 0.9457 | 0.9937 | 0.0000 |
| 0 | 42 | 7 | 0.1290 | 0.9466 | 0.9930 | 0.0000 |
| 0 | 42 | 8 | 0.1278 | 0.9461 | 0.9941 | 0.0000 |
| 0 | 42 | 9 | 0.1241 | 0.9485 | 0.9935 | 0.0000 |
| 0 | 42 | 10 | 0.1238 | 0.9484 | 0.9948 | 0.0000 |
| 0.1 | 42 | 1 | 0.5615 | 0.6902 | 0.9872 | 0.0996 |
| 0.1 | 42 | 2 | 0.6694 | 0.5902 | 0.9860 | 0.1000 |
| 0.1 | 42 | 3 | 0.6793 | 0.5682 | 0.9910 | 0.1000 |
| 0.1 | 42 | 4 | 0.6832 | 0.5565 | 0.9919 | 0.1000 |
| 0.1 | 42 | 5 | 0.6847 | 0.5515 | 0.9933 | 0.1000 |
| 0.1 | 42 | 6 | 0.6849 | 0.5488 | 0.9955 | 0.1000 |
| 0.1 | 42 | 7 | 0.6868 | 0.5443 | 0.9939 | 0.1000 |
| 0.1 | 42 | 8 | 0.6864 | 0.5430 | 0.9947 | 0.1000 |
| 0.1 | 42 | 9 | 0.6862 | 0.5459 | 0.9925 | 0.1000 |
| 0.1 | 42 | 10 | 0.6878 | 0.5406 | 0.9950 | 0.1000 |
| 0.3 | 42 | 1 | 0.6392 | 0.6145 | 0.9871 | 0.2989 |
| 0.3 | 42 | 2 | 0.6871 | 0.5512 | 0.9848 | 0.2999 |
| 0.3 | 42 | 3 | 0.6882 | 0.5431 | 0.9889 | 0.3000 |
| 0.3 | 42 | 4 | 0.6895 | 0.5366 | 0.9916 | 0.3000 |
| 0.3 | 42 | 5 | 0.6905 | 0.5340 | 0.9915 | 0.3000 |
| 0.3 | 42 | 6 | 0.6900 | 0.5337 | 0.9932 | 0.3000 |
| 0.3 | 42 | 7 | 0.6904 | 0.5341 | 0.9931 | 0.3000 |
| 0.3 | 42 | 8 | 0.6906 | 0.5325 | 0.9938 | 0.3000 |
| 0.3 | 42 | 9 | 0.6902 | 0.5346 | 0.9928 | 0.3000 |
| 0.3 | 42 | 10 | 0.6907 | 0.5307 | 0.9928 | 0.3000 |
| 1 | 42 | 1 | 0.6744 | 0.5774 | 0.9846 | 0.9964 |
| 1 | 42 | 2 | 0.6917 | 0.5328 | 0.9804 | 0.9997 |
| 1 | 42 | 3 | 0.6921 | 0.5306 | 0.9900 | 1.0000 |
| 1 | 42 | 4 | 0.6922 | 0.5271 | 0.9922 | 1.0000 |
| 1 | 42 | 5 | 0.6923 | 0.5250 | 0.9905 | 1.0000 |
| 1 | 42 | 6 | 0.6922 | 0.5209 | 0.9912 | 1.0000 |
| 1 | 42 | 7 | 0.6923 | 0.5240 | 0.9910 | 1.0000 |
| 1 | 42 | 8 | 0.6924 | 0.5208 | 0.9935 | 1.0000 |
| 1 | 42 | 9 | 0.6929 | 0.5200 | 0.9924 | 1.0000 |
| 1 | 42 | 10 | 0.6924 | 0.5208 | 0.9931 | 1.0000 |
| 3 | 42 | 1 | 0.6879 | 0.5594 | 0.9827 | 2.9893 |
| 3 | 42 | 2 | 0.6938 | 0.5289 | 0.9870 | 2.9992 |
| 3 | 42 | 3 | 0.6932 | 0.5229 | 0.9889 | 2.9999 |
| 3 | 42 | 4 | 0.6930 | 0.5198 | 0.9865 | 3.0000 |
| 3 | 42 | 5 | 0.6932 | 0.5138 | 0.9891 | 3.0000 |
| 3 | 42 | 6 | 0.6937 | 0.5111 | 0.9818 | 3.0000 |
| 3 | 42 | 7 | 0.6930 | 0.5113 | 0.9923 | 3.0000 |
| 3 | 42 | 8 | 0.6933 | 0.5114 | 0.9925 | 3.0000 |
| 3 | 42 | 9 | 0.6931 | 0.5110 | 0.9921 | 3.0000 |
| 3 | 42 | 10 | 0.6942 | 0.5081 | 0.9912 | 3.0000 |
| 10 | 42 | 1 | 0.7102 | 0.5367 | 0.9766 | 9.9642 |
| 10 | 42 | 2 | 0.6973 | 0.5144 | 0.9813 | 9.9974 |
| 10 | 42 | 3 | 0.6962 | 0.5162 | 0.9685 | 9.9997 |
| 10 | 42 | 4 | 0.6943 | 0.5105 | 0.9806 | 9.9999 |
| 10 | 42 | 5 | 0.6943 | 0.5099 | 0.9841 | 10.0000 |
| 10 | 42 | 6 | 0.6961 | 0.5169 | 0.9776 | 10.0000 |
| 10 | 42 | 7 | 0.6931 | 0.5122 | 0.9857 | 10.0000 |
| 10 | 42 | 8 | 0.6949 | 0.5087 | 0.9856 | 10.0000 |
| 10 | 42 | 9 | 0.6933 | 0.5099 | 0.9871 | 10.0000 |
| 10 | 42 | 10 | 0.6947 | 0.5065 | 0.9880 | 10.0000 |

200-step windows (full series in `runs/<tag>/adversary_windows.json`):

| λ | seed | first CE | first acc | first ‖g‖ | min CE | last CE | last ‖g‖ |
|---:|---:|---|---|---|---|---|---|
| 0 | 42 | 0.2144 | 0.9091 | 0 | 0.1063 | 0.1176 | 0 |
| 0.1 | 42 | 0.2104 | 0.9078 | 0.000249 | 0.2104 | 0.6885 | 9.29e-05 |
| 0.3 | 42 | 0.2458 | 0.8900 | 0.00104 | 0.2458 | 0.6896 | 0.000108 |
| 1 | 42 | 0.3652 | 0.8350 | 0.0042 | 0.3652 | 0.6922 | 0.000216 |
| 3 | 42 | 0.4872 | 0.7867 | 0.0152 | 0.4872 | 0.6920 | 0.000911 |
| 10 | 42 | 0.6850 | 0.7397 | 0.0498 | 0.6850 | 0.6935 | 0.0013 |

## λ-placement hypothesis (Phase 12.1b / iWildCam)

Phase 12.1b and iWildCam multiply λ into the domain head's loss with the GRL at coefficient α; this item puts λ on the reversed gradient only (GRL coefficient λα, head minimises unscaled CE).

**λ-placement verdict:** NOT CONFIRMED (mechanistic). `tests/test_grl_placement.py`, one batch: the gradient entering the encoder is identical under both placements (cos = -1.000 to the unreversed gradient, magnitude exactly λα); the only difference is λ on the domain head's own gradient, and over 5 AdamW steps at the head lr that changes the head update norm by 1.5% (SGD: ×8.3). Placement cannot explain the Phase 12.1b adversary sitting at ln 3. The empirical diagnostic (job 65100) was cancelled before running: it could only re-observe the known 12.1b behaviour. The Camelyon17 and iWildCam limitations stay as written; the present runs use the 'grl' placement, so the adversary CE reported above is the direct evidence on this setup.

