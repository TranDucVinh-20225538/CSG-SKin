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
