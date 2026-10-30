# Paper B R7 C3: Fitzpatrick17k coverage

Design: `decisions/paperB_r7_c3.md`. Commit `b0e93c1`.

CSV: `/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase11_third_domain/fitzpatrick17k.csv` (16577 rows). On-disk jpgs under FITZ_DIR: 3887. Matched kept rows: 3887.

## Missing-image reasons (HEAD sample)

| reason | count |
|---|---:|
| error:HTTP Error 406: Not Acceptable | 113 |
| ok | 37 |

## Class / Fitzpatrick mix: full vs kept (TV)

| variable | n_full | n_kept | TV |
|---|---:|---:|---:|
| (could not map kept rows to csv; see note.json / coverage.json) | 16577 | 3887 | — |
