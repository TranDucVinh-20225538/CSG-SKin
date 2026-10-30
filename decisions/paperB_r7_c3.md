# Paper B R7 / C3 — Fitzpatrick17k coverage (CPU, headers only)

Committed before the C3 job. Deviations appended at the end.

## Design

- From the download log, or by re-requesting headers only (no image download), count why images are missing: HTTP status / host unreachable / removed, per source atlas.
- Compare the class mix (mapped to the ISIC classes) and the Fitzpatrick skin-type mix of the 3,887 images with the full 16,577 in `fitzpatrick17k.csv`.
- One table, total variation distance per variable.
- No image download.

## Outputs

`results/paperB/r7/c3/{coverage.json,REPORT.md}`.

## Deviations
