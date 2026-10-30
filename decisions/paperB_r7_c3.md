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

- 2026-10-30: atlas host www.dermaamin.com returns HTTP 406 for HEAD and Range-GET probes; missing-image reason table is therefore host-refusal rather than 404/unreachable. Kept=3887 matched by md5hash.jpg; TV reported for raw label, ISIC-mapped label, and fitzpatrick_scale.
