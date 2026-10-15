# Paper B R5 / P13 — Fitzpatrick17k bootstrap over near-duplicate clusters

Committed before any P13 run (no image hashed, no P13 output at this commit). Fixed now; deviations appended.

## Question

Fitzpatrick17k CIs resample images (PREREGISTER caveat: anti-conservative). Several images can show the same case. With
clusters of near-duplicates as the resampling unit, how wide are the Fitzpatrick CIs and does rule A still hold?

## Data facts (from the metadata CSV, `results/paperB/phase11_third_domain/fitzpatrick17k.csv`)

All 3,887 OOD images come from atlasdermatologico.com.br (`url = .../img?imageId=N`); the CSV has md5hash, label, url,
Fitzpatrick type — no patient or case ID. imageId runs within a label give ≈ 222 clusters at gap ≤ 1, which are mostly
whole diagnosis blocks of the atlas, so they are too coarse to be a case identifier; they serve as a sensitivity bound.

## Cluster definitions (code: `scripts/r5_p13_fitz_clusters.py --hash`; CPU, reads the 3,887 JPEGs once)

- `cluster_phash` (primary): 64-bit DCT perceptual hash (32×32 grayscale, 8×8 low-frequency block, median threshold;
  implemented with PIL + scipy because `imagehash` is not installed), connected components at Hamming ≤ 8.
- `cluster_embed`: connected components at cosine ≥ 0.95 on the frozen ImageNet ResNet-50 Fitzpatrick features (R2 item 1
  cache; same sorted order — asserted).
- `cluster_union` (primary resampling unit): components of the union of both edge sets.
- `cluster_imageid_run` (sensitivity, coarse): same label and consecutive imageId.
- `cluster_diagnosis` (most conservative bound): the 90 diagnosis labels.
Output `results/paperB/r5/p13/fitz_clusters.csv` (md5hash, label, image_id, phash, all cluster columns) and
`cluster_stats.json` (cluster counts, singletons, largest, effective n = n² / Σ size²).

## Bootstrap (code: `--bootstrap`)

35 lesion-level runs, `z_lesion_norm`, Mahalanobis / kNN / cosine fit on ISIC train. PREREGISTER draws (2,000, rng
20261005, percentile 95%, shared across seeds), ID by lesion group; Fitzpatrick side resampled by each unit above.
Reported per λ: pooled mean and CI, CI width, k/n seeds with CI < 0.5, rule A; CI-width ratio vs image resampling.
The cluster CSV is also read by P3 / P12 (frozen monitor CIs).

## PREDICTED OUTCOME (written now)

- pHash ∪ embedding clusters: 3,500–3,850 clusters, ≥ 90 % singletons, largest ≤ 15; effective n ≥ 3,000.
- Mahalanobis at λ = 2: image CI ≈ [0.38, 0.42]; cluster-union CI ≤ 1.2× wider, still entirely below 0.5; rule A
  unchanged at λ ≥ 0.5. imageId-run clusters: CI 1.5–3× wider, upper end ≤ 0.47. Diagnosis clusters: CI 3–6× wider and may
  reach 0.5 (n = 90 units).
- λ = 0: all units keep the CI above 0.5 except possibly diagnosis clusters.

## Gates (flag, never drop)

If pHash finds a component with > 50 images, it is listed (likely a watermark / template artefact) and the
embedding-only definition is reported next to the union.

## Outputs

`results/paperB/r5/p13/`: `fitz_clusters.csv`, `cluster_stats.json`, `bootstrap.json`, `summary.csv`, `REPORT.md`.

## Cost

GPU 0. CPU: hashing ≈ 0.1 h on 4 CPUs; bootstrap ≈ 1 h on 8 CPUs (6 resampling units × 3 detectors × 7 λ × 2,000 draws).

## Stop rules

8 h cap. No re-thresholding after results are seen (Hamming 8 and cosine 0.95 fixed now).

## Deviations
