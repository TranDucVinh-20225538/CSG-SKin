# Paper B R5 / Q5 — frozen-encoder monitor on every Q1–Q3 run

Committed before any Q1–Q3 run exists and before any Q5 score. Fixed now; deviations appended.

## Question

Does the separate frozen-encoder monitor keep its OOD AUROC where the DANN-feature monitor inverts, beyond the main
task — on the Q1 second task (CXR: NIH CXR14 + Kermany pediatric as training domains, Shenzhen OOD), Q2 alignment
methods (CDAN; CORAL folded into P11) and Q3 backbones (ConvNeXt-T, ViT at λ ∈ {0, 2})?

## Design (code: `scripts/r5_q5_frozen_runs.py`; manifest from `scripts/r5_q4_manifest.py --extra ...`)

- Run monitor: Mahalanobis (pooled + 1e-3·I) and kNN k=50 on each run's scoring representation (Q2/Q3: `z_lesion_norm`
  or the backbone's projection, as written in the run's manifest row), fit on the run's ID train.
- Frozen monitor: ImageNet ResNet-50 (P3 reference; primary) and the best P12 encoder (secondary; a second invocation
  with `--frozen <task>=<that encoder's directory>`), extracted once per task
  on frame CSVs in the same row order as the run features (`r5_frozen_extract.py --frames_dir <task frames>
  --out_root results/paperB/r5/frozen/features_<task>`). For Q2/Q3 (lesion-level ISIC) these are exactly the P3/P12
  features; for Q1 the Q1 pipeline writes the frames (path, label_idx, group = patient).
- Same ID test and OOD sets for both monitors; row-count match asserted per task and reported.

## CI

Frozen monitor: percentile 95%, 2,000 draws, rng 20261005, ID and OOD resampled by the `groups` stored with the frozen
features (lesion / patient / image). Run monitor: mean ± s.d. over seeds per (family, λ); rule A cells come from the Q1–Q3
items' own reports (not recomputed here).

## PREDICTED OUTCOME (written now)

- Q2/Q3 (ISIC → pad_heldout): frozen ResNet-50 0.81–0.86 (same number for every run of a task), above the run monitor
  wherever the run monitor is < 0.6 (λ = 2 for CDAN and for ConvNeXt-T / ViT if they invert).
- Q1 (CXR → Shenzhen): frozen monitor ≥ 0.85 (adult CXR from another country and scanner; ImageNet features separate
  acquisition differences well); run monitor at λ = 2 below the frozen one by ≥ 0.2 if Q1 inverts, by ≥ 0.05 otherwise.

## Gates (flag, never drop)

Missing frozen features for a task → the task's row reads "frozen features missing" with the directory; the run
monitor is still reported. Row-count mismatch → flagged, no comparison drawn.

## Outputs

`results/paperB/r5/q5/`: `summary.csv`, `results.json`, `REPORT.md` (one section per task, LaTeX rows, a paragraph each).

## Cost

GPU: frozen extraction per new task ≈ 0.1–0.3 GPU-h (CXR: NIH + Kermany + Shenzhen test frames, ResNet-50); 0 for Q2/Q3
(P3/P12 features reused). CPU ≈ 0.2 h per task.

## Stop rules

Runs only after the Q1–Q3 run lists exist; no Q1–Q3 run is trained for this item.

## Deviations
