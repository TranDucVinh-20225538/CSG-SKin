# Paper B R5 / P4 — robustness: tail truncation, covariance shrinkage (CPU) and adversary LR ×10 / ×100 (GPU)

Committed before any P4 run (no P4 output exists at this commit; no GPU job submitted). Fixed now; deviations appended.

## Question

Is the lesion-level inversion (Table 1: Mahalanobis 0.403 on pad_heldout at λ = 2) an artefact of (i) a few extreme
scores, (ii) the ridge 1e-3 covariance estimate, or (iii) the adversary learning rate (×30 in every published run)?

## Part 1 — CPU (code: `scripts/r5_p4_robustness.py`)

Data: cached features of the 35 lesion-level runs; fit ISIC lesion-level train; ID = lesion-level test; OOD =
pad_heldout, Fitzpatrick17k. Representations: `z_lesion_norm` (primary), `backbone_raw_lesion` (secondary).
Variants of the Table 1 Mahalanobis:
- Tail truncation (Phase 14 A2 rule): drop scores above the pooled (ID ∪ OOD) p90 / p95 / p99, AUROC on the bulk; ID-only
  rule: drop ID scores above the ID p95 / p99, OOD untouched. Dropped counts per side reported.
- Trimmed fit: refit after removing the 5 % ISIC-train points with the largest own-class distance.
- Shrinkage: ridge eps ∈ {0, 1e-6, 1e-4, 1e-3 (Table 1), 1e-2, 1e-1, 1}; Ledoit–Wolf; OAS; per-class covariances.
Headline cells: Ledoit–Wolf and pooled-p95 drop at λ = 2, PREREGISTER pooled bootstrap (2,000 draws, rng 20261005,
shared across seeds; ID by lesion group, pad_heldout by patient, Fitzpatrick by image) and rule A (pooled CI < 0.5 and
≥ 3/5 per-seed CIs < 0.5). Every other cell: mean ± s.d. over seeds and k/5 seeds below 0.5.

## Part 2 — GPU (marked GPU; code: `scripts/r5_p4_advlr_train.py`, `slurm/r5_p4_advlr.sbatch`, report `scripts/r5_p4_advlr_report.py`)

Lesion-level retrain at λ = 2 with `--adv_lr_multiplier` 10 and 100 (published: 30), seeds 42/52/62, everything else
the R2 item-4 recipe (`train_r2_item4_lesion_split` → `train_phase3_sweep`, 40 epochs, Phase 2 checkpoint reuse off).
The wrapper exists because `train_r2_item4_lesion_split.main` looks for features under a tag without the `_advlr<m>`
suffix that `train_phase3_sweep` appends to the run name (the λ = 2 ×30 features already exist there, so extraction
would be skipped); the wrapper writes runs, checkpoints and features under `results/paperB/r5/p4/advlr/` and
`checkpoints/r5_p4_advlr/` with the suffixed tag. The existing script is not modified.
Report: Table 1 columns (leakage, ID bal acc, ID ECE, OOD ECE, Mahalanobis pad_heldout, Fitzpatrick, X-dom bal acc) and
the five detectors for ×10 / ×30 (existing runs, same seeds) / ×100; pooled bootstrap with k/3 seeds (n = 3, so rule A
here means 3/3).

## Seeds, λ grid, OOD sets

Part 1: all 7 λ, 35 runs. Part 2: λ = 2, seeds 42/52/62 × {×10, ×100} = 6 runs. OOD: pad_heldout, Fitzpatrick17k.

## Gates (flag, never drop)

- Part 2: adversary-collapse flag if leakage ≥ 0.85 at ×10 (adversary too weak to align) — then the run is a
  "not aligned" cell, reported, not excluded. Divergence flag if ID bal acc < 0.30.
- Part 1: a variant that drops > 20 % of OOD images is flagged.

## PREDICTED OUTCOME (written now)

- Part 1: every truncation keeps the λ = 2 mean below 0.5 (pooled p95 0.39–0.43; Phase 14 image-level: 0.416); ID-only
  drop at p95 0.42–0.47. Shrinkage moves λ = 2 AUROC by ≤ 0.03 for eps ≤ 1e-2, LW, OAS; eps = 1 (near-Euclidean)
  0.40–0.48; per-class covariances 0.35–0.48. λ = 0 stays ≥ 0.75 under every variant. Rule A at λ = 2 holds for LW and
  p95-drop on pad_heldout and Fitzpatrick (as for the Table 1 estimator).
- Part 2: leakage ×10 0.58–0.70, ×100 0.52–0.58; Mahalanobis pad_heldout ×10 0.40–0.52, ×100 0.37–0.45; OOD ECE ≥ 0.6
  in both. The inversion does not need the ×30 choice; it weakens as leakage rises.

## Outputs

Part 1: `results/paperB/r5/p4/` (`per_run/*.json`, `headline_bootstrap.json`, `summary.csv`, `REPORT.md`).
Part 2: `results/paperB/r5/p4/advlr/` (`runs/`, `features/`, `per_run/*.json`, `bootstrap.json`, `summary.csv`, `REPORT.md`).

## Cost

Part 1: GPU 0; CPU ≈ 1.5 h on 8 CPUs (35 runs × 2 reps × 12 estimators; 1,536-d backbone dominates) + 0.3 h bootstrap.
Part 2: 6 runs × ≈ 1.6 GPU-h (measured R2 item 4: 83–110 min per run) ≈ 9.6 GPU-h, ≈ 5 h wall at 2 GPUs (`%2`);
report CPU < 0.3 h.

## Stop rules / budget

Part 2 cap 12 GPU-h. If the first two runs (×10 s42, ×100 s42) both diverge (ID bal acc < 0.30), stop and report.

## Deviations
