# Phase 15b status

Started 2026-09-28. Outputs only under `results/paperB/phase15b/`. Existing Phase 15 summaries were not rewritten.

## Gate (done)

`PHASE15_1_GATE.md`. Corrected rule: best-epoch adversary CE below ln 2 − 0.02, and leakage moves. λ=0 is `valid_control`. Returning to chance after that drop is the encoder winning.

25/25 valid (5 control + 20 valid). Inconclusive = 0. Leakage 0.970 → 0.763. Port verdict **sound**. `pending_or_mixed` cleared. λ=8 seeds 42/52/62 are valid (best CE 0.660/0.664/0.658 at epoch 1; last epoch back at ln 2).

## Queue

2026-10-01 23:40 +07: Jobs **62191** (MMD), **62192** (CORAL), **62193** (iWildCam). **Max 2 GPUs** concurrent on MMD/iWildCam (`arraytaskthrottle=2`). CORAL → after MMD array; iWildCam → after CORAL. Holds released; cancelled MMD tasks 6–9 requeued.

Four MMD runs had started (weight 300 seeds 42/52/62, weight 1000 seed 42) and wrote `last.ckpt` around epoch 3. Resume with the same sbatch (`--resume --skip_done`). CORAL and iWildCam had not started. iWildCam has only a dry-run config.

| job | what | not launched |
|---|---|---|
| 61355 array 0–11 | MMD weights 300, 1000, 3000, 10000 × seeds 42/52/62 | nothing past 10000 |
| 61356 | CORAL mean L2 vs covariance Frobenius from existing checkpoints | no new CORAL training |
| 61357 array 0–3 | iWildCam coarse λ ∈ {0, 0.1, 1, 10}, seed 42, 243-way head, 12 epochs, ResNet-50, 448 | dense grid; domain-count ablation {2, 5, 20, all} |

iWildCam dry-run: train 129809 / 243 locations, id_test 8154 (87 classes), OOD test 42791 (102 classes), shared classes 69, zero train–OOD location overlap. Both AUROCs (unrestricted and shared-class) are in the first run. kNN gallery is capped at 20000 class-stratified train embeddings; the other four detectors use the full train set.

Cross-setting figure waits until Item A or Item B has a result. WILDS stops if this coarse scan is inconclusive the way Camelyon17 was.
