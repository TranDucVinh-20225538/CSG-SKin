# Paper B R5 — packing test: one vs two training runs per GPU (user decision 2026-10-15)

## Question
Can two lesion-level training runs share one A100-80GB (a running run uses ≈ 7 GB and 50–98 % utilisation) without
changing results, and does it raise throughput enough to be worth it?

## Design
P8 grid configuration A1 (λ = 0, current data pipeline, 6 loader workers per run), `scripts/r5_pack_test.py`,
`slurm/r5_pack_test.sbatch`.
- Equality: Lightning deterministic mode, 2 epochs. Solo: seed 42 alone on a GPU. Packed: seed 42 and seed 52
  at the same time on one GPU. Compare packed seed 42 with solo seed 42: validation features/logits (bitwise and max
  |Δ|), and the summary metrics (balanced accuracy, AUROCs, ECE).
- Throughput: normal mode, 3 epochs. Solo seed 42 (8 CPUs) vs the packed pair (16 CPUs). Gain per GPU =
  2 × T_solo / T_packed. CPUs per run are reported too, because the 48-CPU user cap is shared with the CPU items.

## Acceptance rule (set by the user before the test)
Packing is adopted for the remaining GPU items only if both hold:
(a) the metrics are identical in deterministic mode, or every difference is well below the seed-to-seed spread
    (|ΔAUROC| < 0.005, and the same bound for accuracy and ECE); and
(b) the throughput gain is ≥ 1.4× per GPU.
If either fails, one run per GPU is kept. No item already running changes mid-table.

## Prediction
(a) identical in deterministic mode (no cross-process interaction on the GPU). (b) gain 1.2–1.5×: the loader uses
≈ 7.7 cores per run, so with 6 workers each the pair is likely limited by JPEG decoding.

## Cost
≈ 0.5 GPU-h on 2 GPU jobs (8 and 16 CPUs), under 1 h wall each.

## Result (2026-10-15, solo job 67962, packed job 67963, both on node002)

| check | solo | packed | outcome |
|---|---|---|---|
| (a) deterministic, 2 epochs, seed 42: 11 summary metrics and the five arrays of `isic_val.npz` | — | — | all identical (bitwise) |
| deterministic wall time | 249 s (1 run) | 517 s (2 runs together) | gain 2 × 249 / 517 = 0.96× |
| (b) normal mode, 3 epochs, wall time | 317 s (1 run, 8 CPUs) | 696 s (2 runs together, 16 CPUs) | gain 2 × 317 / 696 = **0.91×** |

(a) holds; (b) fails (0.91× < 1.4×): two runs on one GPU are slower in total than one after the other. Decision by
the rule above: **one run per GPU is kept** for all remaining items. Wall times include start-up and validation
feature extraction, the same in both arms. The prediction (gain 1.2–1.5×) was wrong; the pair appears limited by
image decoding on the CPU side, not by the GPU.

## Deviations
