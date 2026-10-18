# P10 / pilot — `best_epoch` (locked ID-select checkpoint)

All runs below selected **epoch 1** (adversary warmup; GRL coef = 0). Locked gate / “invariance”
metrics therefore score the **pre-adversary** encoder. `last.pt` was not saved in these runs;
posthoc last-epoch probe/ID is queued separately (`posthoc_last` in summary after retrain).

| stage | tag | best_epoch | locked probe drop | locked ID bal | last-epoch id_select (history only) |
|---|---|---:|---:|---:|---:|
| pilot | camelyon2_P_L1_ladv5_s42 | 1 | 0.0133 | 0.9879 | 0.428 |
| pilot | camelyon2_P_L2_ladv5_s42 | 1 | 0.0133 | 0.9879 | 0.432 |
| pilot | camelyon2_P_L3_ladv10_s42 | 1 | 0.0133 | 0.9879 | 0.432 |
| pilot | camelyon2_P_L4_ladv10_s42 | 1 | 0.0133 | 0.9879 | 0.568 |
| pilot | camelyon2_P_L5_ladv20_s42 | 1 | 0.0089 | 0.9859 | 0.879 |
| shift_arms | camelyon2_natural_l2_ladv2_s42 | 1 | 0.0133 | 0.9879 | 0.432 |
| shift_arms | camelyon2_natural_l2_ladv2_s52 | 1 | 0.0097 | 0.9867 | 0.570 |
| shift_arms | camelyon2_natural_l2_ladv2_s62 | 1 | 0.0111 | 0.9873 | 0.570 |
| shift_arms | camelyon2_natural_l2_ladv2_s72 | 1 | 0.0172 | 0.9863 | 0.871 |
| shift_arms | camelyon2_natural_l2_ladv2_s82 | 1 | 0.0115 | 0.9881 | 0.568 |
| shift_arms | camelyon2_induced_l2_ladv2_s42 | 1 | 0.0080 | 0.9850 | 0.515 |
| shift_arms | camelyon2_induced_l2_ladv2_s52 | 1 | 0.0084 | 0.9837 | 0.843 |
| shift_arms | camelyon2_induced_l2_ladv2_s62 | 1 | 0.0173 | 0.9826 | 0.908 |
| shift_arms | camelyon2_induced_l2_ladv2_s72 | 1 | 0.0131 | 0.9890 | 0.927 |
| shift_arms | camelyon2_induced_l2_ladv2_s82 | 1 | 0.0180 | 0.9834 | 0.922 |

**15 / 15 runs: `best_epoch = 1`.**
