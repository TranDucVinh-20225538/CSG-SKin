# P10 / pilot — posthoc last-epoch site probe & ID bal

Descriptive only. Locked gate still uses ID-select `best.pt`. Derm R1/R2 use `last.ckpt`.

| stage | tag | best_ep | locked drop | locked ID bal | last drop | last ID bal | last probe |
|---|---|---:|---:|---:|---:|---:|---:|
| pilot | camelyon2_P_L1_ladv5_s42 | 1 | 0.0133 | 0.9879 | **pending** | **pending** | **pending** |
| pilot | camelyon2_P_L2_ladv5_s42 | 1 | 0.0133 | 0.9879 | **pending** | **pending** | **pending** |
| pilot | camelyon2_P_L3_ladv10_s42 | 1 | 0.0133 | 0.9879 | **pending** | **pending** | **pending** |
| pilot | camelyon2_P_L4_ladv10_s42 | 1 | 0.0133 | 0.9879 | **pending** | **pending** | **pending** |
| pilot | camelyon2_P_L5_ladv20_s42 | 1 | 0.0089 | 0.9859 | **pending** | **pending** | **pending** |
| shift_arms | camelyon2_induced_l2_ladv2_s42 | 1 | 0.0080 | 0.9850 | **pending** | **pending** | **pending** |
| shift_arms | camelyon2_induced_l2_ladv2_s52 | 1 | 0.0084 | 0.9837 | **pending** | **pending** | **pending** |
| shift_arms | camelyon2_induced_l2_ladv2_s62 | 1 | 0.0173 | 0.9826 | **pending** | **pending** | **pending** |
| shift_arms | camelyon2_induced_l2_ladv2_s72 | 1 | 0.0131 | 0.9890 | **pending** | **pending** | **pending** |
| shift_arms | camelyon2_induced_l2_ladv2_s82 | 1 | 0.0180 | 0.9834 | **pending** | **pending** | **pending** |
| shift_arms | camelyon2_natural_l2_ladv2_s42 | 1 | 0.0133 | 0.9879 | **pending** | **pending** | **pending** |
| shift_arms | camelyon2_natural_l2_ladv2_s52 | 1 | 0.0097 | 0.9867 | **pending** | **pending** | **pending** |
| shift_arms | camelyon2_natural_l2_ladv2_s62 | 1 | 0.0111 | 0.9873 | **pending** | **pending** | **pending** |
| shift_arms | camelyon2_natural_l2_ladv2_s72 | 1 | 0.0172 | 0.9863 | **pending** | **pending** | **pending** |
| shift_arms | camelyon2_natural_l2_ladv2_s82 | 1 | 0.0115 | 0.9881 | **pending** | **pending** | **pending** |

Pending: 15 / 15 runs without `posthoc_last`.

Reading guide (not locked): if last-epoch probe drop is large while ID bal collapses, the Camelyon story is that DANN removes site only by breaking the classifier — unlike derm R1/R2 last.ckpt readouts.

