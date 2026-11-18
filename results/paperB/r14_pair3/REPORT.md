# REPORT — Paper B R14 ISIC 2019 → MILK10k dermoscopic

Generated: 2026-11-18T08:15:26+00:00
Commit at gate: `4a8b823`

## Verdict

**STOP-degenerate** at stage 1. No further jobs. Settles nothing on inversion in this pair (gate fired before the λ=2 reading rules).

- `r14_ladv0_prior100_s42` ID bal acc **0.185** < 0.25 (zero recall on AK/DF/VASC/SCC; majority NV+BCC).
- Optional stage-1 `r14_ladv2_prior0_s42` also ID bal **0.152** < 0.25.

## Stage-1 table

| tag | degenerate | ID bal | V1 | Maha Fitz | Maha MED | Maha SD |
|---|---|---:|---:|---:|---:|---:|
| r14_ladv0_prior100_s42 | False | 0.185 | 0.686 | 0.835 | 0.791 | 0.793 |
| r14_ladv2_prior100_s42 | False | 0.458 | 0.617 | 0.365 | 0.456 | 0.459 |
| r14_ladv2_prior100_s52 | False | 0.453 | 0.615 | 0.393 | 0.512 | 0.487 |
| r14_ladv2_prior0_s42 | False | 0.152 | 0.660 | 0.535 | 0.427 | 0.535 |

Other gates (not triggered): STOP-gap requires V1>0.70 on both λ=2 prior100 seeds (observed 0.617 / 0.615); STOP-headroom requires λ=0 Maha<0.65 on all three labelled unseen (observed ~0.79–0.84).

## Data (pre-GPU)

- API overlap: **exact** box check (859 lesions / 1,895 ISIC 2019 images / remaining counts).
- pHash: dropped **67** MILK images vs ISIC 2019; listed PAD 7 / Fitz 0 / MED 0 / SD-198 6.
- TV(ISIC train, milk_adv) = **0.563** after pHash filter. NV 0.508→0.077, BCC 0.131→0.567.
- attribution uniform (`MILK study team`); no per-site AUROC.

## Jobs / compute

- prepare: 74360; stage1 array: 74361 (0–3%2); full train: not submitted.
- GPU: 2× A100-SXM4-80GB; ~4 stage-1 runs × ~40 epochs ≈ ~8 GPU-h train + CPU prepare/API/pHash.
- Commits: decisions `3db0318` … gate `4a8b823` (+ this REPORT).

## Deviations

See `decisions/paperB_r14_pair3.md` (append-only). No manuscript edits. No retuning.
