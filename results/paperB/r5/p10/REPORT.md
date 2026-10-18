# Paper B R5 P10: Camelyon17 two-hospital adversary toolbox

Design: `decisions/paperB_r5_p10.md`. Gate: held-out hospital probe drop ≥ 0.10 vs λ = 0 mean (0.959); otherwise **invariance not reached** (inconclusive). GPU-h used 0.00 of 40. Semantic OOD: NCT-CRC-HE-100K (CC-BY-4.0). B_heldout accuracy is reported next to OOD AUROC (competence vs detection).

**Checkpoint note (2026-10-18):** all 10 shift-arms selected `best_epoch=1` (warmup; see `BEST_EPOCHS.md`). Locked “invariance not reached” scores the pre-adversary encoder. Posthoc last-epoch probe/ID (derm-style `last.ckpt`) is descriptive only — `POSTHOC_LAST.md` when ready.

## Ladder (seed 42)

| rung | λ | site probe | Δprobe vs λ0 | ID acc | B_heldout acc | Maha B_heldout | kNN | Maha C | gate | invariance | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|

## Predicted outcome (committed before runs)

If the hospital probe falls by ≥ 0.10 vs λ = 0: detection on the held-out hospital inverts, and NCT-CRC-HE at least collapses. If the gate fails: invariance not reached.

