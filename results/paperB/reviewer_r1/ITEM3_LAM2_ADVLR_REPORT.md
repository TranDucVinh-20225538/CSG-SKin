# Item 3 — λ=2 adversary LR sensitivity (retrained)

Canonical ×30 @ λ=2 may reuse Phase-2 ckpt; ×10/×100 **must** train fresh (`reused_phase2_ckpt: false`).

## Mahalanobis pad_heldout (3 seeds)

| adv_lr × | mean ± std | per-seed |
|---:|---:|---|
| 30 | 0.4320 ± 0.0235 | [0.43136248019554724, 0.4088504541931415, 0.4558985019730031] |
| 10 | 0.4181 ± 0.0109 | [0.4160583378262015, 0.40837470630975103, 0.4298089952182652] |
| 100 | 0.4816 ± 0.0195 | [0.4651642846196167, 0.5031229568475171, 0.4765094658944446] |

**Verdict:** **×10:** all seeds below 0.5 (objection answered at weaker adversary). **×100:** mean below 0.5 but not every seed (report per-seed; avoid “all multipliers” wording).
