# Camelyon adversary ladder pilot — verdict

**DANN did not remove site information on Camelyon17.**

No level passed (probe drop ≥ 0.10 and ID bal within 0.05 of λ = 0). No `SELECTED.json`.
No multi-seed arms. Held R2 released per standing queue rule (`AFTER_PILOT_DISPATCH.json`).

Audit of identical L1–L4 headline metrics: `AUDIT_identical_L1_L4.md` — **not** a config bug;
λ and adv LR were applied in training; identical evals come from `best_epoch=1` (warmup, GRL coef = 0).
