# Pilot audit — identical L1–L4 metrics

**Verdict:** not a config / checkpoint-sharing bug. Locked failure stands.

## Checks

| Level | config λ | config adv_lr × | best_epoch | ep1 loss_adv | ep10 loss_adv | ep10 coef | probe drop | encoder md5 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P_L1 | 5.0 | 30.0 | 1 | 0.2568 | 32.7296 | 4.9995 | 0.0133 | `e4ef7aca8dec` |
| P_L2 | 5.0 | 100.0 | 1 | 0.5968 | 37.2617 | 4.9995 | 0.0133 | `e4ef7aca8dec` |
| P_L3 | 10.0 | 30.0 | 1 | 0.2568 | 31.6111 | 9.9991 | 0.0133 | `e4ef7aca8dec` |
| P_L4 | 10.0 | 100.0 | 1 | 0.5968 | 111.5588 | 9.9991 | 0.0133 | `e4ef7aca8dec` |
| P_L5 | 20.0 | 100.0 | 1 | 0.5573 | 26.3891 | 19.9982 | 0.0089 | `931055c4843c` |

## Why L1–L4 match

best.pt is epoch 1 (adv warmup, GRL coef=0) for all P_L1–P_L5 because later epochs collapse id_select_acc. During warmup the encoder+classifier updates are identical for fixed seed when only adv_lr differs among head updates; L1 and L3 share the same best.pt bytes (λ unused while coef=0); L2 and L4 likewise. Site-probe/ID metrics therefore match to many decimals. Training logs after epoch 1 diverge (λ and adv_lr are applied).

- Output dirs and `best.pt` paths are per-level under `pilot/` and `checkpoints/r5_p10/pilot/`.
- L1 `best.pt` bytes == L3; L2 == L4 (warmup-only; λ does not enter the encoder while coef=0).
- L5 encoder differs (deeper head bank → different init path) but also selects epoch 1; drop 0.009.

## Locked conclusion

Under the locked protocol (best epoch by A-select accuracy; pilot pass = probe drop ≥ 0.10 and ID bal within 0.05 of λ=0): **no level passed**. Correct report: **no post-adversary checkpoint passed ID-select; the site-removal test did not take place.** No multi-seed arms. (Not “DANN could not remove site” — post-adversary encoders were never scored.)

