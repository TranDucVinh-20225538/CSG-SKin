# Camelyon adversary ladder pilot — verdict

**Correct statement (locked protocol):** no post-adversary checkpoint passed the ID-select
criterion; the site-removal test did not take place.

All five levels selected `best_epoch = 1` (adv warmup, GRL coef = 0). Locked probe / ID numbers
are from that pre-adversary encoder. Do **not** report “DANN did not remove site information on
Camelyon17” as the factual outcome of a site-removal measurement — that measurement was never
run on a post-adversary encoder.

No `SELECTED.json`. No multi-seed arms. R2 released per queue rule.

Audit of identical L1–L4 locked metrics: `AUDIT_identical_L1_L4.md` (configs/logs applied; identity
from shared warmup `best.pt`). Best-epoch table (pilot + P10 shift-arms): `../BEST_EPOCHS.md`.

Post hoc (not locked): last-epoch probe / ID bal after retrain with `last.pt` — see
`summary.json` → `posthoc_last` when the posthoc job finishes; rollup `../POSTHOC_LAST.md`.
