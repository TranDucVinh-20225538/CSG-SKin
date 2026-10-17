# Paper B R5 / P10 Camelyon adversary ladder pilot

Committed **before any pilot training**. Fixed now; not changed after results are seen. Deviations append at the end.

Depends on P10 shift-arms result (2026-10-17): both natural and induced λ = 2 with the L3 toolbox left the
held-out hospital site probe almost unchanged (drop ≈ 0.01 ≪ 0.10) → **invariance not reached**. This pilot asks
whether a stronger adversary can meet the gate at all on Camelyon17.

## Question

Can a stronger DANN adversary (higher λ and/or adversary LR, then a setting beyond the current toolbox) drive the
held-out hospital site probe down by ≥ 0.10 vs λ = 0 without collapsing ID accuracy? If yes, at which weakest level,
and does that level then support the natural / induced label-shift arms?

## Pilot (seed 42, natural arm only)

Same data / split / DenseNet-121 as P10 shift-arms. Cap **10 GPU-h**.

**L1–L4 — current P10 toolbox unchanged**, vary only λ and adversary LR:

Toolbox (locked, identical to shift-arms `natural_l2`): `grl_schedule=ganin`, `adv_warmup_epochs=1`,
`adv_lr_multiplier` overridden per level, `adv_inner_steps=5`, `reinit_every_steps=500`, head bank
`[[1,1],[1,1],[2,1],[2,2],[3,1]]`.

| Level | λ | adv LR multiplier | Toolbox |
|-------|--:|------------------:|---------|
| **L1** | 5 | ×30 | current P10 (above) |
| **L2** | 5 | ×100 | current P10 |
| **L3** | 10 | ×30 | current P10 |
| **L4** | 10 | ×100 | current P10 |

**L5 — stronger than the current toolbox** (not a λ/LR-only tweak):

| | |
|--|--|
| λ | **20** |
| Head bank | **deepest**: `[[1,1],[1,1],[2,2],[3,1],[3,2]]` (upgrades L3’s derm head to wide×2 and adds a wide depth-3 head vs L3’s `[2,1]` / single `[3,1]`) |
| Re-init | every **100** steps (round-robin head re-init; vs 500 in the current toolbox) |
| Else | Ganin + warm-up 1 + inner steps 5 (same as L1–L4) |
| adv LR | ×100 |

Fixed order L1 → L5; **stop at the first level that passes**.

**Pass** (both required):

1. Site probe drop ≥ **0.10** vs λ = 0 mean (item5 seeds 42–44; same as P10).
2. ID balanced accuracy within **0.05** of λ = 0 mean ID bal (item5 seeds 42–44).

If a level fails, continue to the next. If **no** level passes within the cap: stop and report
**"DANN did not remove site information on Camelyon17"** — do not launch the multi-seed arms.

## Multi-seed arms (only if a level is selected)

Run **5 seeds** `(42, 52, 62, 72, 82)` × **both arms** (natural TV = 0.056; induced A 80% / B 20% tumour,
TV = 0.60) at the **selected** level only (same λ, adv LR, head bank, re-init policy).

**Seed rule (locked):**

- A seed that **fails** the gate at the chosen level is **reported and excluded** from arm-level detection labels.
- **No climbing the ladder per seed.**
- If fewer than **4 of 5** seeds pass in an arm → that arm is **"invariance not reached"** (no collapse / inversion /
  survives claim).

Detection labels, CIs (B = 2,000, rng 20261017, percentile 95%), and Δ vs λ = 0: same as
`decisions/paperB_r5_p10.md` (inversion / collapse / survives / indeterminate). Gate fail still forbids those claims.

## Predicted outcome (written now)

- L1–L2 may still fail the probe drop (λ = 2 already failed); L3–L5 more likely to move the probe if anything will.
- Risk: high adv LR / λ degrades ID bal past the 0.05 band → level fails even if probe moves.
- If the pilot selects a level and ≥ 4/5 seeds pass: natural still predicted collapse-band; induced inversion-band
  (same TV thresholds as P10). If the pilot fails entirely: Camelyon site signal is not removable by this DANN family.

## Outputs

`results/paperB/r5/p10/pilot/` (`SELECTED.json` if a level passes; per-level `summary.json` / `skipped.json`).
Multi-seed arms under `results/paperB/r5/p10/shift_arms_pilot/` (or dated subdir stated at launch). Cap 10 GPU-h for
the pilot ladder; multi-seed arms are outside that cap and are scheduled only after SELECTED.json exists.

## Queue

After R2-prior finishes (both GPUs freed): pilot on **one GPU** (serial `%1`), Q1 headroom gate on the **other**.
Held R2 cells release after both; then R1-hi.

## Deviations

- 2026-10-17 ~22:05 (+07), **before any pilot run**. L1–L4 redefined to keep the full current P10 toolbox (incl. adv LR
  base overridden only by the level’s ×30/×100, re-init every 500 steps); L5 replaced by λ = 20 + deepest head bank +
  re-init every 100 steps (stronger than toolbox), not “L4 + re-init every 2 epochs”.
