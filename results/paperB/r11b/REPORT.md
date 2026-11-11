# Paper B R11b report

Branch `paperb/r11b-adversary` from `paperb/final-numbers` @ `59079ef`.
Outputs only under `results/paperB/r11b/`. No edits to `results/paperB/r11/` or `.tex`.

## Y1 — adversary collapse diagnose

**Verdict: collapse confirmed** (5/5 seeds at X3 λ=2, shift 100 %).

Rule: final in-training adversary ≈ 0.5 / ln 2, and fresh adversary held-out acc ≥ 0.70, in ≥ 4 of 5 seeds.

| seed | final acc_adv | final loss_adv | fresh heldout | dead relu0 | dead relu1 | GRL cos |
|-----:|--------------:|---------------:|--------------:|-----------:|-----------:|--------:|
| 42 | 0.500 | 0.6931 | 0.766 | 0.156 | 1.000 | 0.000 |
| 52 | 0.500 | 0.6932 | 0.854 | 0.312 | 0.938 | 0.000 |
| 62 | 0.501 | 0.6929 | 0.849 | 0.219 | 0.875 | 0.000 |
| 72 | 0.501 | 0.6932 | 0.854 | 0.219 | 0.938 | 0.000 |
| 82 | 0.500 | 0.6932 | 0.796 | 0.312 | 0.938 | 0.000 |

Full table (25 X3 + 30 X5): `y1/TABLE.md`.

Notes from the table:
- At λ = 0, dead-ReLU fractions are 0 and GRL cosine is −1 (unit test passes); fresh adv still reads domain (~0.67–0.71).
- At λ ≥ 2, relu1 is mostly dead (0.75–1.0), in-train adv sits at chance/ln 2, fresh adv recovers 0.69–0.95 — the head retained domain signal the training adversary stopped using.
- Several collapsed runs show GRL cosine 0 (vanishing feature gradients), not a broken GRL implementation (healthy runs stay at −1).

Job: `73473`. CPU on compute nodes.

## Y2 — X3 screen

Screen job `73478` COMPLETED (40/40). **No admissible configuration.**

Verdict: **adversary collapse not resolved** → stop Y2 main and Y3.

| cfg | latent | V1 mean λ=2 | id_val mean λ=2 | why rejected |
|-----|-------:|------------:|----------------:|--------------|
| A | 16 | 0.712 | 0.460 | id_val < 0.528 |
| A | 128 | 0.740 | 0.477 | id_val < 0.528 |
| B | 16 | 0.919 | 0.478 | id_val < 0.528 |
| B | 128 | 0.885 | 0.477 | id_val < 0.528 |
| C | 16 | 0.677 | 0.460 | id_val < 0.528 |
| C | 128 | 0.736 | 0.427 | id_val < 0.528 |

In-training adversary was mostly *not* at chance (acc ~0.51–0.66); gate failed on competence (ID val). Full: `y2/SCREEN.md`.

## Y3 — skipped

Stopped with Y2 (no admissible config).

## Y4 — in progress

BCN → HAM at λ ∈ {4, 8}.

## GPU / CPU hours

See `GPU_ESTIMATE.md`. Actuals to be filled at close.

## Commit hashes

- Base: `59079ef`
- R11b work: _(pending)_
