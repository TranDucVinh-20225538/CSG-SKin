# Y2 screen selection

Locked at `2026-11-11T11:52:27Z` (UTC) — before any Y2 main submit.

## Verdict

**adversary collapse not resolved**

No admissible configuration at λ=2 (both seeds). Admissible requires: (i) not collapsed on either seed (collapse = acc within 0.005 of 0.5 AND loss within 0.001 of ln 2); (ii) ID val balanced accuracy ≥ 0.528 on both seeds. Stop Y2 main and Y3; do not submit further jobs.

## Rules (order)

- Collapse (both seeds at λ=2): acc within 0.005 of 0.5 **and** loss within 0.001 of ln 2.
- Competence: **ID val** balanced accuracy ≥ 0.528 (not test).
- Select: lowest two-seed mean V1; ties within 0.01 → smaller latent, then A < B < C.
- No admissible → stop Y2 and Y3; do not submit main.

## Screen table (λ=2, shift 100 %)

| cfg | latent | admissible | V1 mean | id_val mean | per-seed | notes |
|-----|-------:|:----------:|--------:|------------:|----------|-------|
| A | 16 | False | 0.7115 | 0.4598 | s42 acc=0.541 loss=0.6883 coll=False id_val=0.473 V1=0.661; s52 acc=0.610 loss=0.6652 coll=False id_val=0.447 V1=0.762 | y2A_lat16_ladv2_shift100_s42 id_val=0.473<0.528; y2A_lat16_ladv2_shift100_s52 id_val=0.447<0.528 |
| B | 16 | False | 0.9193 | 0.4776 | s42 acc=0.514 loss=0.6957 coll=False id_val=0.482 V1=0.935; s52 acc=0.499 loss=0.6914 coll=False id_val=0.474 V1=0.903 | y2B_lat16_ladv2_shift100_s42 id_val=0.482<0.528; y2B_lat16_ladv2_shift100_s52 id_val=0.474<0.528 |
| C | 16 | False | 0.6771 | 0.4595 | s42 acc=0.570 loss=0.6812 coll=False id_val=0.447 V1=0.649; s52 acc=0.560 loss=0.6807 coll=False id_val=0.472 V1=0.705 | y2C_lat16_ladv2_shift100_s42 id_val=0.447<0.528; y2C_lat16_ladv2_shift100_s52 id_val=0.472<0.528 |
| A | 128 | False | 0.7401 | 0.4774 | s42 acc=0.554 loss=0.6868 coll=False id_val=0.486 V1=0.771; s52 acc=0.565 loss=0.6811 coll=False id_val=0.469 V1=0.709 | y2A_lat128_ladv2_shift100_s42 id_val=0.486<0.528; y2A_lat128_ladv2_shift100_s52 id_val=0.469<0.528 |
| B | 128 | False | 0.8849 | 0.4774 | s42 acc=0.535 loss=0.6928 coll=False id_val=0.495 V1=0.893; s52 acc=0.541 loss=0.6840 coll=False id_val=0.460 V1=0.877 | y2B_lat128_ladv2_shift100_s42 id_val=0.495<0.528; y2B_lat128_ladv2_shift100_s52 id_val=0.460<0.528 |
| C | 128 | False | 0.7357 | 0.4272 | s42 acc=0.665 loss=0.6125 coll=False id_val=0.437 V1=0.751; s52 acc=0.642 loss=0.6308 coll=False id_val=0.418 V1=0.720 | y2C_lat128_ladv2_shift100_s42 id_val=0.437<0.528; y2C_lat128_ladv2_shift100_s52 id_val=0.418<0.528 |

## Selected

```json
null
```

