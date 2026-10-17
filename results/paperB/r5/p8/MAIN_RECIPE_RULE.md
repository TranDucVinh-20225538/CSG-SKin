# Main-recipe rule (decisions/paperB_r5_p8.md, 2026-10-15 23:48)

λ = 0, lesion-level ISIC validation split, last.ckpt for both recipes, seeds matched.

| seed | A4 val bal acc | old val bal acc | A4 − old |
|---:|---:|---:|---:|
| 42 | 0.5260 | 0.4455 | +0.0805 |
| 52 | 0.4697 | 0.4095 | +0.0602 |
| 62 | 0.4649 | 0.4402 | +0.0248 |
| 72 | 0.4499 | 0.4615 | -0.0116 |
| 82 | 0.4885 | 0.4902 | -0.0017 |

D = +0.0304, 95 % paired t-interval [-0.0186, +0.0794] (df = 4).
D ≥ 0.03: True; CI excludes 0: False.
Outcome: **old** main recipe (RECIPE=auto → current); P8 ext: skip.

LaTeX row (D & CI lo & CI hi & outcome):

```
+0.030 & -0.019 & +0.079 & old \\
```

