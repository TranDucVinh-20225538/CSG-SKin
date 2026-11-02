# Paper B R9 E2: stronger λ = 0 recipe (80 epochs)

Design: `decisions/paperB_r9_e2.md`. Commit `d0acd3c`.

## Stage S

| recipe | epochs | seed | ID val | ID test | degraded |
|---|---:|---:|---:|---:|---|
| A4 | 80 | 42 | 0.472 | 0.487 | False |
| A4 | 80 | 52 | 0.459 | 0.505 | False |
| A7 | 80 | 42 | 0.484 | 0.539 | False |
| A7 | 80 | 52 | 0.473 | 0.505 | False |
| ERM | 80 | 42 | 0.518 | 0.557 | False |
| ERM | 80 | 52 | 0.515 | 0.558 | False |

Selection: **A7** (val means A4=0.466, A7=0.478).

ERM reference = 0.558 (max of 0.558 and 80ep two-seed mean 0.557).

### Gate S

**not met** — selected ID test mean = 0.522, threshold = ERM−0.03 = 0.528.

**competence gap not closed by an 80-epoch schedule**

