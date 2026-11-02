# Paper B R9 consolidated report

Generated at commit `d0acd3c`.

| item | report |
|---|---|
| e1 | yes |
| e2 | yes |

GPU hours: {'gpu_hours': 26.72, 'rows': [{'job': '71492_0', 'name': 'paperB-r9-e1', 'elapsed': '01:30:35', 'hours': 1.51, 'kind': 'gpu'}, {'job': '71492_1', 'name': 'paperB-r9-e1', 'elapsed': '01:30:13', 'hours': 1.504, 'kind': 'gpu'}, {'job': '71492_2', 'name': 'paperB-r9-e1', 'elapsed': '01:31:03', 'hours': 1.518, 'kind': 'gpu'}, {'job': '71492_3', 'name': 'paperB-r9-e1', 'elapsed': '01:30:37', 'hours': 1.51, 'kind': 'gpu'}, {'job': '71492_4', 'name': 'paperB-r9-e1', 'elapsed': '01:31:04', 'hours': 1.518, 'kind': 'gpu'}, {'job': '71492_5', 'name': 'paperB-r9-e1', 'elapsed': '01:31:13', 'hours': 1.52, 'kind': 'gpu'}, {'job': '71492_6', 'name': 'paperB-r9-e1', 'elapsed': '01:31:09', 'hours': 1.519, 'kind': 'gpu'}, {'job': '71492_7', 'name': 'paperB-r9-e1', 'elapsed': '01:30:51', 'hours': 1.514, 'kind': 'gpu'}, {'job': '71492_8', 'name': 'paperB-r9-e1', 'elapsed': '01:31:06', 'hours': 1.518, 'kind': 'gpu'}, {'job': '71492_9', 'name': 'paperB-r9-e1', 'elapsed': '01:31:06', 'hours': 1.518, 'kind': 'gpu'}, {'job': '71494_0', 'name': 'paperB-r9-e2s', 'elapsed': '02:14:20', 'hours': 2.239, 'kind': 'gpu'}, {'job': '71494_1', 'name': 'paperB-r9-e2s', 'elapsed': '02:09:34', 'hours': 2.159, 'kind': 'gpu'}, {'job': '71494_2', 'name': 'paperB-r9-e2s', 'elapsed': '02:11:37', 'hours': 2.194, 'kind': 'gpu'}, {'job': '71494_3', 'name': 'paperB-r9-e2s', 'elapsed': '02:14:13', 'hours': 2.237, 'kind': 'gpu'}, {'job': '71494_4', 'name': 'paperB-r9-e2s', 'elapsed': '01:22:48', 'hours': 1.38, 'kind': 'gpu'}, {'job': '71494_5', 'name': 'paperB-r9-e2s', 'elapsed': '01:22:01', 'hours': 1.367, 'kind': 'gpu'}, {'job': '71540', 'name': 'paperB-r9-e1-fin', 'elapsed': '00:59:42', 'hours': 0.995, 'kind': 'cpu'}, {'job': '71551', 'name': 'paperB-r9-e2s-fin', 'elapsed': '00:00:25', 'hours': 0.007, 'kind': 'cpu'}], 'note': 'E1 10 three-stream + E2S 4 dual + 2 ERM; finalize CPU excluded from gpu_hours'}.

---

# Paper B R9 E1: six-class control

Design: `decisions/paperB_r9_e1.md`. Commit `d0acd3c`.

## Reading-rule case (λ = 2 decides)

**Support not the driver.**

Detail: recovery (labelled) = {'fitzpatrick17k': 0.058, 'mednode': 0.048, 'sd198': 0.035}; Rule A (six-class) = {'fitzpatrick17k': 'met', 'mednode': 'met', 'sd198': 'met'}.

π_6cls (run config): [0.182, 0.518, 0.134, 0.035, 0.106, 0.0, 0.0, 0.025]; TV→ISIC=0.019, TV→PAD=0.579 (ref 0.019 / 0.579).

## Table (Mahalanobis mean ± sd; paired Δ six−ref with seed-bootstrap 95% CI; recovery; Rule A on six-class)

| λ | set | six-class | shift 100% | shift 0% | Δ vs 100% | Δ vs 0% | recovery | Rule A |
|---:|---|---:|---:|---:|---|---|---:|---|
| 2 | pad_heldout (not counted) | 0.372 ± 0.034 | 0.368 ± 0.017 | 0.514 ± 0.123 | +0.005 [-0.029, +0.037] | -0.141 [-0.268, -0.061] | 0.031 | met |
| 2 | fitzpatrick17k | 0.362 ± 0.032 | 0.340 ± 0.065 | 0.719 ± 0.069 | +0.022 [-0.041, +0.086] | -0.358 [-0.409, -0.308] | 0.058 | met |
| 2 | mednode | 0.312 ± 0.050 | 0.298 ± 0.032 | 0.591 ± 0.114 | +0.014 [-0.022, +0.047] | -0.279 [-0.384, -0.191] | 0.048 | met |
| 2 | sd198 | 0.339 ± 0.039 | 0.325 ± 0.069 | 0.720 ± 0.065 | +0.014 [-0.055, +0.089] | -0.381 [-0.438, -0.327] | 0.035 | met |
| 2 | scin (not counted) | 0.496 ± 0.045 | 0.519 ± 0.073 | 0.719 ± 0.041 | -0.023 [-0.071, +0.028] | -0.223 [-0.268, -0.178] | -0.115 | not met |
| 0.5 | pad_heldout (not counted) | 0.374 ± 0.037 | 0.361 ± 0.029 | 0.498 ± 0.025 | +0.014 [-0.015, +0.044] | -0.124 [-0.159, -0.086] | 0.101 | met |
| 0.5 | fitzpatrick17k | 0.394 ± 0.043 | 0.385 ± 0.023 | 0.672 ± 0.046 | +0.009 [-0.010, +0.026] | -0.278 [-0.291, -0.265] | 0.031 | met |
| 0.5 | mednode | 0.332 ± 0.036 | 0.310 ± 0.055 | 0.525 ± 0.050 | +0.022 [-0.016, +0.059] | -0.192 [-0.217, -0.172] | 0.103 | met |
| 0.5 | sd198 | 0.384 ± 0.051 | 0.365 ± 0.029 | 0.664 ± 0.058 | +0.019 [+0.000, +0.038] | -0.280 [-0.301, -0.257] | 0.063 | met |
| 0.5 | scin (not counted) | 0.557 ± 0.034 | 0.534 ± 0.015 | 0.718 ± 0.036 | +0.022 [-0.008, +0.051] | -0.162 [-0.176, -0.148] | 0.121 | not met |

## ID / leakage (six-class)

- λ=2.0: ID bal 0.467 ± 0.010, leakage V1 0.679 ± 0.064, NV-nearest pad 0.618 ± 0.035, degraded seeds [].
- λ=0.5: ID bal 0.470 ± 0.010, leakage V1 0.604 ± 0.065, NV-nearest pad 0.609 ± 0.056, degraded seeds [].


---

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

