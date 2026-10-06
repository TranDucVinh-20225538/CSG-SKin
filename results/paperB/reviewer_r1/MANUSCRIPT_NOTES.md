# Manuscript notes (reviewer R1 — bootstrap & λ wording)

## Uncertainty at λ=2 (pad_heldout, Mahalanobis)

Report **three** quantities together; do not treat the bootstrap CI as total uncertainty.

1. **Point estimate:** mean AUROC over 5 seeds = **0.427** (bootstrap JSON: 0.4267).
2. **Bootstrap 95% CI on the mean** (stratified resample of test ID + PAD patients; **same resample indices for all seeds within each draw**, then average): **[0.414, 0.439]**. This reflects **test-set sampling error** only; sharing resamples across seeds makes this interval **narrower** than a full hierarchical bootstrap over seeds + patients.
3. **Seed dispersion:** **±0.025** (s.d. across 5 seed point AUROCs; ddof=1 ≈ 0.0246).
4. **Strict seed check:** **5/5** seeds have bootstrap 95% CI entirely below 0.5.

Suggested prose pattern: *“Mahalanobis AUROC on held-out PAD was 0.427 (bootstrap 95% CI 0.414–0.439 on the test resample; 0.427 ± 0.025 across five training seeds); every seed’s resampled CI lay below chance.”*

## λ sweep shape (Table 1 / Fig 1)

PAD Mahalanobis mean is **U-shaped**, not monotone in λ_adv:

| λ | Mean (approx.) |
|---:|---:|
| 0.25 | 0.513 |
| 2 | **0.427** (minimum) |
| 8 | 0.494 |

Do **not** write “higher λ always worsens detection.” Describe a **sharp drop toward chance at low λ**, a **sub-chance minimum near λ=2**, and **partial recovery at λ=8** (still near chance).

## Wording: “cliff” at λ=0.25

Until Item 4 completes (n=5 seeds at λ=0.25 and λ=1):

- Replace **“cliff”** with **“drops sharply to chance”** (bootstrap CI at λ=0.25 **straddles 0.5**: [0.498, 0.529], k=0/3).
- Revisit “cliff” only if extra seeds tighten the λ=0.25 story.

## GPU priority (ops)

1. **Item 4** — `slurm/reviewer_r1_item4_extra_seeds.sbatch` (4 runs).
2. **Item 3** — adversary LR ×{10,100} (after Item 4 queued or in parallel if quota allows).
3. Item 5/6 optional.

Submit Item 4 when DST `rigor-*` jobs release the 4-GPU user quota on `defq`.
