# R11b GPU / CPU estimate (before first GPU job)

Basis: R11 X3 array job 72999 ≈ **1.5 min/run** on cached features; R11 X4 array 73068 ≈ **56 min/run** (image-level dual). Config C adds 5 adversary steps → ≈5× feature train wall (~7–8 min). No GPU job submitted until this file exists.

| Item | Runs (approx.) | Wall per run | GPU-h | Notes |
|------|----------------|--------------|-------|-------|
| Y1 diagnose | 25 X3 + 30 X5 | CPU | **0 GPU** | Fresh-adv 20 ep on latent; ~2–4 CPU-h |
| Y2 screen | 40 (= 2×3×3×2 + 4 λ0) | 2–8 min | **2.0–3.5** | A/B ~2 min; C ~8 min |
| Y2 main | 25 | 2–8 min | **1.0–3.0** | Selected config only |
| Y2 score / Rule A | — | CPU | 0 | Bootstrap on features |
| Y3 (if Y2 ok) | 30 | 2–8 min | **1.0–2.5** | Order: ≈1–2 GPU-h |
| Y4 screen | 4 | ~56 min | **3.7** | λ ∈ {4, 8} × seeds 42, 52 |
| Y4 main | ≤ 8 new (+2 screen reused) | ~56 min | **7.5** | Selected λ × 5 × 2 shifts |
| Y4 fwd/score | — | GPU+CPU | **2–6** | As R11 X4 pipeline |

**Totals (order band):** Y2 ≈ **3–5 GPU-h**; Y3 ≈ **1–2**; Y4 ≈ **14–21** (train + fwd/score). Y1 CPU only. First GPU job = Y2 screen after Y1 verdict.
