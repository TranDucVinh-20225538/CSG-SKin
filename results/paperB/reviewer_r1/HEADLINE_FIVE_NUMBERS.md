# Five headline bootstrap numbers (Item 1 — paste block)

Prereg: 2000 resamples, percentile 95% CI, shared resample indices across seeds per draw.  
Primary @ **λ=2**, **`pad_heldout`**, unless noted.

| # | Quantity | Value |
|---|----------|--------|
| **1** | Mahalanobis — mean AUROC, 95% CI (5 seeds) | **0.427** [**0.414**, **0.439**] |
| **2** | Mahalanobis — seeds with entire per-seed 95% CI &lt; 0.5 | **5 / 5** |
| **3a** | kNN-50 — mean, 95% CI | **0.440** [**0.429**, **0.452**]; **5/5** per-seed CIs below 0.5 |
| **3b** | Cosine max — mean, 95% CI | **0.453** [**0.441**, **0.464**]; **4/5** per-seed CIs below 0.5 |
| **4** | HAM→BCN reverse Mahalanobis @ λ=1 (§4.8) — mean, 95% CI | **0.683** [**0.666**, **0.699**]; **3/3** per-seed CIs **above** 0.5 (image bootstrap, 3 seeds) |
| **5** | Fitzpatrick17k Mahalanobis @ λ=2 — mean, 95% CI | **0.391** [**0.386**, **0.396**]; **5/5** below 0.5 |

**Fitz caveat (prereg):** image-level bootstrap only (no patient IDs); intervals may be **anti-conservative (narrow)** vs true sampling uncertainty.

**Manuscript rule (primary):** **Rule A** — keep **“inverts below chance”** for Mahalanobis @ λ=2 on `pad_heldout`.

Source: `bootstrap_results.json` (+ `bootstrap_phase16_ham_bcn_ladv1.json` when job completes).
