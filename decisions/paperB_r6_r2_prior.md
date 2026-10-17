# Paper B R6 / R2-prior — IWDAN with better class-prior estimates

Committed **before any R2-prior training, feature extraction, or scoring**. Fixed now; not changed after
results are seen. Deviations append at the end.

Does **not** change the R2 pre-committed fix reading, gap-recovered formula, or primary leakage variable
(`94e2399` pad_full). Reuses R2 **BBSE** (post-hoc on existing IWDAN λ = 2 runs) and **oracle** λ = 2 as
reference points — those arms are **not** retrained here.

## Question

Does a better estimate of the PAD class prior make label-shift-aware IWDAN meet the R2 “fixes” bar when the
R2 confusion-matrix / QP estimate did not (or improve the within-class held-out probe)?

## Queue (critical)

**P10 must not be delayed.** Amended 2026-10-17 ~18:20: R2-prior paused for EM NaN bugfix; **both GPUs to P10**
(interleaved natural/induced, `%2`). After P10 finishes, R2-prior (fixed EM + 50-label, interleaved by seed,
`%2`) runs on both GPUs; held R2 `68562_[16-39]` releases after R2-prior, then R1-hi.

- Never hold P10 for R2-prior.
- EM s42 run under the broken implementation is **excluded** (labelled `implementation failure (NaN)`); rerun s42
  with the same seed after the computation fix.

GPU order: **R1 → R2 (partial) → P10 (both GPUs) → R2-prior (both GPUs) → R2 remainder → R1-hi → …**

## Arms (new GPU trainings)

All: **IWDAN**, λ = **2**, RECIPE = **current**, primary = **`last.ckpt`**, seeds `(42, 52, 62, 72, 82)`,
same lesion-level ISIC + Phase 2 PAD split as R2. Domain loss and IW classification weighting as R2 IWDAN;
only the **source of w / π_T** changes.

| Arm | Tag suffix | Prior source | Setting label |
|-----|------------|--------------|---------------|
| **(i) EM** | `iwdan_em` | Saerens et al. (2002) EM prior adjustment on **unlabeled** pad_adv soft predictions | unsupervised DA (deployable in principle) |
| **(ii) 50-label** | `iwdan_n50` | Empirical class mix from **50 labelled pad_adv** images | **few-label target** — **not** unsupervised DA; reported as a different setting |

**Pre-specified for (i) EM:**

- Softmax probabilities are **temperature-scaled on ISIC validation** before EM (**yes** — locked).
- Fit a single temperature \(T > 0\) by NLL on ISIC val (classifier frozen); apply \(p = \mathrm{softmax}(z/T)\)
  on pad_adv for EM. **Report \(T\) per seed** in `summary.json` / REPORT.
- EM updates the target prior π; importance weights \(w_j = \pi_T(j) / P_S(y=j)\) (renormalised as R2 oracle).
- Update schedule: re-estimate π each epoch from current model (or once per epoch after a forward on pad_adv),
  with the same moving-average spirit as R2 IW (`ma = 0.5`) unless a numbered deviation says otherwise before
  the first seed. No PAD labels in any loss.

**Pre-specified for (ii) 50-label:**

- Per training seed \(s\), draw a **fixed** set of 50 pad_adv paths with `numpy` Generator seed
  `20261017 + s` (without replacement). Persist the path list under the run dir; never redraw after seeing
  metrics.
- π_T = empirical frequencies on those 50 labels (8-class; zeros allowed). w from π_T as oracle.
- Those 50 labels set w only; they do **not** enter the classification or domain loss. Remaining pad_adv
  stays unlabeled for the adversary stream.

**Reference points (no new GPU):**

- **BBSE:** post-hoc BBSE mix / TV from existing R2 IWDAN λ = 2 (`last.ckpt`) — same reporting as R2.
- **Oracle:** existing R2 `iwdan_oracle` λ = 2 (true pad_adv mix).

## Primary success criterion (unchanged R2 “fixes”)

Primary unseen sets: **Fitzpatrick17k, MED-NODE, SD-198** (pad_heldout reported, **not** counted).

An arm **fixes** a set if Mahalanobis class-conditional AUROC **95% CI lower bound > 0.5** (rule A / R2
bootstrap) **and** pad_full leakage (Table 1 probe) **≤ DANN λ = 2 leakage + 0.05**, where DANN = **R1
shift 100%, RECIPE=current, `last.ckpt`** (same baseline as R2 deviation 2026-10-17; headline leakage ≈ 0.634
→ threshold ≈ 0.684 — verify from the DANN cell at report time).

**Arm-level success:** fixes on **≥ 2 of 3** primary unseen sets. Applied separately to EM and to 50-label.
Degraded runs (ID bal < 0.25) flagged and excluded from the pooled CI used for “fixes”, as R2.

## Secondary criterion (labelled **added after R2**; does not override primary)

Within-class held-out domain probe **(c)** from `decisions/paperB_r6_class_aware_heldout.md`
(≥ 20 images / domain side, class-size-weighted mean bal_acc) ≤ **DANN’s** cell mean **0.618**
(R1 shift 100% λ = 2 from class-aware REPORT `68678`). Reported for EM and 50-label; success on secondary
alone is **not** called an R2 fix.

Also report for every arm (and references): **unweighted (a)** and **class-balanced V3 (b)** held-out probes
on the same protocol.

## Prior-error test (seed-level)

Across the **four estimators** {BBSE, EM, 50-label, oracle} × 5 seeds (up to 20 points; oracle TV = 0 vs
pad_adv by construction):

- \(x\) = total-variation error of estimated π vs **true pad_adv** mix (evaluation labels only).
- \(y\) = mean Mahalanobis AUROC over the three primary unseen sets (seed-level).

**Spearman** ρ(x, y). Bootstrap CI of ρ: 2,000 resamples of the seed-level pairs, rng **20261017**
(percentile 95%).

**Supported** if ρ ≤ **−0.5** and the CI **excludes 0**. Otherwise report not supported (no post-hoc
threshold change).

## Reported (every new arm + references)

- TV error **per seed** and **per class** (50-label is noisy for rare classes — state that in the table).
- Estimated π and true pad_adv π; L1 and TV.
- ID balanced accuracy; pad_full leakage; Mahalanobis (+ rule A) on pad_heldout / Fitz / MED / SD-198.
- Held-out probes (a)/(b)/(c); EM temperature \(T\).
- Primary fixes table; secondary within-class vs 0.618; prior-error Spearman.

Outputs: `results/paperB/r6/r2_prior/cur/` (`runs/`, `features/`, `REPORT.md`, `prior_error.json`).
Checkpoints: `checkpoints/r6_r2_prior/cur/`.

## Predicted outcome (written now)

- EM TV error between BBSE and oracle; Mahalanobis on Fitz/MED/SD between IWDAN-estimated and oracle.
- EM may reach “fixes” on ≥ 2/3 if estimation error was the R2 bottleneck; else both EM and 50-label fail →
  prior quality is not sufficient.
- 50-label: lower TV than EM on common classes, high variance on DF/VASC/SCC; ID bal closer to oracle
  (weights less pathological) but leakage may rise.
- Prior-error Spearman: ρ ≈ −0.6 to −0.9 if the story is “better prior → better detection”; if ρ ≈ 0,
  detection failure is not mainly prior TV.
- Secondary (c): EM within-class probe 0.55–0.62; success vs 0.618 uncertain.

## GPU-hour estimate

Basis ~1.6 GPU-h / run (R2 two-stream). **10** new runs → **~16 GPU-h** (+ 1 failed EM s42 ~1.1 GPU-h sunk,
not counted as a result). Cap **20 GPU-h**. CPU report + probes ~2 CPU-h after features exist. Wall with **2 GPUs**
after P10: ~5 waves × ~1.1 h ≈ **~5.5–6 h**.

## Stop rules

Same IW divergence stop as R2 (estimated/EM branch). 50-label: if a seed’s draw covers < 3 distinct classes,
flag and keep (no redraw). Cap 20 GPU-h.

## Deviations

- 2026-10-17 ~18:20 (+07), **order + computation bug fix** (no arm/design change). EM s42 produced NaN `pi`/`w`
  (temp-fit crashed on val 2-tuple unpack → T stuck at 1; mean-soft EM without log-space/guards). Run kept under
  `results/paperB/r6/r2_prior/cur/_impl_failure_nan_*/` labelled **implementation failure (NaN)** — **excluded from
  results**. Cancelled remaining EM array tasks; **both GPUs given to P10** (`%2`, interleaved). EM fix (log-space
  Saerens, clamp, convergence, BBSE fallback if non-convergent; temp-fit `_unify` 2-tuple + finite NLL) committed
  before rerun. After P10: R2-prior on both GPUs, interleaved EM/n50 by seed, **rerun EM s42 same seed**. Report
  per-seed prior TV for n50 before train and for EM each epoch in `iw_log`.
- 2026-10-17 ~22:00 (+07), **order only**. R2-prior keeps **both GPUs** until the array finishes; then Camelyon
  ladder pilot ∥ Q1 headroom (one GPU each); held R2 after both; then R1-hi.
