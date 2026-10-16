# Paper B R5 / D1 follow-up — placebo (λ0→λ0) and dose (λ0→λk)

Committed before any placebo or dose number is computed. Fixed now; not changed after results are seen.
Deviations are appended at the end. Parent item: `decisions/paperB_r5_d1.md` (criterion and per-image
definitions unchanged unless stated).

## Question

(1) Is the D1 association (domain score vs pull) **DANN-specific**, or does the same ρ_dom appear when the
“target” run is another λ = 0 model of a different seed (placebo: no adversary change)?
(2) How does ρ_dom vary with adversary strength λ ∈ {0.25, 0.5, 1, 2, 4, 8} (dose), relative to the
λ = 0 → λ = 2 result already reported?

## Data and runs (CPU; all needed runs exist)

Old Table 1 recipe, lesion-level, cached features only:
`results/paperB/r2/item4/features/runB_orth1_ladv{0,0p25,0p5,1,2,4,8}_s{42,52,62,72,82}/`
(7 λ × 5 seeds). No new training; no phase16 / BCN↔HAM in this follow-up.

Sets (5 lesion-level, same as D1): pad_heldout, Fitzpatrick17k, MED-NODE, SD-198, SCIN.
Representations: **latent** = `z_lesion_norm` (primary); **backbone** = `backbone_raw_lesion` (secondary).
Bootstrap units unchanged from D1: pad_heldout by patient; Fitzpatrick and SD-198 by P13 / P13-method
clusters; SCIN by case; MED-NODE by image. PREREGISTER: 2,000 draws, rng seed 20261005; CI = percentile
95% of the seed-pair mean. Draws shared across seeds within a cell and between the two arms of a paired
contrast.

## Shared definitions (also to be stated in `d1_domain/REPORT.md`)

- **backbone** = `backbone_raw_lesion`: pre-projection features of the same DANN-trained model (they receive
  the reversed gradient). Not a separately trained backbone.
- **pull** for a contrast λ = 0 → λ = k (same seed unless noted) is the change in Mahalanobis distance from
  the λ = 0 run to the λ = k run, with the same primary / secondary pull formulae as D1
  (`d₀/med₀ − d_k/med_k`, etc.). Domain score and orthogonal control always come from the **source** λ = 0
  model (never from λ = k).

## (1) Placebo: λ = 0 → λ = 0 (different seeds)

For each **source seed** s ∈ {42, 52, 62, 72, 82}, the **placebo partner** is the next seed in the cycle
42→52→62→72→82→42 (five ordered pairs; ≥ 4 required). Domain score and o are fit/scored on the source
λ = 0 model (s). Placebo pull uses Mahalanobis d from λ = 0 (s) vs λ = 0 (partner). The DANN arm for the
same source uses d from λ = 0 (s) vs λ = 2 (s), as in D1.

Per set and representation, per source seed: ρ_dom and ρ_orth (Spearman vs primary pull) for the DANN arm
and for the placebo arm. Statistic = mean over the five source seeds.

**Primary test (paired bootstrap).** On the same image resample (same groups, same source seed),
Δρ_dom = ρ_dom(λ0→λ2) − ρ_dom(placebo). Report mean Δρ_dom and its 95% CI (percentile of the seed mean
over shared draws).

**“DANN-specific” (fixed now).** Supported if, on the **latent** representation, Δρ_dom ≥ 0.15 **and** the
95% CI excludes 0, in **≥ 4 of the 5** lesion-level sets. Backbone is reported with the same numbers and does
not change the verdict.

**CI rule (fixed now).** Two separate CIs (one for ρ_dom(λ0→λ2), one for ρ_dom(placebo)) that happen to
include / exclude 0 do **not** count as evidence of a difference. Only the CI of the paired Δρ_dom does.

Secondary (report, no pass/fail): ρ_dom and ρ_orth points for both arms; Δ = ρ_dom − |ρ_orth| per arm;
partial ρ given P₀ as in D1.

## (2) Dose: λ = 0 → λ ∈ {0.25, 0.5, 1, 2, 4, 8}

Matched seeds (same s for λ = 0 and λ = k). Domain score from λ = 0 (s). Pull = change from λ = 0 to λ = k
as above. Per set × representation × k: mean ρ_dom (and ρ_orth) over 5 seeds, with the usual group bootstrap
CI of the seed mean.

**Prediction / test (only this).** Paired bootstrap of δ = ρ_dom(λ0→λ2) − ρ_dom(λ0→λ0.25), same images and
seeds. Supported if mean δ > 0 and the 95% CI excludes 0, in ≥ 4 of 5 sets (latent primary; backbone
secondary, no verdict change). **Do not** predict monotonicity across λ; Table 1 is not monotone at λ = 4–8.
All levels are reported descriptively (table of ρ_dom by λ).

## Predictions (written before any follow-up number exists)

1. **Placebo.** On pad_heldout and Fitzpatrick (latent), ρ_dom(placebo) is near 0 (|ρ_dom| ≤ 0.15) and
   Δρ_dom ≥ 0.15 with CI excluding 0. Across the five sets, DANN-specific holds in 4 of 5 (latent); MED-NODE
   or SCIN is the plausible miss (small n or weak inversion).
2. **Placebo control.** ρ_orth(placebo) is similar in magnitude to ρ_orth(λ0→λ2) (both track “far from ID”
   images); the gap is in ρ_dom, not in the orthogonal control.
3. **Dose.** ρ_dom(λ0→λ2) exceeds ρ_dom(λ0→λ0.25) with paired CI excluding 0 in ≥ 4 of 5 sets (latent).
   Descriptive: ρ_dom at λ = 0.5 and λ = 1 lies between 0.25 and 2 on most sets; λ = 4 and λ = 8 are not
   assumed larger than λ = 2 (Table 1 non-monotone).
4. **Backbone.** Same qualitative pattern as latent for placebo Δρ_dom and for the λ = 2 vs 0.25 contrast,
   with Δρ_dom about 0.05–0.10 smaller than latent.

## Outputs

`results/paperB/r5/d1_followup/`: `placebo.json`, `dose.json`, `summary.csv`, `REPORT.md` (LaTeX rows).
Parent D1 `results/paperB/r5/d1_domain/REPORT.md` gains two clarifying sentences (backbone definition; pull =
λ = 0 → λ = k) without changing any D1 number or verdict.

## Compute

CPU only: extend `scripts/r5_d1_domain_direction.py` (or `scripts/r5_d1_followup.py`) reading existing
item4 features; one `pB-d1fu` job, 8 CPUs, ≈ 1–2 h. No GPU. Cap 4 CPU-h.

## Deviations
