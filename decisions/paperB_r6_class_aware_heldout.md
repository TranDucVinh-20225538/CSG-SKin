# Paper B — class-aware held-out domain probes (CPU)

Committed before any new class-aware probe number is computed. Does **not** change the primary
leakage variable (`94e2399` pad_full) or the pre-committed R1 leakage-matched test / R2 fix reading.

## Question

On images the adversary never saw (P6 V1 pool: ISIC-test ∪ pad_heldout), how much domain signal
remains after removing or conditioning on class composition?

## Scope (runs with cached features; resume when missing land)

| Cell | Seeds | Feature root |
|------|-------|--------------|
| R2 IWDAN λ = 2 | 42–82 | `r6/r2/cur/features/` |
| R2 IWCDAN λ = 2 | 42–82 | same |
| R2 IWDAN oracle λ = 2 | 42–72 now; **s82 when `68562_14` finishes** | same |
| R1 shift 0% λ = 2 | 42–82 | `r6/r1/cur/features/` |
| R1 shift 100% λ = 2 | 42–82 | same |

Representation: `z_lesion_norm`. Probe seeds 42/52/62. GroupShuffleSplit test_size 0.3 by
ISIC lesion / PAD heldout patient (P6 V1).

## Probes (linear logistic, StandardScaler, max_iter 4000)

**(a) Unweighted** — P6 V1: ISIC-test ∪ pad_heldout, no sample weights.

**(b) Class-balanced (P6 V3)** — same split as (a); ISIC-side sample weights reweight the ISIC class
mix to the pad_heldout mix over the six shared classes (fit + balanced-accuracy weights), as in
`scripts/r5_p6_leakage_probe.py` V3.

**(c) Within-class** — for each ISIC class with **≥ 20 images on both domain sides** (ISIC-test and
pad_heldout), fit the same linear domain probe inside that class only (GroupShuffleSplit on that
subset). Report the **class-size-weighted mean** of per-class balanced accuracy, weights =
\(n^{\mathrm{ISIC}}_c + n^{\mathrm{PAD}}_c\) over eligible classes. Classes below the threshold are
skipped (logged). Pre-committed eligible set on the Phase 2 / lesion-level counts: classes
{1, 2, 3, 4, 7} = NV, BCC, AK, BKL, SCC (`LABELS` order MEL…SCC); MEL/DF/VASC skipped (PAD side < 20).

## Report

Cell means ± s.d. over training seeds; bootstrap CI of the seed-mean when n_seeds ≥ 3
(2,000 draws, rng 20261017). Oracle λ = 2 cell reported **as soon as all 5 oracle seeds** (incl. s82)
have per-run JSON. Partial oracle (4 seeds) may be logged but is not the oracle cell headline.

Outputs: `results/paperB/r6/class_aware_heldout/` (`per_run/`, `summary.csv`, `REPORT.md`).

## Predicted outcome (written now)

- (a) at R1 shift 100% λ = 2: held-out bal_acc ≈ 0.50–0.65; at shift 0% λ = 2: ≈ 0.75–0.90.
- (b) within ~0.05 of (a) on R1 (class composition not the whole story); larger (a)−(b) gap on IW/oracle
  if label-shift reweighting mainly removes class-prior domain signal.
- (c) between (a) and chance: if (c) ≈ 0.5 while (a) high → domain signal was mostly class mix; if
  (c) stays high → within-class domain residue.
- Oracle λ = 2: (a)/(b) higher than estimated IWDAN if oracle restores features that remain
  domain-decodable; (c) still the stress test for within-class invariance.

## Cost / queue

CPU only. Start immediately on existing features; `afterok:68562_14` adds oracle s82 then writes the
oracle-complete REPORT. Does not hold GPU jobs.

## Deviations

(none yet)
