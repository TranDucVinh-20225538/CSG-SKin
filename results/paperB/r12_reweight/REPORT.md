# Paper B R12 — fixed-image reweight + ERM-init REPORT

Locked design: `decisions/paperB_r12_reweight.md`. Not manuscript edits.
Protocol: fixed ISIC/PAD adversary image schedules; source-adv domain-loss weights `w=π_req/π_src` (mean 1 over stream). π_pad_adv from pad_adv labels (disclosed; not label-free).
Implementation: `align=iwdan_oracle` + `iw_cls=False` for λ>0; λ=0 is plain `dann`. Cross-protocol vs R1 resampling.

## Prior definitions

- **prior100** = π_ISIC (source class mix). With fixed images (`adv_mix=None`), `oracle_w = π_req/π_src` normalizes to **all ones** → domain loss is unweighted DANN on the natural ISIC adv stream (protocol-equivalent to R1 shift-100%, not a file reuse).
- **prior0** = π_pad_adv (from pad_adv labels at run start only; disclosed; never in task loss). Domain-loss reweight only (`align=iwdan_oracle`, `iw_cls=False`). **Not** paper IWDAN (which uses PAD prior differently and reports Fitz ~0.74).
- **prior50** = midpoint between π_ISIC and π_pad_adv.
- λ=0 arms ignore prior (plain `align=dann`); prior50/prior0 λ=0 rows are references to the same baseline features.

## Cell status

| tag | block | λ | prior | seed | init | status | note |
|---|---|---|---|---|---|---|---|
| A_ladv0_baseline_s42 | A | 0 | prior100 | 42 | scratch | ok |  |
| A_ladv0_baseline_s52 | A | 0 | prior100 | 52 | scratch | ok |  |
| A_ladv0_baseline_s62 | A | 0 | prior100 | 62 | scratch | ok |  |
| A_ladv0_baseline_s72 | A | 0 | prior100 | 72 | scratch | ok |  |
| A_ladv0_baseline_s82 | A | 0 | prior100 | 82 | scratch | ok |  |
| A_ladv0p5_prior100_s42 | A | 0.5 | prior100 | 42 | scratch | ok |  |
| A_ladv0p5_prior100_s52 | A | 0.5 | prior100 | 52 | scratch | ok |  |
| A_ladv0p5_prior100_s62 | A | 0.5 | prior100 | 62 | scratch | ok |  |
| A_ladv0p5_prior100_s72 | A | 0.5 | prior100 | 72 | scratch | ok |  |
| A_ladv0p5_prior100_s82 | A | 0.5 | prior100 | 82 | scratch | ok |  |
| A_ladv0p5_prior50_s42 | A | 0.5 | prior50 | 42 | scratch | ok |  |
| A_ladv0p5_prior50_s52 | A | 0.5 | prior50 | 52 | scratch | ok |  |
| A_ladv0p5_prior50_s62 | A | 0.5 | prior50 | 62 | scratch | ok |  |
| A_ladv0p5_prior50_s72 | A | 0.5 | prior50 | 72 | scratch | ok |  |
| A_ladv0p5_prior50_s82 | A | 0.5 | prior50 | 82 | scratch | ok |  |
| A_ladv0p5_prior0_s42 | A | 0.5 | prior0 | 42 | scratch | ok |  |
| A_ladv0p5_prior0_s52 | A | 0.5 | prior0 | 52 | scratch | ok |  |
| A_ladv0p5_prior0_s62 | A | 0.5 | prior0 | 62 | scratch | ok |  |
| A_ladv0p5_prior0_s72 | A | 0.5 | prior0 | 72 | scratch | ok |  |
| A_ladv0p5_prior0_s82 | A | 0.5 | prior0 | 82 | scratch | ok |  |
| A_ladv2_prior100_s42 | A | 2 | prior100 | 42 | scratch | ok |  |
| A_ladv2_prior100_s52 | A | 2 | prior100 | 52 | scratch | ok |  |
| A_ladv2_prior100_s62 | A | 2 | prior100 | 62 | scratch | ok |  |
| A_ladv2_prior100_s72 | A | 2 | prior100 | 72 | scratch | ok |  |
| A_ladv2_prior100_s82 | A | 2 | prior100 | 82 | scratch | ok |  |
| A_ladv2_prior50_s42 | A | 2 | prior50 | 42 | scratch | ok |  |
| A_ladv2_prior50_s52 | A | 2 | prior50 | 52 | scratch | ok |  |
| A_ladv2_prior50_s62 | A | 2 | prior50 | 62 | scratch | ok |  |
| A_ladv2_prior50_s72 | A | 2 | prior50 | 72 | scratch | ok |  |
| A_ladv2_prior50_s82 | A | 2 | prior50 | 82 | scratch | ok |  |
| A_ladv2_prior0_s42 | A | 2 | prior0 | 42 | scratch | ok |  |
| A_ladv2_prior0_s52 | A | 2 | prior0 | 52 | scratch | ok |  |
| A_ladv2_prior0_s62 | A | 2 | prior0 | 62 | scratch | ok |  |
| A_ladv2_prior0_s72 | A | 2 | prior0 | 72 | scratch | ok |  |
| A_ladv2_prior0_s82 | A | 2 | prior0 | 82 | scratch | ok |  |
| A_ladv0_prior50_s42 | A | 0 | prior50 | 42 | scratch | reference | features from A_ladv0_baseline_s42 |
| A_ladv0_prior50_s52 | A | 0 | prior50 | 52 | scratch | reference | features from A_ladv0_baseline_s52 |
| A_ladv0_prior50_s62 | A | 0 | prior50 | 62 | scratch | reference | features from A_ladv0_baseline_s62 |
| A_ladv0_prior50_s72 | A | 0 | prior50 | 72 | scratch | reference | features from A_ladv0_baseline_s72 |
| A_ladv0_prior50_s82 | A | 0 | prior50 | 82 | scratch | reference | features from A_ladv0_baseline_s82 |
| A_ladv0_prior0_s42 | A | 0 | prior0 | 42 | scratch | reference | features from A_ladv0_baseline_s42 |
| A_ladv0_prior0_s52 | A | 0 | prior0 | 52 | scratch | reference | features from A_ladv0_baseline_s52 |
| A_ladv0_prior0_s62 | A | 0 | prior0 | 62 | scratch | reference | features from A_ladv0_baseline_s62 |
| A_ladv0_prior0_s72 | A | 0 | prior0 | 72 | scratch | reference | features from A_ladv0_baseline_s72 |
| A_ladv0_prior0_s82 | A | 0 | prior0 | 82 | scratch | reference | features from A_ladv0_baseline_s82 |
| B_ladv0_initerm_s42 | B | 0 | prior100 | 42 | erm | ok |  |
| B_ladv0_initerm_s52 | B | 0 | prior100 | 52 | erm | ok |  |
| B_ladv0_initerm_s62 | B | 0 | prior100 | 62 | erm | ok |  |
| B_ladv0_initerm_s72 | B | 0 | prior100 | 72 | erm | ok |  |
| B_ladv0_initerm_s82 | B | 0 | prior100 | 82 | erm | ok |  |
| B_ladv2_initerm_s42 | B | 2 | prior100 | 42 | erm | ok |  |
| B_ladv2_initerm_s52 | B | 2 | prior100 | 52 | erm | ok |  |
| B_ladv2_initerm_s62 | B | 2 | prior100 | 62 | erm | ok |  |
| B_ladv2_initerm_s72 | B | 2 | prior100 | 72 | erm | ok |  |
| B_ladv2_initerm_s82 | B | 2 | prior100 | 82 | erm | ok |  |

## Block A pooled Mahalanobis (seed mean ± sd)

Seeds `{42,52,62,72,82}` (n=5). Primary detector: class-conditional Mahalanobis. Cluster-bootstrap CIs live in `CELLS.json` (`agg.*.bootstrap`).

| λ | prior | role | pad_heldout | Fitz | MED | SD-198 | SCIN | RuleA Fitz |
|---|---|---|---|---|---|---|---|---|
| 0 | prior100 | train | 0.806±0.035 | 0.728±0.050 | 0.589±0.025 | 0.722±0.055 | 0.714±0.046 | False |
| 0 | prior50 | reference | 0.806±0.035 | 0.728±0.050 | 0.589±0.025 | 0.722±0.055 | 0.714±0.046 | False |
| 0 | prior0 | reference | 0.806±0.035 | 0.728±0.050 | 0.589±0.025 | 0.722±0.055 | 0.714±0.046 | False |
| 0.5 | prior100 | train | 0.361±0.029 | 0.385±0.023 | 0.310±0.055 | 0.365±0.029 | 0.534±0.015 | True |
| 0.5 | prior50 | train | 0.488±0.032 | 0.458±0.076 | 0.332±0.050 | 0.447±0.070 | 0.572±0.070 | True |
| 0.5 | prior0 | train | 0.592±0.026 | 0.657±0.031 | 0.505±0.074 | 0.640±0.034 | 0.678±0.049 | False |
| 2 | prior100 | train | 0.368±0.017 | 0.340±0.065 | 0.298±0.032 | 0.325±0.069 | 0.519±0.073 | True |
| 2 | prior50 | train | 0.494±0.025 | 0.434±0.055 | 0.323±0.048 | 0.436±0.067 | 0.548±0.066 | True |
| 2 | prior0 | train | 0.595±0.019 | 0.661±0.032 | 0.510±0.034 | 0.657±0.035 | 0.689±0.030 | False |

## Paired contrasts prior0 − prior100 (pooled AUROC)

- prior0_minus_prior100_ladv0p5 / fitzpatrick17k: Δ=0.272±0.046 (n=5)
- prior0_minus_prior100_ladv0p5 / mednode: Δ=0.194±0.041 (n=5)
- prior0_minus_prior100_ladv0p5 / sd198: Δ=0.275±0.056 (n=5)
- prior0_minus_prior100_ladv0p5 / pad_heldout: Δ=0.232±0.036 (n=5)
- prior0_minus_prior100_ladv2 / fitzpatrick17k: Δ=0.321±0.079 (n=5)
- prior0_minus_prior100_ladv2 / mednode: Δ=0.211±0.041 (n=5)
- prior0_minus_prior100_ladv2 / sd198: Δ=0.332±0.085 (n=5)
- prior0_minus_prior100_ladv2 / pad_heldout: Δ=0.227±0.024 (n=5)

## Block B (ERM-init trainable dual)

Seeds `{42,52,62,72,82}`. ID bal = ID balanced accuracy (mean±sd). leakage V1 = held-out V1 probe `bal_acc_mean` averaged over seeds (0.921±0.008 at λ=0; 0.649±0.021 at λ=2).

| λ | pad_heldout | Fitz | ID bal | leakage V1 | RuleA Fitz |
|---|---|---|---|---|---|
| 0 | 0.817±0.025 | 0.706±0.034 | 0.491±0.008 | 0.921±0.008 | False |
| 2 | 0.400±0.024 | 0.428±0.013 | 0.484±0.014 | 0.649±0.021 | True |

## Reading (post-hoc vs locked questions)

- Recovery on unseen under fixed exposure → strengthens prior-weighting (not exposure) account.
- Recovery on unseen but not pad_heldout → limited for aligned-domain headline.
- No recovery → weakens prior-only explanation (retain).
- Block B: if ID↑ without invariance, does not answer competent-invariant inversion; if invariant without inversion, retain negative.
- R1 resampling is cross-protocol; gradient variance / optimization differ.

## Far-OOD

- Status: **unavailable** (no permitted non-skin panel on disk; no download).

## Historical anchors

- R1 Fitzpatrick λ2 shift100 mean **0.340**; shift0 series **0.405** (do not swap).
- Verify reported ~45%/43%/20% recovery against R1 source tables before any corrected summary.

## Negative / unresolved

- See paired contrasts and Rule A columns; do not expand grid to hunt inversion.

