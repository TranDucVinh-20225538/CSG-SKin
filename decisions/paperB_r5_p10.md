# Paper B R5 / P10 — Camelyon17: can a stronger adversary meet the "Leakage moved" gate?

Committed before any P10 job runs. Fixed now; not changed after results are seen. Deviations are appended at the end.

## Question

In R2 item 5 (two hospitals, A = 3 labelled, B = 0 adversary target, C = 2 unseen) the adversary learned (window CE
0.21–0.49 at λ ≤ 3) and was then driven to ln 2, but the site probe stayed at 0.915–0.950 against 0.952 at λ = 0: the
gate "Leakage moved" (≤ 0.852) failed and the Camelyon17 test of collapse/inversion could not be read. The
three-hospital Phase 12.1b design had an adversary stuck at ln 3 (implementation failure). Is the failure a property of
the adversary (a derm head that is fooled while a linear probe is not) or of the task? We strengthen the adversary with
the P9 toolbox under a 12 GPU-h cap and, if the gate is met, rerun the detectors.

## Design (scripts/train_r5_p10_camelyon_toolbox.py)

Two-hospital primary design only, identical to R2 item 5 except the adversary: same split file
(`results/paperB/r2/item5/split_camelyon2.npz`; A train / select / test, A slide-disjoint slides 30 + 32, B_adv /
B_heldout by slide, C), DenseNet-121 + BN + linear classifier, Balanced 16 A + 16 B batches, 10 epochs, AdamW
(1e-4, wd 1e-4, heads ×30 unless stated, classifier ×0.2), AMP, 'grl' placement, best epoch on A-select accuracy.
The three-hospital design is not re-run: its failure was an implementation failure already diagnosed, and the
two-hospital design is the one with a pre-committed gate.

Ladder (seed 42; sequential; rungs cumulative):

| Rung | λ | Adversary |
|---|---:|---|
| L1_bank | 1 | 3 heads (linear = probe family, derm 2-layer, 3-layer), loss averaged; 2 head-only inner steps per batch (fp32, on z.detach()); one head re-initialised every 500 steps (round robin, optimiser state cleared) |
| L2_ganin | 1 | + Ganin α schedule (starts at 0) + 1 adversary warm-up epoch (heads on z.detach(), no reversal) |
| L3_strong | 1 | + 5 heads (2 linear, derm, wide derm, 3-layer); 5 inner steps; head LR ×10 |
| L4_lam3 | 3 | L3 |
| L5_lam10 | 10 | L3 |

Stop rule: a rung skips itself (writes `skipped.json`) if an earlier rung met the gate or if P10 GPU-h used + 1.5 > 12.
Stage 2 (automatic, within the cap): the first gate-meeting rung at seeds 43 and 44; λ = 0 comparison = the R2 item-5
λ = 0 runs, seeds 42–44 (no new λ = 0 run).

## Data / splits

From `split_camelyon2.json` (R2 item 5): A = hospital 3 (patch-level train / select / test + 2 withheld slides),
B = hospital 0 split by slide 70/30 into B_adv / B_heldout, C = hospital 2. WILDS copy at
`DST-Skin/data/raw/wilds` (no staging).

## Seeds and λ grid

Ladder seed 42, λ ∈ {1, 3, 10} as in the table. Stage 2 seeds 43, 44 at the selected rung.

## Gate (pre-committed in R2 item 5, unchanged)

Site probe (logistic, A test vs B held-out, ≤ 10,000 patches per side, 70/30, probe seeds 42/52/62, balanced accuracy)
≤ 0.852 (= λ = 0 value 0.952 − 0.10) **and** A-test accuracy > 0.85. Detector headroom (λ = 0 Mahalanobis on
B_heldout 0.732) already holds.

## OOD sets, detectors, metrics

As R2 item 5: detectors fit on A train (Mahalanobis class-conditional shared covariance, kNN k = 50 on a 30k bank,
cosine, MSP, Energy); ID = A test (and A slide-disjoint); OOD = B_heldout (primary), B_adv, C. ID accuracy / balanced
accuracy / ECE, site probe also from A slide-disjoint. Adversary log per epoch: mean head CE / accuracy, inner-step CE,
re-init count.

## Controls

λ = 0 runs of R2 item 5 (seeds 42–44); the item-5 rows at λ ∈ {1, 3, 10} (one derm head) as the no-toolbox reference.

## CI / uncertainty

Per-seed values and SD over seeds, as R2 item 5. No bootstrap: B_heldout is 3 slides, so a slide-level cluster
bootstrap is degenerate and a patch-level one is anti-conservative (stated in the report).

## Reported cells

Ladder table (rung, λ, site probe, slide-disjoint probe, ID accuracy, min / last epoch head CE, last inner CE,
Mahalanobis and kNN on B_heldout, Mahalanobis on C, gate, GPU-h). If the gate is met: detector table for λ = 0
(item 5, 3 seeds) and the selected rung (3 seeds), LaTeX rows, one plain paragraph per rung.

## Predicted outcome (written before any score exists)

- L1 (linear head + inner steps): the largest change; site probe 0.85–0.92. L2: ±0.02. L3: a further 0.00–0.04.
  L4 / L5: 0.80–0.90, with A-test accuracy staying > 0.97.
- Gate met at L3–L5 with probability about one half. If met, B_heldout Mahalanobis falls toward 0.5 (collapse) and
  does not go below 0.5 with a margin (same-modality prediction of R2 item 5: collapse yes, inversion no).

## Outputs

`results/paperB/r5/p10/ladder/<tag>/`, `stage2/<tag>/` (config.json, adversary_log.jsonl, summary.json); `summary.csv`,
`REPORT.md`. Checkpoints `checkpoints/r5_p10/`.

## Recipe hook

`--recipe current|p8`: only the classifier LR multiplier applies to Camelyon (current 0.2 = item 5); latent size, class
balance, batch size and epochs are derm-task settings, listed as not applicable in config.json.

## GPU-h estimate and basis

Basis: R2 item-5 run ≈ 1.4 GPU-h (DenseNet-121, 10 epochs, evaluation of all sets). Heads and inner steps act on a
1,024-d vector (≤ 0.1 h extra): ≈ 1.5 GPU-h per run. Ladder ≤ 5 runs ≈ 7.5; stage 2 ≈ 3.0; total ≤ 10.5, hard cap 12.
CPU: report ≈ 0.05 h.

## Stop rules and caps

Gate met → no further ladder rungs; stage 2 runs once. 12 GPU-h cap enforced by each job. A failed run is re-run once;
a second failure is reported.

## Deviations
