# Paper B R11b — X3/X5 adversary collapse + X4 stronger λ

Locked before first run of each item. Append deviations only; never edit rules after a run starts.
Branch: `paperb/r11b-adversary` (from `paperb/final-numbers` @ 59079ef).
Outputs: `results/paperB/r11b/` only. Do not modify any `results/paperB/r11/` file.
Commits: author `vinhtranduc` only; no Co-Authored-By trailer. No `.tex` edits.

## Why

In every R11 X3 run at λ ≥ 2 the in-training adversary ends at accuracy 0.500 and loss 0.6931 (= ln 2) in all five seeds (λ = 0.5: 0.515–0.536, 0.689–0.692), while held-out V1 reads the domain at 0.75–0.82 (higher than λ = 0: 0.61). The adversary outputs a constant; the head was never pushed toward invariance. X5 uses the same head and adversary; its V1 at λ = 2 (0.63–0.70) is not below λ = 0 (0.63). Neither item tested the question.

Likely causes (not yet confirmed): dead ReLUs in `derm_domain_head` (ReLU + dropout 0.5), adversary LR × 30 tuned for the end-to-end model, GRL α already 0.918 in epoch 1, and a 16-d bottleneck (head ID 0.523 at λ = 0, below competence 0.528).

## Shared rules (as R11)

Rule A, degraded rule, bootstrap, V1, decisions file before the first run of each item, deviations appended, every run reported. Screens are blind to OOD: they read ID validation balanced accuracy and V1 only.

Competence threshold = 0.528 (= ERM 0.558 − 0.03). Invariant (X3/X5) = V1 ≤ 0.60. Invariant (X4) = V1 ≤ 0.55.

## Y1 — diagnose (CPU; existing X3 and X5 runs)

For every X3 and X5 run, from the saved head and adversary:

1. Final in-training adversary accuracy and loss (from `epoch_log.json`).
2. Fraction of adversary hidden units that are zero on every `pad_adv` and ISIC-train input (dead ReLUs), per layer.
3. A fresh adversary of the same architecture trained from scratch on the frozen head output (ISIC train vs pad_adv, 20 epochs, its own LR); accuracy on held-out ISIC test vs pad_heldout.
4. GRL unit test: one batch; head gradient from adversary loss vs adversary’s own gradient direction; report cosine.

Table: run × (1)–(4). Collapse confirmed if (1) ≈ 0.5 and (3) ≥ 0.70 in ≥ 4 of 5 seeds at λ = 2 (X3 shift 100 %).

## Y2 — X3 rerun with a working adversary

Screen (seeds 42, 52; λ ∈ {0.5, 2, 8}; shift 100 %). Six configurations:

latent ∈ {16, 128} ×
- **A:** LeakyReLU(0.2) in the adversary, no dropout, adversary LR × 30;
- **B:** as A with adversary LR × 1;
- **C:** as B with five adversary steps per head step and α ramped from 0 (standard DANN schedule α = 2/(1+exp(−10p)) − 1, no α_min).

Also λ = 0 per latent (seeds 42, 52) for the competence check.

Admissible at λ = 2: in-training adversary accuracy at last epoch not within 0.005 of 0.5 with loss within 0.001 of ln 2 in both seeds (not collapsed), and ID val balanced accuracy ≥ 0.528. Select admissible config with lowest two-seed mean V1 at λ = 2; ties within 0.01 → smaller latent, then A < B < C. No admissible → stop Y2 and Y3, report "adversary collapse not resolved".

Main: selected config, λ ∈ {0, 0.5, 2, 8} × seeds 42–82 at shift 100 %, and λ = 2 at shift 0 % (seeds 42–82). Report per run: ID bal acc, V1, final adversary accuracy, Y1 (3) fresh-adversary accuracy, all detectors and sets, Rule A.

Reading (λ = 2, shift 100 %): Competent = five-seed ID bal ≥ 0.528. Invariant = V1 ≤ 0.60. Case A / B / C as R11 X3; not competent → reported, settles neither; shift 0 % read as R1 in Case A only.

## Y3 — X5 rerun with Y2 configuration

Design as R11 X5 (transformed ISIC train images, class mix resampled to t ∈ {0, 0.25, 0.5, 0.75, 1}), head and adversary = Y2 selection, λ ∈ {0, 2}, three seeds.

Gate: dose–response read only if V1 at λ = 2 is ≤ 0.60 at every t and adversary has not collapsed (as Y2). Otherwise: "invariance not reached; not a test", report table, no reading.

Reading (if gate met): as R11 X5 — reproduces the inversion if Rule A met on ≥ 2 of 3 labelled unseen sets at t = 1 and on none at t = 0.

## Y4 — X4 BCN → HAM at stronger λ

Screen (seeds 42, 52, shift 100 %): λ ∈ {4, 8}, R11 X4 design otherwise. Select the smaller λ whose two-seed mean V1 ≤ 0.55; neither → stop, report.

Main: selected λ, seeds 42–82, at shift 100 % and shift 0 % (seeds 42, 52 from screen reused at shift 100 %).

Reading: as R11 X4, with invariant = V1 ≤ 0.55 required at both shift levels; if shift 0 % is not invariant, shift comparison reported but not read. Also report λ = 0 ID bal of R11 X4 next to main-setting 0.481 and whether ID at selected λ is within 0.03 of λ = 0.

## Deviations

_(append only)_

- 2026-11-11: Y2 screen completed (job 73478). No admissible configuration (all six fail ID val ≥ 0.528 at λ=2 both seeds; adversaries mostly not collapsed). Verdict **adversary collapse not resolved**. Stop Y2 main and Y3. SELECTION.json locked with `selected: null` before any main submit. Watchers must not auto-submit main; selection is committed/pushed first.
