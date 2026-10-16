# Paper B R5 / P10 — Camelyon17: held-out hospital gate, semantic OOD, 40 GPU-h cap

Committed before any P10 job runs. Fixed now; not changed after results are seen. Deviations are appended at the end.
GPU order (2026-10-16): **R1 → R2 → P10 → P9 → P11 → P7(b) → Q1 → Q3**. P10 runs immediately after R2.

## Question

In R2 item 5 (two hospitals, A = 3 labelled, B = 0 adversary target, C = 2 unseen) the adversary learned but the
site probe barely moved, so collapse/inversion on Camelyon could not be read. With the P9 adversary toolbox under a
**40 GPU-h** cap: does the held-out-hospital probe fall by ≥ 0.10 vs λ = 0? If yes, does detection on the held-out
hospital invert, and does a semantic H&E OOD (non-lymph-node tissue) at least collapse?

## Design (scripts/train_r5_p10_camelyon_toolbox.py)

Two-hospital primary design only, identical to R2 item 5 except the adversary: same split file
(`results/paperB/r2/item5/split_camelyon2.npz`; A train / select / test, A slide-disjoint slides 30 + 32, B_adv /
B_heldout by slide, C), DenseNet-121 + BN + linear classifier, Balanced 16 A + 16 B batches, 10 epochs, AdamW
(1e-4, wd 1e-4, heads ×30 unless stated, classifier ×0.2), AMP, 'grl' placement, best epoch on A-select accuracy.
The three-hospital design is not re-run.

Ladder (seed 42; sequential; rungs cumulative):

| Rung | λ | Adversary |
|---|---:|---|
| L1_bank | 1 | 3 heads (linear = probe family, derm 2-layer, 3-layer), loss averaged; 2 head-only inner steps per batch (fp32, on z.detach()); one head re-initialised every 500 steps (round robin, optimiser state cleared) |
| L2_ganin | 1 | + Ganin α schedule (starts at 0) + 1 adversary warm-up epoch (heads on z.detach(), no reversal) |
| L3_strong | 1 | + 5 heads (2 linear, derm, wide derm, 3-layer); 5 inner steps; head LR ×10 |
| L4_lam3 | 3 | L3 |
| L5_lam10 | 10 | L3 |

Stop rule: a rung skips itself (writes `skipped.json`) if an earlier rung met the gate or if P10 GPU-h used + 1.5 > 40.
Stage 2 (automatic, within the cap): the first gate-meeting rung at seeds 43 and 44; λ = 0 comparison = the R2 item-5
λ = 0 runs, seeds 42–44 (no new λ = 0 run).

## Data / splits

From `split_camelyon2.json` (R2 item 5): A = hospital 3 (patch-level train / select / test + 2 withheld slides),
B = hospital 0 split by slide 70/30 into B_adv / B_heldout, C = hospital 2. WILDS copy at
`DST-Skin/data/raw/wilds` (no staging).

## Seeds and λ grid

Ladder seed 42, λ ∈ {1, 3, 10} as in the table. Stage 2 seeds 43, 44 at the selected rung.

## Gate (pre-committed; replaces the absolute 0.852 cut)

**Gate = held-out hospital site probe drops by ≥ 0.10 relative to the matched λ = 0 mean** (R2 item 5 λ = 0,
seeds 42–44; probe = logistic, A test vs B held-out, ≤ 10,000 patches per side, 70/30, probe seeds 42/52/62,
balanced accuracy). Formally: `probe_λ0 − probe_run ≥ 0.10`.

- If the gate is **met**: read detector tables (collapse / inversion).
- If the gate is **not** met: report **“invariance not reached”** and treat the Camelyon detection readout as
  **inconclusive** (do not claim collapse or inversion from that run).

A-test accuracy is still reported; it is not part of the gate (tolerance vs competence is read from held-out
hospital accuracy next to OOD AUROC — see below).

## OOD sets, detectors, metrics

As R2 item 5: detectors fit on A train (Mahalanobis class-conditional shared covariance, kNN k = 50 on a 30k bank,
cosine, MSP, Energy); ID = A test (and A slide-disjoint); OOD = B_heldout (primary), B_adv, C.

**Held-out hospital competence vs detection:** report **accuracy (and balanced accuracy) on B_heldout** in the same
tables as OOD AUROC on B_heldout — competence (can the model classify at the held-out hospital) next to detection
(can a detector separate A-test from B_heldout).

**Semantic OOD (H&E, non-lymph-node tissue):** NCT-CRC-HE-100K colorectal patches (Kather et al., Zenodo
10.5281/zenodo.1214456), licence **CC-BY-4.0** (checked 2026-10-16; attribution required). Use a fixed subsample
for evaluation (paths listed in `results/paperB/r5/p10/nct_crc_he_eval_paths.txt` when fetched). Detectors scored
A-test vs NCT-CRC-HE; no NCT labels enter training. If NCT-CRC-HE cannot be fetched under the licence terms in this
environment, substitute another public H&E non-lymph-node tissue set and log a dated deviation naming the set and
licence.

## Controls

λ = 0 runs of R2 item 5 (seeds 42–44); the item-5 rows at λ ∈ {1, 3, 10} (one derm head) as the no-toolbox reference.

## CI / uncertainty

Per-seed values and SD over seeds, as R2 item 5. No bootstrap: B_heldout is 3 slides, so a slide-level cluster
bootstrap is degenerate and a patch-level one is anti-conservative (stated in the report).

## Reported cells

Ladder table (rung, λ, site probe, Δprobe vs λ = 0, slide-disjoint probe, ID accuracy, **B_heldout accuracy**,
min / last epoch head CE, last inner CE, Mahalanobis and kNN on B_heldout, Mahalanobis on C, Mahalanobis on
NCT-CRC-HE, gate / “invariance not reached”, GPU-h). If the gate is met: detector table for λ = 0 (item 5, 3 seeds)
and the selected rung (3 seeds), LaTeX rows, one plain paragraph per rung.

## Predicted outcome (written before any P10 score exists)

- If the hospital probe falls by ≥ 0.10 vs λ = 0 (gate met): detection on the held-out hospital **inverts**
  (Mahalanobis on B_heldout below 0.5 with margin across the stage-2 seeds), and the semantic OOD (NCT-CRC-HE)
  **at least collapses** (Mahalanobis toward 0.5; inversion on NCT not required).
- If the gate is not met: report “invariance not reached”; no collapse/inversion claim on Camelyon from P10.
- Ladder: L1 largest probe move; L2 ±0.02; L3 a further 0.00–0.04; L4/L5 may deepen the drop; A-test accuracy
  stays high (≥ 0.95) when the gate is met.

## Outputs

`results/paperB/r5/p10/ladder/<tag>/`, `stage2/<tag>/` (config.json, adversary_log.jsonl, summary.json); `summary.csv`,
`REPORT.md`. Checkpoints `checkpoints/r5_p10/`.

## Recipe hook

`--recipe current|p8`: only the classifier LR multiplier applies to Camelyon (current 0.2 = item 5); latent size, class
balance, batch size and epochs are derm-task settings, listed as not applicable in config.json.

## GPU-h estimate and basis

Basis: R2 item-5 run ≈ 1.4 GPU-h; heads/inner steps ≤ 0.1 h extra ≈ 1.5 GPU-h per run. Ladder ≤ 5 ≈ 7.5; stage 2 ≈ 3.0;
semantic-OOD forward passes within the same jobs. **Hard cap 40 GPU-h** (raised from 12 so the ladder + stage 2 +
NCT eval cannot be cut mid-rung by the old cap). CPU: report ≈ 0.1 h.

## Stop rules and caps

Gate met → no further ladder rungs; stage 2 runs once. 40 GPU-h cap enforced by each job. A failed run is re-run once;
a second failure is reported. Gate fail → “invariance not reached”.

## Deviations

- 2026-10-16 09:46 (+07), user-directed, before any P10 run. Gate redefined as held-out hospital probe drop ≥ 0.10
  vs λ = 0 (else inconclusive / “invariance not reached”); held-out hospital accuracy reported next to OOD AUROC;
  semantic OOD NCT-CRC-HE (CC-BY-4.0) added; prediction rewritten (gate met → hospital inverts, semantic OOD at least
  collapses); hard cap raised 12 → 40 GPU-h. Absolute probe cut 0.852 and the old “collapse yes, inversion no”
  same-modality prediction are superseded for P10 (R2 item 5 numbers remain the λ = 0 reference).
- 2026-10-16 14:30 (+07), before R2 starts. GPU order: P10 moved to immediately after R2 (before P9). NCT-CRC-HE
  download / eval manifest prepared so paths exist before R2 ends (`scripts/fetch_nct_crc_he.py`).
