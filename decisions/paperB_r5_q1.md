# Paper B R5 / Q1 — second task: chest X-ray (NIH → Kermany pediatric, OOD Shenzhen)

Committed before any Q1 job runs (no image, feature or model of this task has been touched by Paper B code at this
commit). Fixed now; not changed after results are seen. Deviations are appended at the end.

## Question

Does the λ pattern of the main task (leakage falls, OOD AUROC on the adversary's target domain falls below 0.5)
reproduce on a second task with a different modality and a known, large label shift between the two training domains,
and what happens on a same-modality OOD set that neither the classifier nor the adversary has seen?

Two accounts make different predictions on Shenzhen (written here, before any score):

- §4.8 account: the adversary removes the source/target domain direction; on a same-modality unseen set this leaves
  less to separate, so Shenzhen Mahalanobis AUROC collapses toward 0.5 **without** inversion (pooled CI straddles 0.5,
  rule C).
- Label-shift account: NIH is 2.3 % pneumonia, Kermany 73 %; matching the two marginals moves Kermany-like images onto
  the NIH majority (normal) region, so OOD images are scored *more* in-distribution than ID test images; on Shenzhen
  this predicts **inversion** (rule A at λ = 2).

## Data and splits (scripts/r5_q1_cxr_data.py; images are the existing DST-Skin 256×256 grey stagings, not re-encoded)

| Domain | Role | Source on disk | Label definition |
|---|---|---|---|
| NIH ChestX-ray14 | labelled (domain 0) | `DST-Skin/data/staged/nih_256/images_0{01..12}.tar`, metadata `Data_Entry_2017_v2020.csv` | normal = `Finding Labels == "No Finding"`; pneumonia = `"Pneumonia"` is one of the `|`-separated labels (co-findings allowed); every other image excluded (50,328) |
| Kermany pediatric CXR v2 | adversary target (domain 1) | `DST-Skin/data/staged/p2a_ood_256/kermany_ped.tar` (map: `DST-Skin/results/r3/8/p2a/ood.csv.gz`) | folder NORMAL → normal; PNEUMONIA (bacteria + virus) → pneumonia. Never enters a loss |
| Shenzhen | OOD only | `DST-Skin/data/staged/p2a_ood_256/shenzhen.tar` | none (TB flag `_1` kept for descriptive subsets) |

Splits (patient-level; metadata-only dry run of the index script, numbers below):

- NIH: StratifiedGroupKFold(5, shuffle, seed 42) on image label, group = Patient ID: fold 0 = test; on the rest the
  same procedure, fold 0 = val (the R2 item-4 procedure with patient as the group). Patient overlap 0 for all pairs.
  863 patients have both normal and pneumonia images (kept; the split is by patient).
- Kermany: our own patient split (the official train/test split shares 170 patient IDs). Patient ID from the original
  file name: `person<N>_{bacteria,virus}_<k>[_<j>]` → `person<N>`; `[NORMAL2-]IM-<N>-<k>[-<j>]` → `[NORMAL2-]IM-<N>`.
  StratifiedGroupKFold(10, seed 42): folds 0–2 = `kerm_heldout` (secondary OOD, the pad_heldout analogue), folds 3–9 =
  `kerm_adv` (adversary stream). Normal and pneumonia IDs are disjoint name spaces, so one child with images in both
  folders cannot be linked (limitation, stated in the report).
- Shenzhen: all 662 images (one per patient).

Class mix per domain:

| Set | Images | Patients | Normal | Pneumonia | Pneumonia fraction |
|---|---:|---:|---:|---:|---:|
| nih_train | 39,546 | 16,031 | 38,630 | 916 | 0.023 |
| nih_val | 9,888 | 4,015 | 9,659 | 229 | 0.023 |
| nih_test | 12,358 | 5,006 | 12,072 | 286 | 0.023 |
| kerm_adv | 4,098 | 2,185 | 1,108 | 2,990 | 0.730 |
| kerm_heldout | 1,758 | 933 | 475 | 1,283 | 0.730 |
| kerm_full | 5,856 | 3,118 | 1,583 | 4,273 (2,780 bacterial, 1,493 viral) | 0.730 |
| shenzhen (OOD) | 662 | 662 | 326 normal | 336 TB | — |

Staging: each job extracts the needed PNGs (≈1.7 GB) to node-local `/tmp` (`r5_q1_cxr_data.py --stage_to`), removed
on exit. The index build (`--index`) reads metadata only; it is run as the first CPU job of the gate stage.

## Model and training (scripts/train_r5_q1_cxr.py)

- Dual-encoder CSG, EfficientNet-B3 × 2 (the main-task backbone family), `lesion_classes = 2`, `domain_classes = 2`,
  same latents (16 / 64), heads, GRL schedule, optimiser groups (main lr, adversary ×30, classifier ×0.2) and
  checkpoint rule (best val/acc = macro accuracy = balanced accuracy for 2 classes) as the main task. Recipe via
  `--recipe current|p8` (r5_recipe.py).
- Compatibility of the dual encoder with a 2-class grey task: no code change is needed. Inputs are grey PNGs loaded as
  3-channel RGB, so the lesion branch's internal grey conversion is an identity and both encoders see the same image;
  the colour/shape motivation of the two branches does not apply here (the context branch still predicts the domain,
  and the orthogonality term still acts). Stated in the report.
- Task adaptations (fixed now): class-weighted CE (inverse frequency, mean weight 1; NIH train is 2.3 % pneumonia),
  overridden by the recipe's class balance if it is not `none`; CXR augmentation (random resized crop 224, scale
  0.7–1; rotation ±10°; brightness/contrast 0.3; random erasing p 0.25; no flips, no hue/saturation); epochs set so the
  optimiser step count equals the main task (40 × 506 → 16 × 1,235 steps at batch 32 + 32).
- Batch: 32 NIH (classifier + adversary rows) + 32 kerm_adv (adversary rows only).

## Seeds and λ grid

- Headroom gate: λ = 0, seeds 42, 52, 62.
- Sweep (only if the gate is met): λ ∈ {0, 0.5, 2} × seeds 42, 52, 62, 72, 82 (λ = 0 seeds 42–62 are the gate runs).

## Headroom gate (stop rule)

Shenzhen Mahalanobis AUROC at λ = 0, PREREGISTER_BOOTSTRAP draws (ID resampled by NIH patient, Shenzhen by image).
"Clearly above chance" = **mean ≥ 0.60 and pooled-over-seeds 95 % CI lower bound ≥ 0.55 and every seed's 95 % CI lower
bound > 0.50**. Rationale: an inversion (< 0.5) or a collapse (≈ 0.5) can only be told apart from "no headroom" if the
λ = 0 value is at least 0.10 above 0.5 with a CI clear of 0.55. Prior evidence on disk: NIH-trained CNNs in the
leakage paper scored Shenzhen 0.59–0.64 (patient-disjoint Mahalanobis), Shenzhen-normal 0.44–0.54, Shenzhen-TB
0.72–0.75, so the gate may fail.

If the gate is not met the sweep is not launched (`launch_r5_q1.sh sweep` refuses), and the report lists the
replacement OOD candidates that are on disk:

1. Shenzhen TB subset only (336) — same file, already scored; a disease-shifted rather than domain-only OOD set.
2. Kermany held-out patients (`kerm_heldout`, 1,758 / 933) — the adversary's own domain (pad_heldout analogue).
3. Kermany OCT (`DST-Skin/data/staged/kermany_256.tar`) — retina, cross-modality far-OOD.
4. Brain MRI Cheng (`DST-Skin/data/staged/brain_cheng_256.tar`) — far-OOD.
5. PneumoniaMNIST / OCTMNIST — derivatives of Kermany CXR / OCT.

Not on disk: CheXpert, PadChest, MIMIC-CXR, Montgomery (download and data-use agreements needed). The choice among
candidates goes back to the user; it is not made by the report.

## OOD sets, detectors, metrics

- ID = NIH test; detectors fit on NIH train only (Mahalanobis class-conditional with shared covariance, kNN k = 50,
  cosine max on `z_lesion_norm`; MSP, Energy on logits).
- OOD primary: Shenzhen (all 662). Descriptive subsets: Shenzhen normal (326), Shenzhen TB (336).
- OOD secondary: `kerm_heldout` (per-class AUROC normal / pneumonia; decomposition check = OOD reweighted to the NIH
  class mix; majority-nearest fraction = share nearest to the NIH normal mean).
- Leakage (Table 1 analogue): logistic probe NIH test vs Kermany full, 70/30, probe seeds 42/52/62, balanced accuracy;
  also NIH test vs kerm_heldout and NIH val vs kerm_adv.
- ID: balanced accuracy, pneumonia AUROC, ECE (15 bins). X-domain: kerm_heldout balanced accuracy, AUROC, ECE.
- Adversary gate (phase15_common.adversary_gate) reported per run.

## CI / bootstrap and rule A

PREREGISTER_BOOTSTRAP.json unchanged (2,000 draws, rng seed 20261005, percentile 95 %, draws shared across seeds).
Resampling units: NIH test by patient; Shenzhen by image (= patient); kerm_heldout by patient. Rule A (Shenzhen,
Mahalanobis, λ = 2): pooled CI entirely below 0.5 and ≥ 3 of 5 per-seed CIs entirely below 0.5. Rules B / C / above
as in R3. Reported for every λ cell and for kerm_heldout too.

## Controls

λ = 0 (adversary weight 0) in every comparison. The dose test (below) has its own three-stream λ = 2 control.

## Conditional R1 dose-test repeat (runs only if Q1 inverts)

Trigger: INVERSION.json `inverted = true` (rule A on Shenzhen Mahalanobis at λ = 2 with 5 seeds). Otherwise not run
and reported as "not triggered".

Design (three streams per batch, λ = 2, seeds 42–82): 32 NIH classifier rows (natural mix, no adversary loss), 32 NIH
adversary rows resampled so that a fraction f of the NIH-vs-Kermany pneumonia gap is kept (f = 1: 2.3 %, f = 0.5:
37.6 %, f = 0: 73.0 % = Kermany), 32 kerm_adv rows. f ∈ {1, 0.5, 0} × 5 seeds = 15 runs; f = 1 is the three-stream
control. Prediction under the label-shift account: Shenzhen and kerm_heldout AUROC rise toward / above 0.5 as f → 0;
under §4.8 they do not depend on f.

## Reported cells

Per λ (mean ± SD over seeds): leakage, ID balanced accuracy, ID pneumonia AUROC, ID ECE, Shenzhen Mahalanobis / kNN /
cosine / MSP, Shenzhen-TB and Shenzhen-normal Mahalanobis, kerm_heldout Mahalanobis and its ID-mix reweighted value,
X-dom balanced accuracy, majority-nearest fraction; bootstrap rule per cell. Dose table per f. Class-mix table.
LaTeX rows in the Table 1 column order (leakage, ID bal acc, ID ECE, OOD ECE = kerm_heldout, OOD AUROC = Shenzhen
Mahalanobis, second OOD AUROC = kerm_heldout Mahalanobis, X-dom bal acc).

## Predicted outcome (written before any score exists)

- λ = 0: leakage ≥ 0.95 (adult vs pediatric CXR is trivially separable); ID balanced accuracy 0.60–0.72; Shenzhen
  Mahalanobis 0.55–0.68 (borderline for the gate); kerm_heldout ≥ 0.90.
- λ = 0.5, 2: leakage 0.60–0.80; kerm_heldout Mahalanobis below 0.5 (the adversary's own domain, as pad_heldout),
  with the reweighted-to-NIH-mix value higher than the all-class value.
- Shenzhen at λ = 2: we record both accounts above; our point prediction is **collapse toward 0.5 (rule C)**, because
  Shenzhen's class content (TB / normal) is not the label that is shifted between NIH and Kermany. If rule A is met
  instead, the label-shift account is supported and the dose test runs.

## Outputs

`results/paperB/r5/q1/`: `index/` (CSV per set, CLASS_MIX.json, SPLIT_AUDIT.json), `gate/<tag>/`, `sweep/<tag>/`,
`dose/<tag>/` (config.json, adversary_log.jsonl, summary.json, scores.npz, features/), `summary.csv`, `GATE.json`,
`INVERSION.json`, `BOOTSTRAP.json`, `REPORT.md` (LaTeX rows, one plain paragraph per result). Checkpoints under
`checkpoints/r5_q1/`. P8-recipe runs carry the `_recp8` suffix and report to `results/paperB/r5/q1/recipe_recp8/`.

## GPU-h estimate and basis

Basis: the lesion-level dual-encoder EfficientNet-B3 run measured ≈ 1.6 GPU-h for ≈ 20k steps at 64 images/step plus
evaluation. Q1 has the same step count, cheaper PNG decoding, a larger evaluation (≈ 74k images vs ≈ 45k), and
≈ 3 min of node-local staging: ≈ 1.8 GPU-h per run. Dose runs carry 96 images/step: ≈ 2.7 GPU-h.

| Stage | Runs | GPU-h |
|---|---:|---:|
| Gate | 3 | 5.4 |
| Sweep | 12 | 21.6 |
| Dose (conditional) | 15 | 40.5 |

Gate + sweep ≈ 27 GPU-h. The dose test (≈ 40 GPU-h) exceeds the +20 GPU-h approved for "Q1 inverts"; a 3-seed
version (9 runs, ≈ 24 GPU-h) is the fallback if the budget is held at +20 — the user chooses before launch.
CPU: index ≈ 0.05 h, each report ≈ 0.3 h.

## Stop rules and caps

Gate not met → stop, report candidates. Sweep cap 12 runs. Dose test only on INVERSION.json `inverted = true`.
`FORCE=1` overrides a launcher check and must be logged below as a deviation. A failed run is re-run once with the same
seed; a second failure is reported, not replaced by another seed.

## Pre-run amendment (user decision 2026-10-15, before any Q1 run)

The conditional dose test uses 3 seeds (42, 52, 62): f ∈ {1, 0.5, 0} × 3 = 9 runs, **≈ 24 GPU-h**. Still runs only if
INVERSION.json has `inverted = true`.

## Amendment 2026-10-17 (~22:00 +07) — natural / induced arms (before headroom gate runs under the new schedule)

After the headroom gate on Shenzhen (unchanged: λ = 0, seeds 42/52/62; mean ≥ 0.60, pooled CI lower ≥ 0.55, every
seed CI lower > 0.50), if the gate is **met**, run two pre-registered arms at **λ = 2** × seeds `(42, 52, 62, 72, 82)`,
same dual-encoder recipe as above. Mirror P10’s label-shift contrast; detection labels and CIs match
`decisions/paperB_r5_p10.md` (mutually exclusive **inversion / collapse / survives / indeterminate**; always report
\(\bar a\), \([\ell,u]\), \(\Delta=\bar a-\bar a_0\), \([\ell_\Delta,u_\Delta]\); B = **2,000** percentile bootstrap,
rng **20261017**, seed-mean only; λ = 0 cell mean from the gate runs as \(\bar a_0\)).

**Class mix (normal / pneumonia) — locked from the index table above:**

| Domain | Role | n | Normal | Pneumonia | P(pneumonia) |
|--------|------|--:|------:|----------:|-------------:|
| nih_train | labelled / adv domain 0 | 39,546 | 38,630 | 916 | **0.023** |
| kerm_adv | adversary domain 1 | 4,098 | 1,108 | 2,990 | **0.730** |
| nih_test | ID eval | 12,358 | 12,072 | 286 | 0.023 |
| kerm_heldout | secondary OOD | 1,758 | 475 | 1,283 | 0.730 |

Natural adversary-domain TV (Bernoulli pneumonia): \(\mathrm{TV}=|0.023-0.730|=\mathbf{0.707}\).

| Arm | Adversary class mix | TV | Prediction (Shenzhen Maha) if headroom gate met |
|-----|---------------------|---:|--------------------------------------------------|
| **Natural** | Existing NIH / Kermany mixes (table) | 0.707 | Same two accounts as the original Q1 prediction (point: collapse toward 0.5); report with locked labels |
| **Induced** | Resample **adversary-stream** rows only: domain-0 (NIH) batches **20%** pneumonia, domain-1 (Kermany) **80%** pneumonia; classifier stream keeps natural NIH mix | **0.60** | **Inversion** if labels apply |

**Confounder note (locked):** natural NIH vs Kermany also differs in **age, scanner, and anatomy**. Only the
**induced-vs-natural** contrast (same modality pair, controlled class-mix gap) speaks to label shift. Do not attribute
a natural-arm inversion/collapse solely to label shift.

Headroom gate still runs first; if it fails, neither arm is launched (replacement OOD candidates unchanged).

## Deviations

- 2026-10-17 ~22:00 (+07), **order only**. Q1 headroom gate re-queued to run **after R2-prior**, in parallel with the
  Camelyon adversary ladder pilot (one GPU each). Old dependency chain left pending is superseded for the gate stage.
