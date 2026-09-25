# WORK ORDER — CSG-Skin "Paper B": Factorized OOD Monitoring

**Repo root on HPC:** `CSG-Skin/` (contains `src/`, `scripts/`, `checkpoints/`, `results/`, `data/`, `configs/`, `cbm_revision/`).
**Agent:** you are running in Cursor on an HPC login/compute node. Long jobs go through the scheduler (SLURM assumed — detect and confirm).
**All new outputs go under `results/paperB/`. Do NOT modify or overwrite anything already in `results/`, `results/cbm_revision/`, or `results/effb3_control/`.**

---

## 0. Context you need (read this, don't re-derive it)

The existing manuscript claims: a ResNet-50 baseline leaks acquisition domain at probe accuracy 0.9791, CSG-Lite reduces it to ~0.71, but CSG-Lite's Mahalanobis OOD AUROC collapses to ~0.41 — which the manuscript apologises for and attributes to `z_lesion` being only 16-dimensional.

**That explanation is almost certainly wrong, and the "limitation" is actually the paper's main result.** Evidence already in the repo:

| Source file | Fact |
|---|---|
| `results/effb3_control/significance_effb3.csv` | EffNet-B3 single-encoder control already leaks only 0.8017 vs ResNet-50's 0.9792 → most of the headline "leakage reduction" is a backbone confound, not CSG |
| `results/csg_lite/runB_orth1_s42/leakage.json` | `z_context_acc_mean = 0.9997` — the context branch separates domains **better than the baseline's entire 2048-d representation (0.9795)** |
| `results/cbm_revision/ood_comparison.csv` | MSP, Energy, Cosine and Mahalanobis **all** collapse to 0.41–0.43 for CSG, together. Four independent detectors failing identically is not a covariance-estimation artifact — it is the training objective working as specified |
| `scripts/eval_ood_benchmarks.py` (`_collect_csg_features`) | `z_context` is computed and then **thrown away**: `out, _dctx, _dadv, z_l, _zctx = model(...)`. No OOD score has ever been computed on it |

### The thesis this work order tests

1. **Proposition.** If the OOD set differs from the ID set only by domain (covariate shift), then any detector operating on a domain-invariant representation must approach AUROC = 0.5 *by construction*. Domain-invariance training and covariate-shift OOD detection are formally opposed objectives.
2. **Mechanism.** CSG's AUROC sits *below* 0.5 (0.41) because PAD-UFES is used inside the adversarial objective: the encoder is explicitly trained to push PAD inside ISIC's manifold, making PAD *more* typical than held-out ISIC test data.
3. **Resolution.** The trade-off only binds for single-encoder debiasing. A factorized model gets both: `z_lesion` (invariant) for diagnosis, `z_context` (deliberately leaky) as a covariate-shift monitor — two clinically distinct alarms instead of one uninterpretable score.

### Falsifiable predictions — report PASS/FAIL on each

| # | Prediction | Phase |
|---|---|---|
| P1 | Mahalanobis AUROC on `z_context` > 0.95, vs 0.41 on `z_lesion`, same checkpoints | 1 |
| P2 | With PAD held out of the adversarial objective, `z_lesion` AUROC rises to ≈ 0.50 (not 0.41); `z_context` stays > 0.95 | 2 |
| P3 | `z_lesion` OOD AUROC decreases monotonically as λ_adv increases, tracking leakage; `z_context` AUROC stays flat and high | 3 |
| P4 | **Double dissociation.** On semantic OOD with domain held constant, `z_lesion` ≥ baseline while `z_context` ≈ 0.50. Each branch wins exactly one OOD type | 4 |
| P5 | `z_context` flags an unseen third domain it was never trained on (kills the "you supervised it" objection) | 11 |
| P6 | On background-only images, baseline stays well above chance while CSG `z_lesion` falls to chance | 8 |

A FAIL is a valid and useful result. **Report it plainly; do not tune anything to rescue a prediction.**

---

## Phase 0 — Verification gates (blocking, no compute)

Do all of these before any training. Write findings to `results/paperB/PHASE0_CHECKS.md` and **stop and report if any gate fails.**

**0.1 — Inventory.** Confirm which checkpoints exist for: `baseline_soft`, `effb3_control`, `runA_grl`, `runB_orth1`, `runB` (orth=5), for seeds 42/52/62/72/82. Produce a table of method × seed × checkpoint path × exists. Note whether each is `best-*.ckpt` or `last*.ckpt` — the repo mixes both (`leakage.json` for `runB_orth1_s42` points at `last-v1.ckpt` while the results tables cite `best-36.ckpt`). **Pick one policy (best-by-`val/acc`), apply it everywhere, and state it.**

**0.2 — CRITICAL: train/eval transform parity.** In `src/models/csg_lite.py` and `scripts/train_csg.py`, determine what each branch receives during training. `METHODS_SUMMARY.md` documents a `build_lesion_branch_transform_gray` (Resize → CenterCrop → **Grayscale(3)** → Normalize) for the lesion branch. But `scripts/eval_ood_benchmarks.py` evaluates with `model(images, x_lesion=None, ...)`.

Determine exactly what `x_lesion=None` does in `CSGLite.forward`. If it routes the *same colour tensor* to both branches while training fed the lesion branch grayscale, **every published OOD and leakage number for CSG is computed under a train/eval mismatch.** Report which it is. If mismatched:
- Fix eval to mirror training exactly.
- Recompute the existing leakage + OOD tables under the fix, into `results/paperB/phase0_reeval/`.
- Report the delta against `results/cbm_revision/`. This may change the entire paper — flag it loudly, do not silently proceed.

**0.3 — Domain/label entanglement.** PAD-UFES has no DF and no VASC. Quantify how much of the domain probe's accuracy is obtainable from the *label* alone: fit a logistic probe on one-hot ground-truth labels → domain. Report its accuracy against the 0.688 majority baseline. This upper-bounds how much "leakage" is really label-domain correlation.

**0.4 — Split hygiene.** Confirm the ISIC train/val/test split and the PAD usage in `src/datasets/splits.py`. Specifically: are the PAD images used in the adversarial/context branch during training **the same images** later used as the OOD set at evaluation? Answer yes/no explicitly with the code path. (Expected: yes — this is the mechanism behind P2.)

**0.5 — Environment record.** Capture GPU model, CUDA, torch/lightning versions, node spec, and per-epoch wall time into `results/paperB/ENVIRONMENT.md`. This closes the manuscript's Limitation #6.

---

## Phase 1 — Headline experiment: dual-branch OOD (no retraining, ~1 GPU-hour)

Create `scripts/eval_ood_dual_branch.py`. Do **not** edit `eval_ood_benchmarks.py` in place; copy the parts you need so existing results stay reproducible.

**Feature spaces to score**
- Baseline / EffB3 control: `backbone_raw`, `logits`
- CSG variants: `z_lesion`, `z_context`, `concat(z_lesion, z_context)`, `backbone_raw_lesion`, `backbone_raw_context`, `logits`

**Detectors**
- MSP, Energy (T=1), ODIN (T ∈ {1,10,100}, ε ∈ {0, 0.0014}), max-cosine-to-class-mean, Mahalanobis (class-conditional, shared covariance), Mahalanobis (single Gaussian, class-agnostic), kNN distance (k=50, normalised features)
- **`context_predictor` softmax** as a direct score on CSG models — free, zero extra parameters

**Statistics fitting rule — enforce and assert in code:** every mean, covariance, class centroid and kNN bank is fit on the **ISIC training split only**. No PAD, no ISIC val, no ISIC test may touch the fit. Add a runtime assertion, not just a comment.

**Grid:** 4 methods × 5 seeds × feature spaces × detectors. Metrics: AUROC, AUPR-in, AUPR-out, FPR@95.

**Outputs**
- `results/paperB/ood_dual_branch_per_seed.csv` — one row per (method, seed, feature_space, detector, metric)
- `results/paperB/ood_dual_branch_aggregate.csv` — mean ± std over seeds
- `results/paperB/PHASE1_REPORT.md` — verdict on P1, plus this table filled in:

| Branch | Domain probe acc | OOD AUROC (Maha) | ID Bal Acc |
|---|---|---|---|
| Baseline (entangled) | 0.979 | 0.885 | 0.655 |
| `z_lesion` (invariant) | 0.724 | ? | 0.702 |
| `z_context` (leaky) | 0.9997 | **?** | n/a |

**Honesty requirement, non-negotiable.** `z_context` is trained with domain supervision; the baseline's Mahalanobis score is not. Every table and figure must label `z_context` detectors as **domain-supervised**, and the report must state this asymmetry explicitly in its own sentence. The defence against "of course it works, you supervised it" is Phase 4 and Phase 11, not spin.

**Gate: if P1 fails (`z_context` AUROC < 0.9), stop and report before starting Phase 2.**

---

## Phase 2 — Mechanism test: hold PAD out of the adversary (retraining, 5 runs)

**Split.** Partition PAD-UFES **at patient level** using `patient_id` from `pad_ufes20/metadata.csv` (never split a patient or a `lesion_id` across partitions): `pad_adv` 70% (visible to the context/adversarial branches during training) and `pad_heldout` 30% (never seen in any form).

Retrain `runB_orth1` (λ_adv=2.0, λ_orth=1.0) with only `pad_adv` in the training branches, 5 seeds (42/52/62/72/82), all other hyperparameters identical.

Evaluate OOD three ways: OOD = `pad_adv` / OOD = `pad_heldout` / OOD = full PAD.

**Prediction P2:** `z_lesion` AUROC on `pad_heldout` ≈ 0.50 while on `pad_adv` it stays ≈ 0.41. That gap *is* the proof that sub-chance AUROC comes from the training objective, not from a 16-d covariance problem.

Outputs → `results/paperB/phase2_pad_holdout/` + `PHASE2_REPORT.md`.

---

## Phase 3 — The dial: λ_adv sweep (largest compute block)

λ_adv ∈ {0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0}, λ_orth = 1.0 fixed, 3 seeds each (21 runs); extend endpoints λ_adv ∈ {0, 2.0, 8.0} to 5 seeds. Use the Phase 2 PAD split so results compose. **Submit as a SLURM array job**, one run per task, with checkpointing and resume — do not run this in a foreground shell.

Record per run: leakage on `z_lesion` and `z_context` (balanced acc + AUROC, see Phase 5), ID acc / bal acc / ECE, OOD AUROC for every detector on both branches, cross-domain PAD accuracy (Phase 6 metric).

**Analysis → `results/paperB/phase3_sweep/`**
- Master curve: x = leakage(`z_lesion`), y = OOD AUROC, one line per detector, with `z_context` overlaid as a flat reference
- Fit `dAUROC/dLeakage` with a bootstrap CI
- **Semantic Purity Score (SPS)** = AUROC extrapolated to leakage = majority baseline (0.688). SPS ≈ 0.5 means the benchmark measured domain, not pathology. Report SPS per detector with CI
- Utility cost curve: bal acc vs leakage, to show what invariance actually costs

---

## Phase 4 — Double dissociation: semantic OOD with domain held constant

**This is the experiment that makes the paper.** Everything so far varies domain. Here domain is fixed and only semantics change.

Train on ISIC only, holding out classes as unseen. Two configurations:
- **4a:** hold out {DF, VASC} (rare) → train on 6 classes
- **4b:** hold out {SCC} (clinically important, n=628) → train on 7 classes

OOD set = held-out-class images **from ISIC only**. ID = ISIC test of the kept classes. PAD is not involved at all. Run baseline, effb3_control and runB_orth1, 3 seeds each.

**Prediction P4:** `z_lesion` ≥ baseline on semantic OOD, while `z_context` ≈ 0.50 (context carries no pathology information). Combined with Phase 1, each branch wins exactly one OOD type — a clean 2×2 double dissociation.

Produce the 2×2 figure (branch × OOD type) — this is the paper's Figure 1. Outputs → `results/paperB/phase4_semantic_ood/`.

---

## Phase 5 — Conditional leakage probe (rigor, cheap)

Replace the current probe protocol:
1. Report **balanced accuracy and AUROC**, not raw accuracy against a 0.688 majority baseline — raw accuracy is a poor metric under that imbalance.
2. **Per-class conditional probe:** fit domain probes separately within each class present in both domains (MEL, NV, BCC, AK, BKL, SCC), then report the class-weighted mean. This removes the label-domain confound quantified in 0.3.
3. Keep the shuffle control; also add a label-only probe baseline from 0.3 as a reference line.
4. Probe over 5 seeds with bootstrap CIs.

Apply to every model and every checkpoint in Phases 1–4. Outputs → `results/paperB/phase5_conditional_probe/`.

---

## Phase 6 — Cross-domain diagnosis: the "so what"

The current manuscript never evaluates diagnostic performance on PAD at all — leakage is reduced but no generalisation benefit is ever demonstrated. Fix this.

Evaluate 8-class → 6-class-restricted diagnosis on `pad_heldout` (Phase 2 split, never seen), for baseline / effb3_control / runA_grl / runB_orth1 across 5 seeds. Report accuracy, balanced accuracy, per-class recall, **melanoma sensitivity at fixed specificity 0.85**, and macro AUC. Note explicitly that PAD labels were never used for `L_cls` (`ignore_index=-1`), so this is zero-shot cross-domain transfer.

Outputs → `results/paperB/phase6_crossdomain/`.

---

## Phase 7 — Patient/lesion-level splits (closes Limitation #1)

ISIC 2019 metadata has a `lesion_id` column; PAD has `patient_id` and `lesion_id`. Build grouped splits (`GroupShuffleSplit` on `lesion_id`, falling back to `image` id where `lesion_id` is null — **report the null fraction**). Re-run the main table under grouped splits for baseline / effb3_control / runB_orth1, 3 seeds, and report the delta vs image-level splits.

Outputs → `results/paperB/phase7_grouped_splits/`.

---

## Phase 8 — Background-only shortcut test (replaces the missing GradCAM)

Reuse the Otsu pipeline in `scripts/build_lesion_only_metadata.py`, but **invert** it: zero out (or inpaint with the border median) the lesion bounding box and keep only the surrounding context. Build a background-only version of the ISIC test set.

Evaluate every model's diagnostic accuracy on background-only images. **Prediction P6:** baseline stays well above the 8-class chance / majority floor (evidence of genuine shortcut reliance), CSG `z_lesion` falls toward it. Report against both the uniform-chance and the majority-class floors.

This is direct causal evidence and answers manuscript Reviewer Concern 6 far better than GradCAM would. Also export GradCAM panels for 12 qualitative examples while you are here.

Outputs → `results/paperB/phase8_background_only/`.

---

## Phase 9 — Fairness subgroup analysis

PAD metadata has a `fitspatrick` column (note the dataset's own spelling). Stratify all Phase 6 cross-domain metrics by Fitzpatrick group; report per-group balanced accuracy, melanoma sensitivity and leakage, with group sizes and CIs. Drop groups with n < 30 from significance claims but still report their n.

Outputs → `results/paperB/phase9_fairness/`.

---

## Phase 10 — Statistics, figures, tables

**Statistics.** 5 seeds wherever feasible. Bootstrap CIs (2000 resamples) over *test samples* for every headline metric — this gives real CIs independent of the small seed count. Paired seed-wise comparisons where seeds align; Wilcoxon signed-rank plus Cohen's d; **Holm-Bonferroni correction across the full comparison family**, with the family defined explicitly. With n=5, lead with effect sizes and CIs, not p-values.

**Mandatory correction to carry forward:** every comparison must use **EffNet-B3 single-encoder as the primary control**, not ResNet-50. The honest headline is leakage 0.80 → 0.72 and bal acc +1.9pp, *not* 0.98 → 0.71 and +3.9pp. Any table that omits the EffB3 control row is wrong.

**Figures** (PNG + PDF, 300 dpi, colourblind-safe, into `results/paperB/figures/`):
1. Double dissociation 2×2 — branch × OOD type (the headline)
2. Master curve — leakage vs OOD AUROC across λ_adv, per detector, with SPS annotated
3. Branch comparison — probe accuracy vs OOD AUROC vs bal acc, three models
4. Utility cost — bal acc vs leakage along the dial
5. Cross-domain PAD per-class recall
6. Background-only accuracy collapse
7. Fairness by Fitzpatrick
8. t-SNE of `z_lesion` and `z_context` coloured by domain and by class (4 panels)

**Tables** as CSV + LaTeX booktabs into `results/paperB/tables/`.

---

## Phase 11 — Optional tier-up: unseen third domain

Add a third dermatology dataset never seen in training (DDI, Derm7pt or Fitzpatrick17k — check licence and record it). Evaluate `z_context`'s OOD detection on it **zero-shot**.

**Prediction P5:** `z_context` flags the unseen domain despite never having seen its label. This is the decisive answer to "you supervised it to detect domain" — it shows the branch learned a general acquisition-context representation, not a two-way ISIC/PAD classifier. It also breaks the two-domain modality confound (dermoscopy vs smartphone) that a reviewer will otherwise raise, since the manuscript's own Reviewer Concern 8 currently has no strong answer.

Do this only after Phases 1–4 report. Outputs → `results/paperB/phase11_third_domain/`.

---

## Execution rules

1. **Phases 0 and 1 are gates.** Stop and report after each. Do not start Phase 2 until Phase 1's verdict on P1 is in.
2. **Never overwrite existing results.** Everything new goes under `results/paperB/`.
3. **No retraining in Phase 1.** It runs entirely on existing checkpoints.
4. Long jobs go to SLURM (confirm the scheduler first) as array jobs with resume; never in a foreground shell.
5. Set all seeds, use `pl.seed_everything(seed, workers=True)`, log per-run config as JSON next to its outputs.
6. Every script takes `--dry_run` that validates paths, shapes and splits on a handful of batches before the full run.
7. Emit a per-phase `PHASE{n}_REPORT.md` with: what ran, exact numbers, PASS/FAIL against the relevant prediction, and anything surprising.
8. **Report failures as findings.** If `z_context` AUROC comes back at 0.6, or the double dissociation does not appear, say so with the numbers. Do not tune hyperparameters to make a prediction pass — that would invalidate the whole design.
9. If Phase 0.2 reveals the train/eval transform mismatch, treat it as a blocking finding and report before doing anything else.

## Final deliverable

`results/paperB/MASTER_REPORT.md` containing:
- The PASS/FAIL table for P1–P6
- Final main results table with the EffB3 control as primary baseline
- The dual-branch OOD table
- The double-dissociation 2×2
- SPS per detector with CIs
- Cross-domain PAD diagnosis results
- Every deviation from this work order, and why
- An explicit list of claims in `manuscript_main.md` that these results contradict, with the corrected numbers

## Compute budget guide

| Phase | Runs | Rough cost |
|---|---|---|
| 0 | 0 | minutes |
| 1 | 0 (inference only) | ~1 GPU-hour |
| 2 | 5 | ~5 × 40 epochs |
| 3 | 21–27 | largest block — array job |
| 4 | 18 (2 configs × 3 models × 3 seeds) | medium |
| 5 | 0 (probes only) | minutes |
| 6 | 0 (inference only) | ~1 GPU-hour |
| 7 | 9 | medium |
| 8 | 0 (inference) + preprocessing | ~2 hours |
| 9 | 0 | minutes |
| 11 | 0 (inference) + data acquisition | small |

**Start with Phase 0, then Phase 1, and report back before going further.**
