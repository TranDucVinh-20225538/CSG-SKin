# ADDENDUM — Phase 4.5: what Phase 4 actually found

**Phase 4 result:** P4 not clean. `z_context` did not reach 0.50 on semantic OOD (4a 0.64, 4b 0.61). `z_lesion` lost to baseline on 4a (0.65 vs 0.69) and beat it on 4b (0.81 vs 0.68).

**The clean 2×2 double dissociation is dead. Do not attempt to rescue it.** But the ordering does flip between OOD types (domain: `z_context` 1.00 ≫ `z_lesion` 0.41; 4b: `z_lesion` 0.81 > `z_context` 0.61), and 4b may contain a better result than the dissociation would have been. Phase 4.5 establishes whether it is real.

---

## 4.5a — Variance first (blocking, free)

**No conclusion may be drawn from Phase 4 until this is reported.** For every Phase 4 number (4a, 4b, all three models, all seeds): mean, std, per-seed values, and bootstrap 95% CI over test samples.

0.65 vs 0.69 at n=3 may be noise. 0.81 vs 0.68 needs a spread before it can carry a claim. Report paired seed-wise differences and Cohen's d. If the 4b gap's CI includes zero, say so and stop there.

## 4.5b — Decompose 4a by class (free, explanatory)

4a held out DF (n=239) and VASC (n=253) together. **VASC lesions have a strong low-level colour signature** (red/purple vascular lesions); DF (brownish) does not. Hypothesis: `z_context`'s 0.64 on 4a is driven almost entirely by VASC detected on colour, not by any pathology sensitivity.

Split the 4a OOD set and score **DF-only** and **VASC-only** separately, for all three models, all seeds, all detectors.

**Prediction:** `z_context` scores high on VASC and ≈ 0.50 on DF.

If confirmed, this is a finding in its own right and must be written up as one: **the semantic/covariate dichotomy is not clean in real data — some diagnostic classes carry low-level appearance signatures, so a context encoder detects them without any pathology representation.** That explains why `z_context` did not reach 0.50 and is more useful than the failed prediction would have been.

Also decompose by class for `z_lesion` and the baseline, to see whether `z_lesion`'s 4a loss is concentrated in one class.

## 4.5c — Near/far characterisation (GPU, queue behind Phase 3)

Two configurations, one winning and one losing, with the winner carrying the nicer story, is the textbook cherry-pick setup. **4b cannot be reported alone.** A systematic picture is required.

**Design (cheap):** hold out **{DF, VASC, SCC} together**, train once, then score OOD **separately for each held-out class**. Three semantic-OOD conditions per training run: 3 models × 3 seeds = **9 runs**, versus 54 for full leave-one-class-out.

For each held-out class report, per model and detector: AUROC, FPR@95, and a **near/far index** — e.g. mean cosine similarity between the held-out class centroid and the nearest retained class centroid in the baseline's feature space, computed on training data only.

**Hypothesis to test:** `z_lesion`'s advantage over the baseline correlates with the near/far index — it wins on near-OOD (SCC, confusable with AK/BCC) and loses on far-OOD (DF, VASC, visually distinctive). Plot AUROC advantage against the near/far index across all held-out classes, pooling with the existing 4a/4b runs.

If the correlation holds, the claim becomes:

> The invariant lesion representation is not uniformly better at novelty detection. It is better at **hard near-OOD, which is the clinically dangerous regime**, and worse at easy far-OOD that low-level features already solve.

If it does not hold, report that 4a and 4b simply differ and no systematic pattern was found. **That is an acceptable outcome; an unsupported near/far story is not.**

**Priority:** strictly behind Phase 3. Do not let this compete for GPUs with the λ_adv sweep.

---

## Framing instruction

The manuscript may **not** claim a double dissociation. The defensible framing after Phase 4 is graded and class-dependent:

- Domain OOD: `z_context` 1.00 vs `z_lesion` 0.41 — a large, robust separation between branches
- Semantic OOD: both branches land in 0.6–0.8, ordering depends on which class is held out
- Therefore: the branches measure different things, but the split is graded rather than categorical, because real diagnostic classes are not purely semantic — some carry low-level appearance signatures

Report 4a and 4b with equal prominence in every table and figure. If 4.5c is not completed before submission, **4b must not be used to support a near-OOD claim** — report both configurations descriptively and leave the interpretation open.

**Outputs:** `results/paperB/phase4_5/` + `PHASE4_5_REPORT.md`.
