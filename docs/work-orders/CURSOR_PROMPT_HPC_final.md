# FINAL WORK ORDER — Paper B: mechanism, cost, and the writing package

All experimental phases are complete. This is the last round before drafting. Three items of new work, a set of explicit drops, and the figure/table package.

**Headline claim the whole package must support:**

> Domain-adversarial invariance training collapses covariate-shift OOD detection from 0.86 to ≤0.51 at the smallest nonzero adversarial weight tested, and inverts it below chance thereafter — while domain leakage improves and in-distribution balanced accuracy does not drop. Every metric a practitioner monitors improves while the OOD safety gate silently fails.

It is a **cliff, not a trade-off curve.** There is no safe operating point. Do not describe it as a tunable trade-off anywhere.

---

## DROP — do not compute, do not revisit

**SPS (Semantic Purity Score) is withdrawn.** It presumed a continuous leakage↔AUROC relationship. Phase 3 refutes that: between λ=0.25 and λ=2 leakage *rises* (0.553 → 0.584) while AUROC *falls* (0.508 → 0.413). The apparent linear relation exists only because λ=0 anchors it. No extrapolation is defensible. Remove SPS from all plans and reports.

**The n=4 preview curve moves to the appendix**, captioned explicitly as a correlation across heterogeneous models, superseded by the λ sweep. Do not present it as evidence.

---

## Item 1 — Phase 6: what the invariance costs (mostly already on disk)

The recap states Phase 3 already records **PAD acc per λ**. Surface it rather than re-running, after verifying three things:

1. **Which PAD subset?** It must be `pad_heldout` only (716 images / 412 patients). If the recorded number used `pad_full` or `pad_adv`, those images were in `L_ctx`/`L_adv` during training and the number is contaminated — recompute on `pad_heldout`.
2. **Which classes?** PAD covers 6 of 8. Report the 6-class-restricted evaluation and state the restriction.
3. **Which metric?** Accuracy alone is not enough under PAD's class imbalance.

Extend to a full column set, per λ ∈ {0, 0.25, 0.5, 1, 2, 4, 8}, all seeds, and for the ResNet-50 baseline and EffB3 control as reference rows:

- accuracy, **balanced accuracy**, macro AUC
- per-class recall
- **melanoma sensitivity at fixed specificity 0.85**

Note explicitly in the report that PAD labels never entered `L_cls` (`ignore_index=-1`), so this is zero-shot cross-domain transfer.

**Why this matters more than anything else left.** The obvious reviewer question is: *you lose covariate-shift OOD detection — do you at least gain cross-domain accuracy?*

- If cross-domain balanced accuracy is flat or falls across λ: the finding becomes **you pay a safety cost and get nothing for it**, which is the strongest version of the paper.
- If it rises with λ: there is a genuine trade-off, and the paper must present it as such rather than as pure loss.

Report whichever the data says. This column goes **into Table 1 beside leakage, ID bal acc and OOD AUROC** — the four-column table is the paper.

## Item 2 — Phase 3.5: the mechanism (inference, data already on disk)

Three mechanisms are refuted (image memorisation, class composition, domain specificity) and the inversion is unexplained. One hypothesis remains consistent with every negative result:

> Invariance training does not merely remove domain information — it **compresses unfamiliar inputs toward the high-density regions of the training manifold.** The cheapest way for the encoder to satisfy the adversary is to map anything unfamiliar near the class centroids, which is exactly the low-Mahalanobis region. In-domain test samples retain their natural spread. The model is therefore most typical-looking on precisely the inputs it should be least certain about.

**Measure it.** For λ ∈ {0, 0.25, 2, 8}, all seeds, on `z_lesion`:

1. Distance-to-nearest-class-centroid distributions for: ISIC train, ISIC test, `pad_heldout`, Fitzpatrick17k. Report median, IQR, and full density plots.
2. **Ratio of median distance, OOD ÷ ISIC test**, as a function of λ. Prediction: > 1 at λ=0, < 1 by λ=0.25, falling further as λ rises. This single curve is the mechanism figure.
3. **Spread ratio**, IQR(OOD) ÷ IQR(ISIC test), same λ values. Prediction: OOD distributions tighten as λ rises.
4. Local density check: mean kNN distance (k=50) within the ISIC-train feature bank, for each group, per λ.
5. Feature norm statistics per group, per λ, as a control for trivial scaling.

If the predictions hold, the paper has an explanation rather than a list of refutations. **If they do not hold, report that and leave the inversion explicitly unexplained** — that remains an honest and publishable position given three refuted alternatives. Do not construct a post-hoc story to fill the gap.

## Item 3 — Phase 9 (optional, cheap, do only if capacity is free)

PAD metadata has `fitspatrick`. Stratify the Item 1 cross-domain metrics by Fitzpatrick group: per-group balanced accuracy, melanoma sensitivity, group sizes, CIs. Exclude groups with n < 30 from any significance claim while still reporting their n.

Valuable for a medical venue, not load-bearing. Skip if it delays the writing package.

---

## Item 4 — Phase 10: statistics, tables, figures

### Statistics
- Bootstrap 95% CIs (2000 resamples) over test samples for every headline number
- **The cliff specifically:** test λ=0 (n=5) vs λ=0.25 (n=3) on OOD AUROC, with effect size and CI. This comparison carries the paper — give it its own line in the results
- Paired seed-wise comparisons where seeds align; Cohen's d; Holm-Bonferroni across an explicitly defined family
- Lead with effect sizes and CIs, not p-values, given n=3–5
- Keep the locked convention: **`AUROC > 0.9999 (2 discordant pairs / 11.6M)`**, never `1.0000 ± 0.0000`

### Tables
- **Table 1 (the paper).** Per λ_adv: leakage (balanced acc), ID balanced acc, ECE, OOD AUROC on PAD, OOD AUROC on Fitzpatrick, 6-class-restricted OOD AUROC, **cross-domain PAD balanced acc**, `z_context` OOD AUROC. Baseline and EffB3 control as reference rows
- **Table 2 — "a monitor is cheap."** Every representation × (linear domain-head AUROC, PCA k=1 AUROC, variance explained at k=1). Lead with the **k=1 column**; the AUROC column shows near-parity and must not be presented as a `z_context` advantage
- **Table 3 — refuted mechanisms.** Each prediction, its test, its result, and what it rules out. This table is a strength; give it main-text space, not an appendix

### Figures (PNG + PDF, 300 dpi, colourblind-safe, `results/paperB/figures/`)
1. **The dial.** Three panels sharing the λ_adv axis: leakage, ID balanced acc, OOD AUROC. The cliff between 0 and 0.25 must be visually unmistakable. Annotate the region where the first two improve while the third collapses — this is the paper in one image
2. **Mechanism.** Distance-ratio curve (OOD ÷ ID) vs λ, plus distance distributions at λ=0 and λ=2
3. **The monitor is retained.** `z_lesion` vs `z_context` OOD AUROC across the full dial: one collapses, one stays flat at ~1.0
4. **It generalises.** PAD and Fitzpatrick17k collapse together across λ, with the baseline's 0.994 on Fitzpatrick as a reference line
5. *(supp)* Semantic OOD: 4a and 4b, both branches, equal prominence, reported as a null result
6. *(supp)* Domain-axis distributions for ISIC / Fitzpatrick17k / PAD, captioned only as **"Fitzpatrick17k does not fall on the ISIC side"** — overlap with PAD is 0.47 and no tighter claim is supported
7. *(supp)* The n=4 preview curve, captioned as superseded

---

## Not running — record these as limitations with justification

- **Phase 7 (patient-level ISIC splits).** Not run. Note in the limitations that ID balanced accuracy is compared *across λ under an identical split*, so any split-induced inflation applies equally to every row and does not threaten the claim. The absolute ID numbers may be optimistic; the relative pattern is unaffected.
- **Phase 8 (background-only).** Not run. The paper no longer makes a shortcut-attribution claim, so this is no longer load-bearing.
- **Phase 5 (full per-class conditional probe).** Partially covered: the label-only domain probe (0.7968 vs majority 0.688) from Phase 0 is the key control and is reported; leakage is already reported as balanced accuracy. Note the per-class conditional probe as future work.
- **Non-medical replication.** Not run. State plainly that generality beyond dermatology is untested.
- **Raw-source trivial-detector repeat.** Blocked by PAD PNG permissions; record as a known gap.

---

## Final deliverable

Update `results/paperB/MASTER_REPORT.md` to contain:
- The four-column dial table (Table 1) as the centrepiece
- The cliff comparison with its CI
- The mechanism verdict — supported, or explicitly unexplained
- The cross-domain cost answer from Item 1
- Table 2 and Table 3
- Every limitation above, with its justification
- A list of claims in the original `manuscript_main.md` that this work contradicts, with corrected numbers — that manuscript was written for a different paper and will be rewritten, not patched

**Standing instruction, unchanged and final:** report what the numbers say. Four predictions have already failed and were reported as findings; that is why this result is credible. Do not retune anything, and do not soften a null result into a suggestive one.
