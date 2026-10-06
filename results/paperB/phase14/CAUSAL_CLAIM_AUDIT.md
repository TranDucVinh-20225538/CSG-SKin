# Phase 14 Item D — causal-language audit

The paper must not claim that reducing leakage causes the OOD collapse. Two independent results contradict mediation by linear domain-decodability:

- Phase 3: λ=0.25 → 2, leakage **rises** 0.553 → 0.584 while AUROC **falls** 0.508 → 0.413.
- Phase 13.2: backbone leakage is flat 0.982 → 0.941 while backbone Mahalanobis falls 0.749 → 0.475.

**Corrected claim.** Adversarial training causes both the leakage drop and the OOD collapse. The OOD collapse is not mediated by linear domain-decodability. It follows from geometric changes a linear domain probe does not capture.

Leakage stays in Table 1 as a measured quantity. The dial is plotted against λ, never against leakage. The n=4 preview curve is doubly superseded (heterogeneous-model correlation, and it is an AUROC-vs-leakage plot).

## Instances

| Location | Language | Verdict | Action |
|---|---|---|---|
| `MASTER_REPORT.md` L5 headline | Collapse “while domain leakage improves”; “every metric a practitioner monitors improves” | Co-occurrence, not an explicit “leakage causes OOD”, but leakage is the implied success variable and ECE is not the lead | Rewritten. ECE 0.247→0.746 is the lead. Leakage is a co-effect of λ, not the mediator |
| `MASTER_REPORT.md` L9 SPS paragraph | Opposite directions 0.553→0.584 vs 0.508→0.413; preview appendix-only | Already correct on SPS. Does not yet state the mediation rejection or 13.2 | Kept. New “Corrected causal claim” section added above the dial |
| `MASTER_REPORT.md` L140 `fig_dial` caption | Three panels vs λ_adv (leakage, ID, OOD) | Axis is already λ. Caption treated leakage as a peer outcome of the cliff | Caption now: x-axis is λ; leakage is measured, not explanatory. ECE noted as lead |
| `MASTER_REPORT.md` L146 `fig_preview_superseded_supp` | “Captioned as superseded. Not evidence.” | Necessary but incomplete: the figure is also an AUROC-vs-leakage plot | Caption now: doubly superseded (heterogeneous n=4, and leakage-on-x asserts mediation) |
| `MASTER_REPORT.md` L175 original-claim table | Leakage reduction “coincides with the OOD cliff” | “Coincides” is weaker than “causes”, but still invites mediation | Corrected: joint effects of λ; 13.2 shows they can decouple |
| `MASTER_REPORT.md` L177 | Leakage vs OOD is continuous (SPS withdrawn) | Already a refutation of the preview curve, not a mediation claim | Kept. Preview marked doubly superseded |
| `MASTER_REPORT.md` L190–192 Discussion | **“The OOD gate degrades as a function of achieved invariance”** | **Direct mediation claim. Contradicted by Phase 3 and 13.2** | Deleted. Replaced with the corrected claim |
| `phase12/PHASE12_1B_AUDIT.md` L31 | Same “as a function of achieved invariance” sentence | Same mediation claim | Corrected |
| `scripts/aggregate_final_package.py` L774, L943, L958–960 | Generator of the three MASTER passages above | Would re-introduce the claim on the next writing-package run | Generator updated to match the corrected MASTER |
| `phase10/table3.md` / Table 3 in MASTER | SPS row and 16-d row | SPS row is a continuity refutation, not a mediation claim. Missing the 13.2 decoupling row | Added a mediation-rejected row. Leakage remains a measured column in Table 1 |
| `phase13/PHASE13_REPORT.md` §13.2 | “Outcome (mixed)” | Accurate as a depth classification, but the backbone leak-vs-Maha split is a finding, not a detail | Promoted: linear domain-decodability and OOD collapse decouple at the backbone |
| `phase13/PHASE13_REPORT.md` §13.3 | “The more invariant the model is made, the more confident…” | Ties confidence to “invariance” (the leakage probe) | Rewritten: λ raises OOD confidence/ECE; ID ECE stays flat. This is the paper’s lead number |
| `analyze_phase13.py` fig title “dimensionality vs leakage vs OOD” | Three series vs λ | Not an AUROC-vs-leakage plot | Left. Axis is λ |
| `figures/preview_leakage_vs_ood.{png,pdf}` | AUROC vs leakage, n=4 | Asserts the mediation just disproved | Stays appendix-only; MASTER caption now says doubly superseded. Not regenerated |
| `fig_dial` itself | Panels vs λ_adv | Compliant | Caption only |

No other report, caption or table under `results/paperB/` asserted that reducing leakage causes the OOD collapse. Historical Phase 0–2 uses of “leakage” are the domain-probe metric or data-split hygiene, not this mediation.

## What was not changed

- Table 1 still reports leakage balanced accuracy as a measured column.
- `fig_dial` still has a leakage panel. The x-axis remains λ_adv.
- Phase 12.1b still treats leakage as the invariance diagnostic on Camelyon (that is what the experiment measures). Outcome (c) is “this recipe did not induce invariance”, not “leakage caused OOD”.
