# Paper B — outline and draft abstract

Target: **Medical Image Analysis** (primary) or **TMLR** (secondary — no novelty gate, reviews on evidence quality, no page limit).

`manuscript_main.md` is for the previous paper and is **superseded, not edited**. Start a new file.

---

## Title candidates

1. **Confidently Wrong on Unseen Domains: Domain-Adversarial Training Silently Disables Out-of-Distribution Monitoring**
2. **The Safety Gate Fails While Every Metric Improves: Invariance Training and OOD Monitoring in Skin Lesion Classification**
3. **Invariance Has a Hidden Cost: Adversarial Shortcut Removal Breaks Out-of-Distribution Detection**

(1) names both the harm and its invisibility. Prefer it unless the venue dislikes colons.

---

## Draft abstract (~280 words)

> **Background.** Domain-adversarial invariance training is widely recommended for removing acquisition shortcuts from medical imaging models. Its effect on out-of-distribution (OOD) monitoring has not been measured.
>
> **Methods.** We trained a dual-encoder factorization model on ISIC 2019 (dermoscopy, 25,331 images) and PAD-UFES-20 (clinical photographs, 2,298), sweeping the adversarial weight λ_adv across seven values with 3–5 seeds each (27 runs). At each λ we measured domain leakage by linear probe, in-distribution balanced accuracy, calibration on both in-distribution and out-of-domain data, and OOD detection using five detectors (MSP, Energy, cosine, Mahalanobis, kNN). We validated on Fitzpatrick17k, a third dataset never seen by the adversary.
>
> **Results.** Calibration on out-of-domain data degraded from ECE 0.247 to 0.746 as λ_adv rose, while in-distribution ECE stayed flat at ≈0.10. Mean confidence on never-seen-domain images rose from 0.489 to 0.930, overtaking in-distribution confidence (0.910). OOD detection collapsed from AUROC 0.863 to 0.508 at the smallest nonzero weight tested and inverted below chance (0.413), with all five detectors collapsing together; OOD samples were ranked as more in-distribution than in-distribution data throughout the score distribution, not only in the tails. The failure was invisible to routine monitoring: domain leakage improved (0.915 → 0.553) and in-distribution balanced accuracy did not fall (0.692 → 0.707). Leakage did not mediate the collapse — the two moved in opposite directions over part of the range, and at the backbone leakage was flat while OOD AUROC fell from 0.749 to 0.475. Findings replicated on Fitzpatrick17k (0.399), where a non-adversarial baseline reached 0.994. Six candidate explanations were tested and refuted.
>
> **Conclusions.** When domain-adversarial training succeeds, it silently disables OOD monitoring and leaves models confidently wrong on unseen domains, while every routinely monitored metric improves. OOD detection and out-of-domain calibration should be reported as a function of achieved invariance, not at a single operating point.

**Placeholder to fill:** one clause on cross-domain diagnostic accuracy (Item 1 / Phase 6, array 60585). If it is flat across λ, add *"while cross-domain diagnostic accuracy did not improve"* — that completes the cost argument. **This number has not yet been reported; retrieve it before finalising.**

---

## Section structure

### 1. Introduction

- Shortcut learning in medical imaging; adversarial invariance as the standard remedy.
- The gap: nobody measures what invariance does to the safety machinery that is supposed to catch distribution shift at deployment.
- Preview the finding in one paragraph, leading with calibration, not AUROC.
- **Contributions:** (i) a controlled λ sweep isolating the effect; (ii) the calibration/confidence failure; (iii) the OOD detection cliff with all detectors; (iv) demonstration that the failure is invisible to monitored metrics and not mediated by the standard leakage probe; (v) six refuted explanations; (vi) a protocol recommendation.

### 2. Related work

Shortcut learning in dermatology (Bissoto, Winkler, Daneshjou); DANN/DSN and invariant representation learning; OOD detection and the covariate/semantic shift distinction (full-spectrum OOD, OpenOOD); calibration under shift.

**State plainly that the architecture is DSN-equivalent and is not a contribution.** The contribution is the measurement.

### 3. Methods

Datasets and licences; preprocessing; architecture; training objective; the λ sweep (7 values, 3–5 seeds, 27 runs); the five detectors with statistics fit on ISIC train only; the leakage probe (balanced accuracy, with the correct chance floor — **audit which floor each table uses**, 0.688 is a plain-accuracy majority, the 2-class balanced-accuracy floor is 0.5); Fitzpatrick17k as a never-adversarial third domain.

### 4. Results

**4.1 — Models become confidently wrong on unseen domains** *(lead)*
OOD ECE 0.247 → 0.746; ID ECE flat ≈0.10; the confidence crossing at λ≈0.5.

**4.2 — OOD detection collapses, as a cliff, across every detector**
0.863 → 0.508 at λ=0.25 → 0.413 at λ=2. MSP 0.970 → 0.437, Energy 0.983 → 0.441, cosine 0.901 → 0.435, kNN 0.941 → 0.424 (stable for k ∈ {10, 50, 200}). **It is a cliff, not a trade-off curve — there is no safe operating point.** Bulk location shift, not a tail effect (dropping p90/p95/p99 leaves it at 0.412–0.424; medians invert, ID 13.8 vs OOD 10.0).

**4.3 — The failure is invisible**
Leakage improves, ID balanced accuracy does not fall, cross-domain accuracy [pending]. Nothing in a standard evaluation reveals it.

**4.4 — Leakage does not mediate the collapse**
Two independent results: λ=0.25→2 leakage rises while AUROC falls; backbone leakage flat 0.982 → 0.941 while backbone Mahalanobis falls 0.749 → 0.475. **Plot everything against λ, never against leakage** — an AUROC-vs-leakage plot asserts the mediation this section disproves.

**4.5 — What it is not: six refuted explanations** *(main text, not appendix)*
Image memorisation (P2: 0.427 vs 0.407); class composition (0.414 vs 0.409, and reweighting *strengthens* to 0.240); domain specificity (Fitzpatrick 0.399, baseline 0.994 on the same data); dimensional collapse (PR 4.879 → 4.500 while leakage 0.914 → 0.553; Ledoit-Wolf leaves 0.413 unchanged); latent compression (OOD is *already* 4.5× tighter at λ=0 and the adversary *decompresses* it); tail asymmetry (refuted by the truncation test). Plus a pre-registered synthetic reproduction that failed.

**4.6 — What survives**
Semantic OOD is unaffected in the same way (both branches 0.6–0.8, no clean split — reported as a null result with 4a and 4b at equal prominence). A domain monitor is cheap: frozen ImageNet ResNet-50 with a linear head reaches 0.998. `z_context`'s distinctive property is concentration onto one axis (k=1 AUROC > 0.9999 at 84.5% variance vs 0.55–0.70 elsewhere), **not** detection performance. Report `z_context` numbers as **> 0.9999 (2 discordant pairs / 11.6M)**, never as 1.0000.

### 5. Discussion

- **The danger zone is where the method succeeds.** A weak adversary produces neither debiasing nor the safety failure — Phase 12's audit showed leakage frozen at 0.96 across four orders of magnitude of λ with an adversary that never trained. Risk scales with *achieved* invariance, not with nominal λ or with having an adversarial term in the loss.
- **Recommendation (the answer to "so what do I do"):** report OOD detection and out-of-domain calibration as a function of achieved invariance, not at a single operating point; and do not use a different dataset as the OOD set, since that measures covariate shift. Near-zero cost, turns an invisible failure into a visible number.
- **Open mechanism.** Six accounts refuted, one pre-registered synthetic reproduction failed. State it plainly; the eliminations are a contribution to whoever solves it.

### 6. Limitations

Two domains and essentially one modality contrast (dermoscopy vs clinical photography); Fitzpatrick17k is also clinical photography and does not resolve this. Image-level rather than patient-level ISIC splits — note that ID accuracy is compared *across λ under an identical split*, so any inflation applies equally to every row and does not threaten the claim. No background-only shortcut attribution, no fairness subgroup analysis. **Phase 12 (Camelyon17/WILDS) is inconclusive**: the adversary head never trained (CE = ln 3 from epoch 1 at every λ, post-hoc probe 0.94) despite a gradient graph verified correct (head cosine +1.000, encoder −1.000). No Camelyon17 result is reported. **Do not write that Camelyon17 resists invariance.** Record the diagnostic itself — an adversary that fails to train under a verified-correct GRL — as a breadcrumb for future work. Generality beyond dermatology is untested.

### 7. Conclusion

One paragraph. Close on the protocol recommendation, not on the architecture.

---

## Figures

**Figure 1 — four panels, shared λ axis.** Leakage · ID balanced accuracy · OOD AUROC · OOD ECE. Two panels improve, two collapse, and nobody monitoring the first two would notice. **This single figure is the paper.**

**Figure 2 — confidence.** Mean MSP for ISIC test, PAD held-out and Fitzpatrick17k vs λ, with the crossing marked.

**Figure 3 — every detector together.** Five detectors vs λ; they fall as one.

**Figure 4 — it generalises.** PAD and Fitzpatrick17k collapse together; the non-adversarial baseline's 0.994 on Fitzpatrick as a reference line.

**Figure 5 — leakage does not mediate.** Backbone leakage flat against backbone AUROC falling.

**Supplementary.** Semantic OOD null (4a and 4b equal prominence); domain-axis distributions, captioned only as *"Fitzpatrick17k does not fall on the ISIC side"* (overlap with PAD is 0.47); the n=4 preview curve, captioned as superseded.

## Tables

**Table 1** — the dial: per λ, leakage · ID balanced accuracy · ID ECE · OOD ECE · OOD AUROC (PAD) · OOD AUROC (Fitzpatrick) · 6-class-restricted AUROC · cross-domain accuracy [pending] · `z_context` AUROC. Baseline and EffB3 control as reference rows.

**Table 2** — a monitor is cheap: each representation × (linear domain-head AUROC, PCA k=1 AUROC, variance at k=1). **Lead with the k=1 column**; the AUROC column shows near-parity.

**Table 3** — the six refuted explanations: prediction, test, result, what it rules out.

---

## Before drafting can finish

1. **Retrieve the cross-domain PAD diagnostic number** (Item 1 / Phase 6, array 60585). It completes the cost argument in the abstract and Table 1.
2. **Audit the leakage chance floors** across every table — plain-accuracy majority (0.688) versus balanced-accuracy floor (0.5). Fix any mixed comparisons. This changes no conclusion but is exactly the kind of error that costs credibility.
3. Apply the `> 0.9999 (2 discordant pairs / 11.6M)` convention everywhere.
4. Complete the causal-claim audit corrections from `CAUSAL_CLAIM_AUDIT.md`.

## Writing rules for this manuscript

- Every comparison uses **EffNet-B3 single-encoder** as the primary control. The honest ID gain is **+1.9pp (0.683 → 0.702)**, never the +4.7pp against ResNet-50.
- Never describe the collapse as a tunable trade-off. It is a cliff.
- Never claim leakage reduction causes the OOD collapse.
- The six refutations go in the main text. They are the reason the result is credible, and hiding them makes the paper weaker, not stronger.
