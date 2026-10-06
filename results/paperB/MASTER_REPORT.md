# Paper B — MASTER REPORT (writing package)

Numbers as measured. Detectors, splits and hyperparameters were never retuned toward a prediction. Writable tree: `/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/`. Original `CSG-Skin/` was not overwritten.

**Headline.** Adversarial invariance training makes a dermatoscope model more confident on never-seen-domain data than on its own test set. OOD expected calibration error on pad_heldout rises **0.247 → 0.746** while ID ECE stays flat at **~0.10**; mean OOD softmax confidence overtakes ID by λ=2. The same training collapses covariate-shift OOD detection from **0.86 to 0.51** at λ_adv=0.25 and inverts it below chance thereafter. ID balanced accuracy does not drop. ECE is the lead result: clinicians and regulators already use it, and it does not require Mahalanobis.

**It is a cliff, not a trade-off curve. There is no safe operating point.** Do not describe this as a tunable trade-off.

**Corrected causal claim.** Adversarial training causes both the leakage drop and the OOD collapse. The OOD collapse is **not mediated** by linear domain-decodability. Phase 3: from λ=0.25 to λ=2, leakage rises (0.553 → 0.584) while AUROC falls (0.508 → 0.413). Phase 13.2: backbone leakage is flat (0.982 → 0.941) while backbone Mahalanobis falls 0.749 → 0.475. Domain information remains linearly decodable; OOD detection collapses anyway. The collapse follows from geometric changes a linear domain probe does not capture. Plot the dial against λ, never against leakage.

SPS (Semantic Purity Score) is **withdrawn**. The n=4 preview curve is **doubly superseded**: it is a correlation across heterogeneous models, and it is an AUROC-vs-leakage plot that asserts the mediation just disproved. Appendix only. It is not evidence.

Never print `1.0000 ± 0.0000`. Form: **AUROC > 0.9999 (2 discordant pairs / 11.6M)**.

---

## The four-column dial (the paper)

PAD transfer is **6-class-restricted** (PAD covers MEL, NV, BCC, AK, BKL, SCC; no DF, no VASC), evaluated on **pad_heldout only** (716 images / 412 patients). PAD labels never entered `L_cls` (`ignore_index=-1`): zero-shot cross-domain transfer. Phase 3 `pad_acc` was computed on `pad_full` and is contaminated; it is not used here.

Leakage is **2-class balanced accuracy** (ISIC vs PAD). Its chance floor is **0.5**. The figure 0.688 is the **plain-accuracy** majority on the Phase 0 label-only probe set (always predict ISIC) and is not the floor for this column. λ=0.25 leakage 0.553 is 0.053 above that floor — closer to chance than a mixed-scale comparison against 0.688 would suggest.

| λ_adv | n | leakage bal acc | ID bal acc | OOD AUROC PAD | PAD xfer bal acc (6-cls, heldout) |
|---:|---:|---|---|---|---|
| 0 | 5 | 0.915 ± 0.021 | 0.692 ± 0.015 | 0.863 ± 0.025 | 0.291 ± 0.016 |
| 0.25 | 3 | 0.553 ± 0.008 | 0.696 ± 0.009 | 0.508 ± 0.022 | 0.291 ± 0.020 |
| 0.5 | 3 | 0.555 ± 0.029 | 0.701 ± 0.016 | 0.449 ± 0.008 | 0.270 ± 0.034 |
| 1 | 3 | 0.569 ± 0.010 | 0.701 ± 0.013 | 0.439 ± 0.040 | 0.261 ± 0.016 |
| 2 | 5 | 0.584 ± 0.027 | 0.707 ± 0.009 | 0.413 ± 0.021 | 0.249 ± 0.030 |
| 4 | 3 | 0.548 ± 0.035 | 0.686 ± 0.015 | 0.443 ± 0.038 | 0.276 ± 0.025 |
| 8 | 5 | 0.625 ± 0.045 | 0.679 ± 0.013 | 0.481 ± 0.026 | 0.269 ± 0.023 |

ResNet-50 and EffB3 16-d are reference rows in Table 1 below, not points on this dial.

### Cross-domain cost (Item 1)

Full diagnosis: `phase6_xfer/PHASE6_XFER_DIAGNOSIS.md` (array 60585, 37 JSON files). Floors for this column: 6-class balanced-acc chance **1/6 ≈ 0.167**; 6-class plain-acc majority (always BCC) **276/716 = 0.385**. Do not use 0.385 as the balanced-accuracy floor.

Cross-domain 6-class balanced accuracy on pad_heldout is flat or falls across λ (λ=0: 0.291 ± 0.016; λ=0.25: 0.291 ± 0.020; mean λ>0: 0.267; λ=2: 0.249 ± 0.030). Plain accuracy falls (0.245 → 0.215 at the cliff; 0.184 at λ=2). Macro AUC is flat (0.605 → 0.621). You pay a safety cost and get nothing for it. The abstract sentence is **“while cross-domain accuracy does not improve at all.”** This is the strongest version of the paper. It is not a trade-off.

Balanced accuracy being flat at 0→0.25 hides a collapse onto NV: mean predicted NV 113 → 374; AK recall 0.245 → 0.028; SCC recall 0.088 → 0.015. CSG plain acc at every λ is below the majority floor 0.385. The entangled baseline transfers better (bal acc 0.391 ± 0.028) than CSG λ=0.

| λ_adv | n | plain acc | bal acc | macro AUC | NV rec | AK rec | pred NV |
|---:|---:|---|---|---|---|---|---:|
| 0 | 5 | 0.245 ± 0.033 | 0.291 ± 0.016 | 0.605 ± 0.013 | 0.498 ± 0.026 | 0.245 ± 0.102 | 113 |
| 0.25 | 3 | 0.215 ± 0.009 | 0.291 ± 0.020 | 0.621 ± 0.026 | 0.894 ± 0.033 | 0.028 ± 0.030 | 374 |
| 2 | 5 | 0.184 ± 0.002 | 0.249 ± 0.030 | 0.581 ± 0.019 | 0.884 ± 0.012 | 0.016 ± 0.012 | 426 |

Melanoma sensitivity at specificity 0.85: λ=0 0.489 ± 0.099; λ=0.25 0.333 ± 0.111; λ=2 0.222 ± 0.176. **pad_heldout contains 9 melanoma images**. No significance claim is attached to that cell.

---

## The cliff (the comparison that carries the paper)

λ_adv=0 (n=5) vs λ_adv=0.25 (n=3), unpaired. Lead with effect size and CI, not p. Family **F_cliff** (4 tests, Holm–Bonferroni α=0.05): OOD AUROC PAD, leakage balanced acc, ID balanced acc, PAD transfer balanced acc.

| Contrast | λ=0 | λ=0.25 | Δ | bootstrap 95% CI | Cohen's d | Welch p | Holm reject |
|---|---|---|---|---|---|---|---|
| OOD AUROC PAD (z_lesion Maha) | 0.863 ± 0.025 | 0.508 ± 0.022 | 0.354 | [0.327, 0.383] | 14.66 | 5.754e-06 | yes |
| leakage balanced acc | 0.915 ± 0.021 | 0.553 ± 0.008 | 0.362 | [0.345, 0.379] | 20.69 | 9.74e-08 | yes |
| ID balanced acc | 0.692 ± 0.015 | 0.696 ± 0.009 | -0.004 | [-0.018, 0.010] | -0.29 | 0.664 | no |
| PAD transfer balanced acc (6-cls, heldout) | 0.291 ± 0.016 | 0.291 ± 0.020 | 0.000 | [-0.022, 0.022] | 0.01 | 0.9857 | no |

OOD AUROC PAD collapses by **0.354** (bootstrap 95% CI [0.327, 0.383], Cohen's d = 14.66, n=5 vs n=3). The CI excludes 0: **true**.

Paired seed-wise comparisons are available only for seeds {42,52,62} (the interior grid). Those three paired Δs on OOD AUROC PAD:

- seed 42: 0.874 → 0.511 (Δ=0.364)
- seed 52: 0.895 → 0.485 (Δ=0.410)
- seed 62: 0.870 → 0.529 (Δ=0.341)

Paired mean Δ = 0.372, Cohen's d (paired) = 10.70.

Sample-level bootstrap 95% CIs (2000 resamples) for headline cells are in `results/paperB/phase10/sample_bootstrap.json`. Seed-level CIs above are the right object for the λ=0 vs 0.25 contrast (the intervention is a training run, not a test image).

---

## Mechanism verdict

**NOT SUPPORTED — inversion left unexplained. Six accounts are closed.**

The inversion is a **bulk location shift**: OOD samples are ranked as more typical than ID samples across the whole score distribution, not at the extremes (Phase 14.A: hold median 10.0 vs ID 13.8; dropping the tail does not remove the inversion). It is not a tail effect, not latent compression (13.4), not dimensional collapse (13.1), not class mix, not image memorisation, and not PAD-specific. A locked synthetic reproduction failed twice under pre-registration (13.5 v1: BN artifact at λ=0; v2: legal start 0.693, no inversion). No post-hoc story is constructed. That is a publishable position.

| λ | n | PAD Maha median ratio | PAD Euclid median ratio | PAD Maha IQR ratio | PAD kNN median ratio | PAD norm median ratio | Fitz Maha median ratio |
|---:|---:|---|---|---|---|---|---|
| 0 | 5 | 2.948 ± 0.306 | 2.227 ± 0.108 | 0.740 ± 0.056 | 3.283 ± 0.223 | 0.377 ± 0.031 | 2.312 ± 0.320 |
| 0.25 | 3 | 1.117 ± 0.068 | 1.148 ± 0.062 | 1.129 ± 0.102 | 1.121 ± 0.032 | 0.972 ± 0.006 | 0.804 ± 0.219 |
| 2 | 5 | 0.727 ± 0.101 | 0.841 ± 0.064 | 0.938 ± 0.078 | 0.851 ± 0.021 | 0.993 ± 0.033 | 0.653 ± 0.164 |
| 8 | 5 | 0.998 ± 0.153 | 1.017 ± 0.096 | 1.132 ± 0.082 | 1.012 ± 0.100 | 0.991 ± 0.012 | 0.905 ± 0.239 |

Prediction (pre-registered): ratio > 1 at λ=0, < 1 by λ=0.25, falling further as λ rises; OOD IQR tightens. Feature norm is a scaling control.

---

## Phase 13.3 — lead result: OOD ECE

ID ECE stays at ~0.10 across the sweep. OOD ECE on pad_heldout rises **0.247 ± 0.043 → 0.746 ± 0.005** from λ=0 to λ=2. Mean softmax confidence on pad_heldout and Fitzpatrick17k overtakes ID by λ=2 (ID 0.910, hold 0.930, Fitz 0.935). The model becomes more sure, and worse calibrated, on data it has never seen. This number does not require Mahalanobis.

## Phase 13.2 — backbone decoupling (a finding, not a detail)

At the backbone, linear domain leakage is essentially flat (**0.982 → 0.941**) while backbone Mahalanobis falls **0.749 → 0.475**. Domain information is still linearly decodable; OOD detection collapses anyway. Combined with Phase 3 (leakage rises 0.553 → 0.584 while AUROC falls 0.508 → 0.413 between λ=0.25 and λ=2), this is the evidence that the OOD collapse is not mediated by the standard leakage probe.

## Table 1 — full dial

OOD AUROC on PAD is Phase 3 `pad_full` unrestricted Maha on `z_lesion`, matching the locked 0.86 → 0.51 number. Transfer is `pad_heldout` only. `z_context` is domain-supervised. Leakage is 2-class balanced accuracy, reported as a **measured quantity**, not as the explanatory variable; chance floor 0.5 (not 0.688). The ID ECE column below is in-distribution; the lead OOD ECE series is in the 13.3 section.

| λ_adv | n | leakage bal acc | ID bal acc | ECE | OOD AUROC PAD | OOD AUROC Fitz | 6-class OOD AUROC | PAD xfer bal acc (6-cls, heldout) | z_context OOD |
|---:|---:|---|---|---|---|---|---|---|---|
| 0 | 5 | 0.915 ± 0.021 | 0.692 ± 0.015 | 0.107 ± 0.005 | 0.863 ± 0.025 | 0.764 ± 0.038 | 0.872 ± 0.025 | 0.291 ± 0.016 | 1.000 ± 0.000 |
| 0.25 | 3 | 0.553 ± 0.008 | 0.696 ± 0.009 | 0.100 ± 0.006 | 0.508 ± 0.022 | 0.439 ± 0.061 | 0.515 ± 0.022 | 0.291 ± 0.020 | > 0.9999 |
| 0.5 | 3 | 0.555 ± 0.029 | 0.701 ± 0.016 | 0.096 ± 0.005 | 0.449 ± 0.008 | 0.449 ± 0.025 | 0.455 ± 0.008 | 0.270 ± 0.034 | > 0.9999 |
| 1 | 3 | 0.569 ± 0.010 | 0.701 ± 0.013 | 0.097 ± 0.004 | 0.439 ± 0.040 | 0.457 ± 0.088 | 0.445 ± 0.040 | 0.261 ± 0.016 | > 0.9999 |
| 2 | 5 | 0.584 ± 0.027 | 0.707 ± 0.009 | 0.099 ± 0.002 | 0.413 ± 0.021 | 0.391 ± 0.068 | 0.418 ± 0.022 | 0.249 ± 0.030 | 1.000 ± 0.000 |
| 4 | 3 | 0.548 ± 0.035 | 0.686 ± 0.015 | 0.100 ± 0.007 | 0.443 ± 0.038 | 0.431 ± 0.036 | 0.449 ± 0.039 | 0.276 ± 0.025 | 1.000 ± 0.000 |
| 8 | 5 | 0.625 ± 0.045 | 0.679 ± 0.013 | 0.100 ± 0.008 | 0.481 ± 0.026 | 0.474 ± 0.060 | 0.487 ± 0.026 | 0.269 ± 0.023 | > 0.9999 |
| ResNet-50 baseline | 5 | 0.979 ± 0.001 | 0.655 ± 0.018 | — | 0.868 ± 0.019 | 0.994 | — | 0.391 ± 0.028 | n/a |
| EffB3 16-d control | 5 | 0.802 ± 0.020 | 0.683 ± 0.012 | — | 0.726 ± 0.021 | 0.649 | — | 0.298 ± 0.010 | n/a |

---

## Table 2 — a monitor is cheap

Lead with the **k=1** column. Linear domain-head AUROC is near-parity and is **not** a `z_context` advantage. Frozen ImageNet ResNet-50, never trained on this data, reaches 0.998.

| Representation | PCA k=1 AUROC (var) | Linear domain-head AUROC | supervision |
|---|---|---|---|
| ImageNet ResNet-50 (frozen) | 0.647 (14.6%) | 0.998 ± 0.0003 | ImageNet |
| ImageNet EffNet-B3 (frozen) | 0.545 (10.7%) | 0.994 ± 0.0013 | ImageNet |
| Trained R50 backbone_raw | 0.648 (32.5%) | 0.997 ± 0.0011 | class-supervised |
| Trained EffB3 backbone_raw | 0.698 (10.9%) | 0.991 ± 0.0015 | class-supervised |
| z_context | > 0.9999 (84.5%) | > 0.9999 | domain-supervised |

`z_context` is domain-supervised. Factorization does not make domain detectable — it already is. Factorization concentrates domain onto k=1 (AUROC 1.00 at 84.5% variance vs 0.55–0.70 for every other representation).

---

## Table 3 — refuted mechanisms (main text, not appendix)

| Prediction | Test | Result | Rules out |
|---|---|---|---|
| Image memorisation of PAD in L_adv | Phase 2: retrain with pad_heldout never in L_ctx/L_adv (716 / 412 patients) | pad_heldout z_lesion Maha 0.427 ± 0.025 vs pad_adv 0.407 ± 0.021 (Δ ≈ 0.02) | Inversion is not memorisation of adversarial images |
| PAD-specific mapping (would be ~0.50 on a third domain) | Phase 2.5a: Fitzpatrick17k, never in any train branch, n=3887 | z_lesion Maha 0.399 ± 0.039, same inversion as PAD 0.409 | Not PAD-specific; structural of the invariant branch |
| Class-mix / missing DF,VASC drives AUROC | Phase 2.5b: 6-class-restricted ID and ID reweighted to PAD mix | 6-class 0.414 vs 0.409; reweight 0.240 (inversion strengthens) | Class composition is not the driver |
| Clean 2×2: z_context detects domain not class; z_lesion the reverse | Phase 4: 4a hold {DF,VASC}, 4b hold {SCC} | 4a z_lesion 0.648 vs z_context 0.641 vs baseline 0.687; 4b z_lesion 0.806 vs z_context 0.607 | No double dissociation. Semantic OOD is graded, 4a=4b equal prominence, null 2×2 |
| z_context 4a is a VASC-colour detector (high VASC, ~0.50 DF) | Phase 4.5b: class-conditional Maha on 4a | z_context DF 0.61, VASC 0.67 — both above 0.50 | VASC/DF hypothesis not confirmed |
| 4b win is near-OOD, 4a loss is far-OOD | Phase 4.5c: cosine-to-nearest-keep on DF/VASC/SCC | cosines 0.570 / 0.566 / 0.565; Pearson(advantage, cosine)=−0.12 | No near/far axis. 4a and 4b simply differ |
| Leakage↔AUROC is a continuous trade-off (SPS) | Phase 3 λ sweep: leakage and AUROC between λ=0.25 and λ=2 | leakage 0.553 → 0.584 while AUROC 0.508 → 0.413. Linearity is an artefact of anchoring at λ=0 | SPS withdrawn. Not a tunable trade-off. There is no safe operating point |
| OOD collapse is mediated by reduced linear domain leakage | Phase 3 λ=0.25→2 + Phase 13.2 backbone | leakage 0.553→0.584 while AUROC 0.508→0.413; backbone leak 0.982→0.941 while backbone Maha 0.749→0.475 | Mediation rejected. Joint effects of λ, not leakage→OOD. "Adversarial training breaks OOD detection through a channel invisible to the standard leakage probe" |
| Inversion is an ID-tail effect (Maha tail-sensitive; OOD tail clipped) | Phase 14.A2: drop pooled p90/p95/p99, recompute AUROC | λ=2 pad_heldout 0.427 → 0.412 / 0.416 / 0.424. Inversion persists in the bulk | Tail hypothesis wrong. The bulk itself is inverted (hold median 10.0 vs ID 13.8) |
| Controlled toy with heavy-tailed ID reproduces F1–F7 | Phase 13.5 v2, pre-registered, BN off | λ=0 Maha 0.693 (legal start); λ=2 Maha 0.786. No inversion | Synthetic reproduction failed. No third revision |
| OOD collapse is because z_lesion is only 16-d (original manuscript) | Phase 1 EffB3 16-d control + Phase 3 λ=0 (same 16-d CSG) | EffB3 16-d Maha 0.726; CSG λ=0 Maha 0.863; CSG λ=0.25 Maha 0.508 | Dimensionality is not the cause. λ_adv is the cause |

---

## Figures

All 300 dpi, PNG+PDF, colourblind-safe (Wong), `results/paperB/figures/`.

| File | Panel |
|---|---|
| `fig_dial` | Three panels sharing **λ_adv** (never leakage): leakage as a measured quantity, ID bal acc, OOD AUROC. Cliff 0→0.25 annotated. Leakage panel chance floor is 0.5. ECE is the lead result (Phase 13.3); this figure is the λ-dial, not the explanation. |
| `fig_mechanism` | Distance-ratio curve (OOD ÷ ID) vs λ; Maha densities at λ=0 and λ=2. |
| `fig_monitor` | z_lesion vs z_context OOD AUROC across the dial. One collapses, one stays ~1.0. |
| `fig_generalises` | PAD and Fitzpatrick17k collapse together; baseline Fitz 0.994 as a reference line. |
| `fig_semantic_ood_supp` | 4a and 4b, both branches, equal prominence, null result. |
| `fig_domain_axis_supp` | ISIC / Fitzpatrick17k / PAD on the domain axis. Caption: Fitzpatrick17k does not fall on the ISIC side. Overlap with PAD is 0.47. |
| `fig_preview_superseded_supp` | n=4 preview curve. Doubly superseded: heterogeneous-model correlation, and AUROC-vs-leakage asserts a mediation Phase 3 and 13.2 refute. Appendix only. Not evidence. |

---

## Limitations (not run, with justification)

**Phase 7 (patient-level ISIC splits). Not run.** ID balanced accuracy is compared across λ under an identical split, so any split-induced inflation applies equally to every row and does not threaten the claim. The absolute ID numbers may be optimistic; the relative pattern is unaffected.

**Phase 8 (background-only). Not run.** The paper no longer makes a shortcut-attribution claim, so this is no longer load-bearing.

**Phase 5 (full per-class conditional probe). Partially covered.** The label-only domain probe (0.7968 vs majority 0.688, **plain accuracy**) from Phase 0 is the key control and is reported. The leakage column in the dial is 2-class **balanced** accuracy (chance floor 0.5), not plain accuracy against 0.688. The per-class conditional probe is future work.

**Non-medical replication. Not a result.** Camelyon17 Phase 12 is inconclusive: the online adversary never trained (CE = ln 3 at every epoch; post-hoc probe still 0.94). A 30-minute GRL sign check shows the graph is standard (head cosine +1, encoder −1). This is not evidence that Camelyon resists invariance. No Camelyon number is reported. Limitations, not Results. Left for later work. iWildCam remains gated.

**Raw-source trivial-detector repeat. Blocked** by PAD PNG permissions (owner `anhnv`, mode 600). Known gap. Not ours to chmod.

**Phase 9 (Fitzpatrick strata on PAD transfer). Not run.** PAD `metadata.csv` is a dangling symlink (`/mnt/data2/Vinh/Ban_sao_datn/...`); Fitzpatrick group labels are not readable. Optional and not load-bearing. `pad_heldout` melanoma n=9 would also make per-group melanoma sensitivity uninterpretable for several strata.

**SPS.** Withdrawn, not a limitation of execution.

---

## Claims in the original manuscript this work contradicts

The original `manuscript_main.md` was written for a different paper (CSG-Lite as a utility–leakage method; OOD collapse as a 16-d limitation) and will be rewritten, not patched. It was not found under `CSG-Skin/` in this tree; claims below are taken from that manuscript's published numbers in `results/cbm_revision/`, `results/table1_draft.md`, and the Paper B work-order recap of what the manuscript claimed.

| Original claim | Corrected |
|---|---|
| CSG-Lite Maha OOD AUROC ~0.41 is a limitation of 16-dimensional `z_lesion` | False. EffB3 16-d control is **0.726 ± 0.021**. CSG at λ_adv=0 (same 16-d) is **0.863 ± 0.025**. Collapse is caused by λ_adv, already **0.508 ± 0.022** at λ=0.25. |
| ResNet-50 leaks domain at probe acc **0.9791**; CSG-Lite reduces it to **~0.71**, and that reduction is the result | Leakage reduction is real. Phase 3 **balanced** acc: λ=0 0.915 → λ=0.25 0.553 (floor 0.5). Published runB **plain** acc ~0.72 (floor 0.688) is a different metric on a different checkpoint; do not put the two in one clause. It is the wrong success metric: a co-effect of λ, not the cause of the OOD cliff (Phase 3 opposite directions; Phase 13.2 backbone decoupling). A practitioner watching leakage, ID accuracy and ID ECE sees improvement while OOD ECE and the safety gate fail. |
| Run B (orth=1) is the strongest method by a utility–leakage criterion | That criterion ignores covariate-shift OOD. Under it, the operating point that looks best is the one where Maha AUROC is ~0.41. |
| Leakage vs OOD AUROC is a continuous, tunable relationship (preview slope; SPS) | Phase 3 refutes continuity. Between λ=0.25 and λ=2 leakage rises while AUROC falls. SPS withdrawn. Preview n=4 is a heterogeneous-model correlation, superseded, appendix only. |
| `z_context` is the reason domain is detectable | Frozen ImageNet ResNet-50 linear head **0.998 ± 0.0003**. Near-parity of linear AUROC. Architectural remainder is k=1 concentration (84.5% var, AUROC 1.00), not detectability. `z_context` is domain-supervised; say so. |
| Clean factorization: lesion branch for class, context branch for domain (implied 2×2 with semantic OOD) | Domain OOD is split (`z_context` > 0.9999 vs `z_lesion` 0.41). Semantic OOD is not: both branches 0.6–0.8, order depends on held-out class. 4a and 4b at equal prominence. No near-OOD story (4.5c null). |
| Shortcut / background attribution as the mechanism of leakage | Not tested (Phase 8 not run). The paper no longer makes this claim. |

---

## Discussion — a weak adversary is not the danger zone

This paragraph is independent of how the Camelyon17 matched-recipe rerun resolves. It is already established by the coarse scan plus the configuration audit (`results/paperB/phase12/PHASE12_1B_AUDIT.md`).

The Camelyon17 coarse DANN (SGD, one param group, no feature BN, unbalanced hospital batches, Linear→ReLU→Linear head, GRL on raw features, `adv_lr_multiplier=1`) left 3-class leakage frozen at 0.977–0.963 across λ ∈ {0, 0.1, 1, 10}. Seven of eight adversary knobs mismatched the derm recipe that actually moved leakage. Those four points are void as evidence about the OOD cliff.

A weak adversary produces neither the leakage drop nor the OOD collapse. Both are effects of a working adversarial objective. That does not mean the collapse is mediated by the leakage probe. Linear domain-decodability can stay high (backbone 0.94) while OOD detection fails. Adversarial training breaks OOD detection through a channel invisible to the standard leakage probe. Many deployed DANN setups are too weak to induce either effect and are correspondingly uninformative about this failure. λ alone does not characterise adversarial pressure. The audit table supports that last sentence, not a mediation claim.

---

## Locked numbers (unchanged)

- Phase 1: `z_context` > 0.9999 (2 discordant pairs / 11.6M); `z_lesion` 0.409 ± 0.020; R50 0.868 ± 0.019; EffB3 0.726 ± 0.021; ID bal CSG 0.702 vs EffB3 0.683.
- Phase 1.5: trivial Maha max 0.647; logistic 0.876; PCA k=1 `z_context` AUROC 1.00 at 84.5% var.
- Phase 2: pad_adv 1582/961, pad_heldout 716/412, zero patient overlap. Heldout Maha 0.427 ± 0.025.
- Phase 2.5: Fitz 0.399 ± 0.039; axis OVL Fitz–PAD 0.47; claim: not on the ISIC side.
- Label-only domain probe: 0.7968 vs majority 0.688 (**plain** accuracy; not the leakage-column floor).
- Leakage column: 2-class balanced accuracy; chance floor 0.5. λ=0.25: 0.553 (0.053 above floor). Same probe, plain acc 0.702 (0.014 above majority 0.688). Both near chance. Audit: `LEAKAGE_FLOOR_AUDIT.md`.
- Item 1 pad_heldout 6-class transfer: bal acc 0.291 → 0.291 at the cliff, then 0.249 at λ=2. Plain acc 0.245 → 0.215. NV collapse. Diagnosis: `phase6_xfer/PHASE6_XFER_DIAGNOSIS.md`.

Four (now more) predictions failed and were reported as findings. That is why the result is credible.

## Files

- Phase 6: `phase6_xfer/` (diagnosis: `phase6_xfer/PHASE6_XFER_DIAGNOSIS.md`)
- Leakage floor audit: `LEAKAGE_FLOOR_AUDIT.md`
- Phase 3.5: `phase35_mech/`
- Phase 10: `phase10/` (this package's json)
- Figures: `figures/fig_*.{png,pdf}`
- This report: `MASTER_REPORT.md`
- Phase 14: `phase14/PHASE14_REPORT.md` (tails, saturation, causal-claim audit)
- Phase 13.5 v2: `phase13_5/v2/PHASE13_5_V2_REPORT.md`

