# Claims and evidence

Every proposition the paper is entitled to make, with the number behind it and where
that number lives. No prose — write from this, not from the existing draft.

**Partition rule:** all main-text OOD numbers are `pad_heldout` (716 images, 412
patients, patient- and lesion-disjoint). `pad_full` includes the 1,582 images the
adversarial branch saw in training and appears only where labelled.

**Floors:** leakage is balanced accuracy, floor 0.5 (2 domains). Cross-domain accuracy
is balanced over 6 shared classes, floor 0.167. Plain-accuracy majority on PAD is 0.385.

---

## A. The intervention

| # | Claim | Evidence | Source |
|---|---|---|---|
| A1 | A seven-point sweep of $\lambda_{adv}$ with everything else fixed, 3–5 seeds, 27 runs | $\lambda \in \{0, .25, .5, 1, 2, 4, 8\}$; n=5 at 0, 0.25, 2, 8; n=3 at 0.5, 1, 4 | `phase3_sweep/*/summary.json` |
| A2 | Leakage falls to its chance floor at the smallest non-zero weight | $0.915 \to 0.553$ (floor 0.5) | same |
| A3 | In-distribution balanced accuracy does **not** degrade | $0.692 \to 0.707$; observed range 0.679–0.707 | same |
| A4 | In-distribution calibration is unchanged | ECE flat at $\approx 0.10$ across all $\lambda$ | `phase13/per_run/*.json` |

**Do not claim** A3 as an improvement. The difference is small relative to seed variance.

## B. The collapse and the inversion — keep these separate

| # | Claim | Evidence | Source |
|---|---|---|---|
| B1 | Detection falls to chance in one step | $0.843 \to 0.509$ between $\lambda=0$ and $\lambda=0.25$ | `phase16/table1_ood_pad_dual_column.json` |
| B2 | The sub-chance region is reached only at $\lambda \geq 0.5$ | 0.465, 0.455, **0.427**, 0.455, 0.494 at $\lambda$ = .5, 1, 2, 4, 8 | same |
| B3 | At $\lambda=2$ the inversion is significant | **0.427, 95% CI [0.414, 0.439]**, 5/5 seeds individually below 0.5 | `reviewer_r1/bootstrap_results.json` |
| B4 | $k$-NN agrees | 0.440 [0.429, 0.452], 5/5 | same |
| B5 | Cosine agrees, one seed short | 0.453 [0.441, 0.464], **4/5** | same |
| B6 | All five rules move together | $\lambda=0 \to 2$: MSP .970→.437, Energy .983→.441, cosine .901→.435, kNN .941→.424, Maha .843→.427 | `phase3_sweep` |
| B7 | It is a bulk shift, not a tail effect | Truncating pooled p90/p95/p99 leaves 0.412/0.416/0.424 vs 0.427; medians invert (ID 13.8, OOD 10.0) | `phase14/tails/` |

**Caveat for B6:** the five rules are all functions of the same 16-d latent. Say so.
Independent replication is D1 and F2.

## C. Robustness of the sub-chance result

| # | Claim | Evidence | Source |
|---|---|---|---|
| C1 | Not a covariance artefact | Ledoit–Wolf leaves 0.427 unchanged on `pad_heldout` | `cleanup/ledoit_pad_heldout.json` |
| C2 | Not an implementation artefact | max $|\Delta$AUROC$| = 9.4\times10^{-4}$ across float32/64, shrinkage, 1/4/8 BLAS threads | `reviewer_r1/maha_stability.json` |
| C3 | Not an artefact of adversary strength | ×10: **0.418 ± 0.011**, 3/3 below 0.5. ×30: 0.432 ± 0.024. ×100: 0.482 ± 0.020 (seed 52 at 0.503) | `reviewer_r1/ITEM3_LAM2_ADVLR_REPORT.md` |
| C4 | Not confined to the adversarial domain | Fitzpatrick17k $0.764 \to 0.391$, CI [0.386, 0.396], 5/5 below 0.5 | `bootstrap_results.json` |

**C4 caveat:** Fitzpatrick has no patient IDs; bootstrap is over images and the interval
is anti-conservative. **C3 note:** weakening the adversary *strengthens* the inversion.

## D. Leakage does not mediate the collapse

| # | Claim | Evidence | Source |
|---|---|---|---|
| D1 | Backbone: leakage flat, detection falls | leak $0.982 \to 0.941$; Maha $0.749 \to 0.475$ | `phase13/` depths |
| D2 | Leakage saturates while AUROC keeps falling | leak floor reached at $\lambda=0.25$; AUROC 0.509 → 0.427 beyond it | `phase3_sweep` |
| D3 | **Strongest:** collapse with almost no leakage reduction | ResNet-50, $\lambda=0.5$, gate **valid**: leak $0.965 \to 0.939$ (0.026) while kNN $0.962 \to 0.563$ and cosine $0.922 \to 0.543$ | `reviewer_r1/ITEM5_RESNET50_REPORT.md` |
| D4 | Two objectives, same leakage range, opposite outcome | see E | — |

D3 is a valid single experiment in which the outcome moves 0.40 while the supposed
mediator moves 0.026. This is the cleanest form of the argument.

## E. The failure belongs to adversarial optimisation

| # | Claim | Evidence | Source |
|---|---|---|---|
| E1 | GRL buys invariance at no accuracy cost and collapses everything | leak $0.970 \to 0.763$, ID acc $0.845 \to 0.845$, all five detectors to chance | `phase15/` single-encoder |
| E2 | MMD moves leakage the same way and collapses nothing | $0.967 \to 0.900$, ID bal $0.779 \to 0.730$, Maha $0.813 \to 0.716$ | `phase15b/` |
| E3 | MMD can only go further by destroying the task | w=10000: leak 0.836, ID bal **0.337**, Maha 0.665 | same |
| E4 | IRM reaches the extreme of the same pattern | w=100: leak 0.950, ID bal **0.167** | same |
| E5 | CORAL does not induce invariance at all | w=100: leak **0.998**, Maha 0.999 | same |
| E6 | GroupDRO leaves leakage unmoved | 0.976 at every weight | same |
| E7 | Opposite calibration signatures | MMD OOD ECE $0.344 \to 0.142$ (less confident as it degrades); GRL $0.247 \to 0.746$ | same + `phase13/` |

**Bound on E:** MMD never reached GRL's invariance level. Claim only that at the levels
MMD reaches, GRL has already inverted and MMD has not.
**Supervision split:** IRM and GroupDRO use PAD labels; ERM, CORAL, MMD, GRL do not.
Their cross-domain numbers (0.60–0.64) are label access, not generalisation.

## F. Architecture and backbone

| # | Claim | Evidence | Source |
|---|---|---|---|
| F1 | Not specific to the factorized architecture | single-encoder DANN: kNN .944→.528, Energy .911→.507, cosine .884→.509, MSP .868→.515 at $\lambda=0.25$; 4/5 below chance by $\lambda=4$ | `phase15/` |
| F2 | ID accuracy flat there too | $0.845 \to 0.832$ | same |
| F3 | The dual encoder reaches deeper invariance | leak 0.553 vs single-encoder plateau 0.76–0.81; context branch probe $>0.9999$ | `phase1/`, `phase15/` |
| F4 | **Backbone question is open, not negative** | ResNet-50 at the only valid $\lambda$ (0.5) reached leak 0.939 vs EffNet-B3's 0.742 on the same partition; Maha 0.814, not inverted | `reviewer_r1/ITEM5_RESNET50_*.md` |

**F4 is the one to write carefully.** At $\lambda=2$ the ResNet-50 adversary never
learned (best-epoch CE 0.686–0.688 against a 0.673 threshold); those three runs are
**null experiments and carry no evidence either way**. Comparable invariance was never
reached on ResNet-50, so this is *untested*, not *refuted*.

**Partition trap:** EffNet-B3's familiar 0.970 leakage is `pad_full`. On `pad_heldout`
it is 0.952 → 0.742 → 0.760. Compare like with like or state the partition.

## G. Scope — same-modality site shift

| # | Claim | Evidence | Source |
|---|---|---|---|
| G1 | ISIC contains a same-modality two-site contrast | BCN 12,413 / HAM 10,015 via `lesion_id` prefix | `ISIC_2019_Training_Metadata.csv` |
| G2 | The sites are separable and not trivially so | probe 0.962; vignette-masked 0.943; hand-crafted max 0.901 (vs 0.876 on ISIC↔PAD); label-only reference 0.668 | `phase16/b0/` |
| G3 | The collapse reproduces | leak $0.963 \to 0.821$, kNN $0.658 \to 0.486$, cosine $0.576 \to 0.477$, 3 seeds | `phase16/b1/dense3seed_aggregate.json` |
| G4 | The inversion does **not** | reverse direction: baseline Maha 0.846 (matching 0.843), falls only to **0.683, CI [0.666, 0.699]**, 3/3 seeds entirely **above** 0.5 | `bootstrap_phase16_ham_bcn_ladv1.json` |
| G5 | Neither headroom nor invariance explains the gap | same baseline (0.846 vs 0.843), same achieved invariance (0.77 vs 0.763), different endpoint (0.683 vs 0.427) | — |

## H. Consequences

| # | Claim | Evidence | Source |
|---|---|---|---|
| H1 | Confidence on unseen domains overtakes in-distribution confidence | ID MSP 0.915 flat; PAD $0.489 \to 0.930$; crossing by $\lambda \approx 0.5$ | `phase13/` |
| H2 | Out-of-domain calibration triples while ID calibration is flat | OOD ECE $0.247 \to 0.746$; ID ECE $\approx 0.10$ | same |
| H3 | Cross-domain accuracy does not improve at any $\lambda$ | bal acc 0.249–0.291 (floor 0.167), none above $\lambda=0$'s 0.291, all below the control's 0.298; plain 0.245→0.184, under the 0.385 majority | `phase6_xfer/` |
| H4 | Predictions collapse toward the benign majority class | nevus predictions 113 → 426 of 716; **AK recall 49/202 → 3/202** | same |
| H5 | Discrimination survives; the decision rule does not | macro AUC $0.605 \to 0.581$ | same |

**H2 is not a new phenomenon** — Wang et al. (NeurIPS 2020) established that adaptation
trades calibration on the target. New here: dose-dependence, invisibility to ID
calibration, and that it is separate from the detection collapse.
**H4 limits:** SCC $6/66 \to 1/66$ is directionally consistent but too small to carry a
claim; melanoma $n=9$ supports none.
**H1/H2/H3 are one phenomenon, three views** — OOD ECE $\approx$ confidence − accuracy
here. Do not present them as independent corroboration.

## I. What it is not — six refuted accounts

| # | Account | Test | Result |
|---|---|---|---|
| I1 | Memorisation of the adversarial images | hold PAD out at patient level | 0.427 vs 0.407, unchanged |
| I2 | Class composition | restrict to shared classes; reweight to OOD mix | 0.432 vs 0.427; reweighting **strengthens** to 0.240 |
| I3 | Domain specificity | Fitzpatrick17k, never in $\mathcal{L}_{adv}$ | 0.391 |
| I4 | Dimensional collapse | participation ratio across $\lambda$ | $4.879 \to 4.500$ while leakage $0.914 \to 0.553$ |
| I5 | Latent compression | latent variance and centroid distance | OOD already 4.5× tighter at $\lambda=0$; the adversary *decompresses* it |
| I6 | Tail asymmetry | truncate the pooled upper tail | 0.412–0.424 vs 0.427; medians invert |
| I7 | Synthetic reproduction | pre-registered toy model, heavy-tailed ID, grid fixed first | no inversion ($\lambda=0$: 0.693, $\lambda=2$: 0.786) |

I7 is **not** a seventh refuted explanation. Six accounts, plus a failed reproduction.

## J. Secondary

| # | Claim | Evidence |
|---|---|---|
| J1 | A domain monitor is cheap from any representation | frozen ImageNet ResNet-50 + linear head: **0.998**; trained baseline 0.997; EffB3 control 0.991 |
| J2 | What distinguishes the context branch is concentration, not detection | $k{=}1$ AUROC 1.00 at 84.5% variance, vs 0.55–0.70 at $k{=}1$ elsewhere |
| J3 | Semantic novelty is unaffected in the same way | 4a: lesion 0.648 / context 0.641 / baseline 0.687. 4b: 0.806 / 0.607 / 0.684. No near/far structure (Pearson $-0.12$) — a null result |

## K. Limitations the paper must state

1. Absolute cross-domain performance is poor for every model (0.245 vs 0.385 majority). No model here is deployable; the claim is about the effect of $\lambda$.
2. The inversion is observed only across imaging modality. The same-modality contrast (G) separates it from the collapse.
3. Backbone: comparable invariance was not reached on ResNet-50; untested, not refuted. Three $\lambda=2$ runs are null experiments.
4. Two external replications were inconclusive — Camelyon17 (adversary never trained), iWildCam (trained but encoder never became invariant; leakage 0.911 → 0.891 across $\lambda \in [0,10]$).
5. Invariance appears harder to achieve as training domains multiply. An observation resting on two negatives, not a claim.
6. The five detectors are not independent; D1 and D3 carry the independent evidence.
7. OOD ECE $\approx$ confidence − accuracy here.
8. ISIC splits are image-level. The comparison is across $\lambda$ under one split, so inflation applies to every row.
9. Fitzpatrick17k: 3,887 of 16,577 images, web-scraped, mixed copyright, evaluation only.
10. 5 seeds at endpoints, 3 at interior points. No formal testing across seven points at n=3.
11. No mechanism. Six accounts refuted and a pre-registered reproduction failed.

## L. Recommendation

Report OOD detection and out-of-domain calibration as a function of **achieved
invariance**, not at one operating point; report cross-domain accuracy beside any claim
of tolerance to covariate shift; state when a separate dataset serves as the OOD set
that this measures covariate shift rather than semantic novelty.

---

## Figures available

`results/paperB/figures/` — fig1 intervention (4 panels), fig2 confidence, fig3
detectors, fig4 generalises, fig5 leakage no mediation, fig6 site shift; S1 semantic
OOD, S2 domain axis, S3 superseded preview. PDF + PNG at 300 dpi, grayscale checks
included.

## Still open

- `\author` and affiliations
- Generative-AI disclosure statement, per the target journal's policy
- Figure 1: the shaded-span annotation overlaps panel (b)'s x-axis label
