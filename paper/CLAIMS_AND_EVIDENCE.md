# Claims and evidence

Every proposition the paper is entitled to make, with the number behind it and where
that number lives. No prose — write from this, not from the existing draft.

**Status (R2 closed).** Sections M and N are new and carry the two largest changes: a
mechanism now exists, and there is a partial mitigation. Section I is demoted from a
result to the support for M. Every number here is checked by
`scripts/verify_manuscript_numbers.py`; the load-bearing ones are declared with their
source key in `paper/number_manifest.json`, which is the only mode that catches a value
attached to the wrong dataset.

**Pending Item 4.** The lesion-level sweep (15 runs) replaces Table 1. Sections A, B and
C hold the image-level values and must be re-read against the new table when it lands;
everything else is independent of it.

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
| A3 | In-distribution balanced accuracy does not degrade **under the image-level split** | $0.692 \to 0.707$; observed range 0.679–0.707 | same |
| A3b | Under a lesion-disjoint split it **also does not** degrade, at $n=5$ | paired per-seed difference $-0.023$, 95% CI $[-0.063, +0.017]$, contains zero. Report as a trend, claim nothing. (At $n=3$ it read $-0.040$; seeds 72 and 82 halved it) | `r3/b1/` |
| A3d | The split artefact is $0.20$, roughly ten times any $\lambda$ effect | $0.496 \to 0.692$ at $\lambda=0$ | `r3/b1/`, `r2/item4/` |
| A3e | The lesion-level inversion carries its own interval | Maha `pad_heldout` $\lambda=2$ $= 0.403$, CI $[0.380, 0.425]$, **5/5 seeds individually below 0.5**; Fitzpatrick $0.402$ $[0.390, 0.415]$; kNN 0.427, cosine 0.420 | `r3/a2/` |
| A3c | 60.0% of the image-level ISIC test split shares a `lesion_id` with train or val | 3,041 of 5,067; lesion-level overlap 0/0/0 | `r2/item4/REPORT.md`, `r2/item4/split_record.json` |
| A4 | In-distribution calibration is unchanged | ECE flat at $\approx 0.10$ across all $\lambda$ | `phase13/per_run/*.json` |

**Do not claim** A3 as an improvement. The difference is small relative to seed variance.
**The invisibility claim is restored**, at $n=5$ and under both splits. An earlier
draft, written at $n=3$, said accuracy falls 0.04 under the honest split; seeds 72 and
82 took the paired difference to $-0.023$ with a CI containing zero. The class mix is
not the explanation for the level difference either: reweighting the lesion-level test
set to the image-level mix leaves every detector identical to three decimals.

**Open integrity item in Table 1 (image-level).** The $\lambda = 0.25$ row declared
$n = 5$ while leakage, ID accuracy and Fitzpatrick AUROC were three-seed means. At five
seeds they are 0.571, 0.692 and 0.460, and the abstract's "$0.915 \to 0.553$" was
likewise a three-seed figure, now 0.571. The row's standard deviations and its OOD
AUROC cell are unverified and marked `\pending{R4}`; the $\lambda = 1$ row still uses
three of five available seeds. Note separately that Phase 13's $0.914 \to 0.553$ is its
own three runs and is now labelled as such rather than silently mixed with the
five-seed scale.

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
| H4 | Predictions collapse toward the benign majority class | nevus predictions 113 → 374 of 716; **AK recall 49/202 → 3/202** | same |
| H5 | Discrimination survives; the decision rule does not | macro AUC $0.605 \to 0.581$ | same |

**H2 is not a new phenomenon** — Wang et al. (NeurIPS 2020) established that adaptation
trades calibration on the target. New here: dose-dependence, invisibility to ID
calibration, and that it is separate from the detection collapse.
**H4 limits:** SCC $6/66 \to 1/66$ is directionally consistent but too small to carry a
claim; melanoma $n=9$ supports none.
**H1/H2/H3 are one phenomenon, three views** — OOD ECE $\approx$ confidence − accuracy
here. Do not present them as independent corroboration.

## M. The mechanism — the inversion runs through the majority-class centroid

Three criteria fixed in `r2/PRECOMMIT_ITEM1_ITEM2.json` before the test. All three hold.
Source for all rows: `r2/item2/item2_results.json`, report `r2/item2/REPORT.md`.

| # | Claim | Evidence |
|---|---|---|
| M1 | Out-of-domain images migrate into the nevus cell; in-distribution images do not | NV-nearest fraction (OOD) $0.332 \to 0.601$, $\Delta = 0.268$ vs $2\mathrm{SE} = 0.055$; same fraction for ID flat $0.528 \to 0.531$ |
| M2 | It tracks the diagnostic shift exactly | Spearman $\rho = 0.982$ ($p = 1.2\mathrm{e}{-19}$) between NV-nearest fraction and predicted-nevus count, across all 27 runs; count $105 \to 424$ of 716 |
| M3 | The inversion is confined to those images | AUROC NV-nearest $0.807 \to 0.282$; non-NV-nearest stays above chance $0.861 \to 0.644$ |
| M4 | They move and the class-matched reference does not | median squared Mahalanobis OOD$\to$NV $49.8 \to 17.5$; ID-NV$\to$NV $7.5 \to 7.4$; ratio $6.63 \to 2.37$ |
| M5 | It retrodicts I2, previously unexplained | PAD is 10.6% nevus against ISIC's 50.8%; thinning NV from the ID reference should strengthen the inversion, and it does — to 0.240 |
| M6 | Not a property of the 16-d projection | replicates on the 1536-d pre-projection backbone: NV-nearest fraction $0.484 \to 0.603$, NV-nearest AUROC $0.726 \to 0.330$ |

**Scope limit that must be stated with M.** The ratio in M4 stays **above 1** at every
weight. Out-of-domain images become more typical than ID images of every class *other
than the majority one*, and because that class is half the data the pooled score crosses
chance. This is also why pooled and per-class AUROC disagree at $\lambda = 0$ (N4).
| M7 | Pooled AUROC **is** the class-share-weighted mean of the per-class AUROCs | An **identity**, not a result: AUROC averages over the ID samples. Do not cite it as evidence. Computed 0.424 vs measured 0.427 at $\lambda=2$; 0.835 vs 0.843 at $\lambda=0$; the residual is the DF+VASC weight (0.019), absent from the per-class table (R3 A1 closes it) |
| M8 | **The content is the two per-class values** | 0.559 against ID-NV — OOD is close to indistinguishable from nevi, whose centroid it is drawn toward — and 0.296 weighted over the other five, which is inverted. This is the mechanism; M7 only propagates it |
| M9 | For the pooled score, with model and OOD set fixed, the inversion deepens as the ID set becomes **less** dominated | $0.559p + 0.296(1-p)$ returns above chance only at $p = 0.776$. **Scope: one metric, one fixed model.** Not a general claim about deployments |

**Two corrections recorded here so they are not reintroduced.**

1. An earlier draft said the effect should *weaken* where the ID set is less dominated.
   Backwards — M9 is increasing in $p$.
2. An earlier draft explained the gap between the predicted and measured reweighting
   result by saying reweighting shifts the per-class AUROCs. **Wrong, and deleted.**
   Reweighting the ID set changes only $w_c$: the model, every image's score and the OOD
   set are unchanged, so each $\mathrm{AUROC}_c$ is unchanged by construction. The
   prediction itself was the approximation — a two-group form with the non-NV classes
   lumped at *ISIC* proportions gives 0.3285 against a measured 0.240, while the full
   six-class identity with PAD held-out weights (NV 88, BCC 276, AK 202, BKL 75, SCC 66,
   MEL 9 of 716) gives **0.2585**, so the lumping accounts for 0.070 of the 0.0885.
   **R3 A1 and A4 closed both.** With all eight ID classes the identity holds to
   $2.2\times10^{-16}$ at every $\lambda$, both OOD sets and both ID sets; the
   six-class restriction is what leaves a $0.010$ discrepancy. And my $0.2585$ was
   itself a dataset mismatch — **the published $0.240$ is a `pad_full`
   measurement**, while $0.2585$ combined `pad_heldout` weights with `pad_heldout`
   per-class values. Under the exact protocol the identity reproduces the measured
   reweighted AUROC to $8\times10^{-7}$. The manuscript now reports the
   `pad_heldout` figure, $0.259 \pm 0.022$, so the table stays on one partition.
   Provenance for the record: the number is Phase 2.5b
   (`phase2_5/per_seed/runB_orth1_s*.json`, `pad_full`, n=5, exact `sample_weight`),
   not Item 2, and `manuscript_number_check.md` mis-sources it to `recall_BKL`.

## N. Is there a usable monitor? — partial, below the pre-registered bar

Pre-registered rule: Mahalanobis $\ge 0.90$ in all four cells (2 OOD sets $\times$ full /
lesion-disjoint ID). Source: `r2/item1/headline.json`, report `r2/item1/REPORT.md`.

| # | Claim | Evidence |
|---|---|---|
| N1 | A frozen ImageNet ResNet-50 never inverts | $\ge 0.83$ in all four cells |
| N2 | It beats the adversarially trained model by a wide margin | Fitzpatrick **0.988** vs 0.391; `pad_heldout` **0.833** vs 0.427 (both $\lambda = 2$) |
| N3 | **It misses the bar, and does not beat the model before training on PAD** | 0.833 < 0.90 on `pad_heldout`; against $\lambda = 0$ it is higher on Fitzpatrick (0.988 vs 0.764) but *not* on `pad_heldout` (0.833 vs 0.843, within one s.d.). Partial mitigation, not a fix. kNN on the same features gives 0.913 but **was not the pre-registered score** |
| N4 | Per-class AUROC, which does not depend on the ID class mix, inverts for every non-nevus class | `pad_heldout` 0.14–0.39; Fitzpatrick 0.10–0.34, at $\lambda = 2$ |
| N5 | For the two rarest classes the inversion is **deepened, not created** | at $\lambda = 0$ already AK 0.499, SCC 0.444 on `pad_heldout` |
| N6 | Detection degrades without invariance, outside dermatology | two-hospital Camelyon17, 3 seeds: five detectors move 0.12–0.22 (Maha $0.651 \to 0.534$, cosine $0.785 \to 0.589$, kNN $0.794 \to 0.648$, MSP $0.798 \to 0.589$, energy $0.794 \to 0.575$) while the hospital probe moves 0.007 ($0.959 \pm 0.007 \to 0.952 \pm 0.001$) and ID accuracy is flat at 0.995. Source `r2/item5/EXPLORATORY_REPORT.md` |
| N7 | Relative Mahalanobis is a uniform offset, not a repair | pad 0.427 $\to$ 0.477, Fitzpatrick 0.391 $\to$ 0.467, still below chance; it lifts $\lambda = 0$ too ($0.843 \to 0.898$), and applies to Mahalanobis only — kNN, MSP, energy and cosine have no analogue. Source `r2/rmd/REPORT.md` |

**N6 carries one claim only** — non-mediation (section D), where it is the cleanest
evidence in the paper. It is **not** evidence of breadth for the collapse or the
inversion: the detectors stay above chance (0.53–0.65), and these runs failed their own
pre-registered leakage gate (0.007 against a required 0.10). Say so in the text; a
reviewer will read the probe column.

**Ruled out without running:** per-class normalisation of the score. M4's ratios are
already below 1 for the minority classes, so normalising per class cannot repair them.

## I. Eliminated before the mechanism was found

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

**These are now the support for M, not a terminal result.** They are why M is credible:
it is what survived elimination rather than the first story tried. I2 is more than
eliminated — it is **retrodicted** by M (see M5).

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
4. Three external attempts, none supporting a breadth claim — Camelyon17 three-hospital (adversary never trained), iWildCam (trained but encoder never became invariant; leakage 0.911 → 0.891 across $\lambda \in [0,10]$), Camelyon17 two-hospital (adversary trained to chance, encoder still not invariant). **No inversion is observed outside dermatology.** A single-seed sub-chance reading (0.478) did not replicate: 0.534 ± 0.050 over three seeds.
5. ~~Invariance appears harder to achieve as training domains multiply.~~ **Falsified by our own data** (N6): two hospitals, verified gradient path, adversary at $\ln 2$, probe still 0.95. Domain count is not the barrier; the correct statement is that invariance was obtained in dermatology and not in histopathology at any weight or domain count tried, cause unknown. The pattern — adversary at chance while the information stays linearly decodable — is a known failure mode of adversarial removal (Elazar & Goldberg 2018).
6. The five detectors are not independent; D1 and D3 carry the independent evidence.
7. OOD ECE $\approx$ confidence − accuracy here.
8. ISIC splits in the main sweep are image-level, and 60.0% of the test set shares a lesion with training (A3c). This is reported as a measured quantity, not as a caveat, and the lesion-disjoint replication is in A3b.
9. Fitzpatrick17k: 3,887 of 16,577 images, web-scraped, mixed copyright, evaluation only.
10. 5 seeds at endpoints, 3 at interior points. No formal testing across seven points at n=3.
11. ~~No mechanism.~~ A mechanism exists (M). What remains open: why gradient reversal takes this route and moment matching does not, why the synthetic reproduction failed, and whether the majority class is special or merely the densest available.
12. The mechanism explains the **pooled** AUROC only. Out-of-domain images never become more typical than real nevi (M4). Do not write that they look like nevi.
13. Eight numbers in the paper trace only to a markdown report, not to JSON (phases 1.6 and 13). Listed on every run of the verification script.

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

- `\author` and affiliations — author only
- Generative-AI disclosure statement, per the venue's policy
- **Item 4, the lesion-level sweep (15 runs)**, which replaces Table 1. Sections A, B, C pending it
- **Body is 17 pages against a 14-page target.** One cutting pass after Item 4 lands, not before; confirm the limit applies to the Validation Studies track first
- Resolve the 105→424 (8-class argmax) vs 114→426 (6-class PAD argmax) labelling
- Extend `paper/number_manifest.json` beyond the current 15 entries as Table 1 settles
- Retire the eight markdown-sourced numbers by having phases 1.6 and 13 emit JSON
