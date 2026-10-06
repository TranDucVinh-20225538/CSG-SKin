# Phase 14 — tails, saturation, and a corrected causal claim

Numbers as measured. No retuning. Existing Phase 3 / 12 / 13 files were not overwritten except for the Item D language corrections listed in `CAUSAL_CLAIM_AUDIT.md`.

**Lead result (unchanged, 13.3).** OOD ECE on pad_heldout rises **0.247 → 0.746** while ID ECE stays flat at **~0.10**. Mean softmax confidence on never-seen-domain data overtakes ID by λ=2. That is the paper’s headline from here on.

---

## Item D — corrected causal claim (no compute)

The paper must not claim that reducing leakage causes the OOD collapse.

| Evidence | What it shows |
|---|---|
| Phase 3, λ=0.25 → 2 | Leakage **rises** 0.553 → 0.584 while AUROC **falls** 0.508 → 0.413 |
| Phase 13.2 backbone | Leakage flat 0.982 → 0.941; backbone Maha falls 0.749 → 0.475 |

**Claim.** Adversarial training causes both the leakage drop and the OOD collapse. The OOD collapse is not mediated by linear domain-decodability. It follows from geometric changes a linear domain probe does not capture.

Audit: `phase14/CAUSAL_CLAIM_AUDIT.md`. The mediation sentence in `MASTER_REPORT.md` and `PHASE12_1B_AUDIT.md` (“OOD gate degrades as a function of achieved invariance”) is deleted. `fig_dial` stays vs λ. The n=4 preview is doubly superseded (heterogeneous-model correlation, and it is an AUROC-vs-leakage plot). Leakage remains a measured column in Table 1. 13.2 is promoted as its own result.

---

## Item A — is the inversion a tail effect?

Cached Phase 13 features. Maha: class-conditional shared covariance, `reg_eps=1e-3`, ISIC-train fit. Primary space `z_lesion_norm`; A1–A2 also at the backbone.

### A2 is decisive. The inversion is a bulk location shift, not a tail.

On pad_heldout, `z_lesion_norm`, λ=2:

| | full | drop p90 | drop p95 | drop p99 |
|---|---|---|---|---|
| AUROC | **0.427 ± 0.025** | **0.412 ± 0.025** | 0.416 ± 0.022 | 0.424 ± 0.023 |

Inversion **persists in the bulk** and slightly deepens when the pooled upper 10% is removed. Same pattern on Fitzpatrick17k (0.391 → 0.412 after p90; still inverted). Backbone pad_heldout at λ=2: 0.489 → 0.433 after p90.

**Statement.** OOD samples are ranked as more typical than ID samples across the whole score distribution, not only at the extremes. The tail is not the driver. This line closes.

### A1 — the predicted tail asymmetry is not there

Prediction: medians converge as λ rises while ID p95/p99 stay above OOD.

`z_lesion_norm`, ID vs pad_heldout:

| λ | ID p50 | hold p50 | ID p95 | hold p95 | ID p99 | hold p99 |
|---:|---|---|---|---|---|---|
| 0 | 13.6 | **40.2** | 54.4 | 56.8 | **83.1** | 64.5 |
| 2 | **13.8** | 10.0 | 53.7 | 54.8 | 79.3 | 80.1 |

At λ=0 the OOD *median* is far above ID (normal detection). By λ=2 the **bulk has inverted**: hold median 10.0 sits below ID 13.8. The upper tails have matched, not diverged. ID does not retain a heavier p95/p99. The inversion is a location shift of the whole OOD score distribution, not a clipped OOD tail.

That is the same picture as 13.4: the adversary matches distributions. It does not compress OOD into a core while leaving an ID tail.

### A3 — the ID top 1% is mixed

λ=2, top 1% of ID Maha (k ≈ 51 per seed):

- Accuracy 0.592 vs 0.811 overall. Confidence 0.749 vs 0.910.
- DF+VASC are 26% of the tail vs 2% base rate. They do not dominate.
- Incorrect common-class 38%. Correct common-class 35%. Correct rare 24%.

Phase 2.5b already showed class restriction does not fix the inversion (0.414 vs 0.409). A rare-class tail cannot be the mechanism. No single identity (rare class / error / correct outlier) owns the tail. Report the breakdown; do not force one.

Full tables and figures: `phase14/tails/PHASE14_A_TAILS.md`.

---

## Item B — Phase 12 is inconclusive. No Camelyon result is reported.

The adversary was never trained. `train_adv_acc` sits at 3-class chance and CE equals ln(3) at every epoch, including epoch 1, at every λ. A head that is learning starts near ln(3) and improves. Sitting exactly at ln(3) forever means it never extracted anything. The post-hoc probe still reads 0.94: the hospital signal is in the representation; the online head did not use it. ||g_enc through GRL|| = 0 is a symptom of a constant uniform head, not a conclusion.

**GRL sign check (30 minutes, then stop).** Head cosine vs no-GRL CE = **+1.000**. Feature-stream cosine = **−1.000**. GRL flips only the encoder path. It is not a minus-sign on the head. First-pass Dropout made this look unclear; eval-mode, shared-z check is decisive. WILDS does not reopen. B2 is not run.

**Correct statement.** Phase 12 **cannot conclude**. Outcome (c) in the work-order sense was “remaining implementation difference,” not “Camelyon17 resists invariance.” Writing the latter would not survive a reviewer who asks whether the adversary learned. It did not.

No Camelyon17 number enters Results. Limitations: tried, implementation unresolved (online adversary never left ln(3); GRL graph itself is standard), left for later work. iWildCam stays gated.

`phase14/camelyon_b1/PHASE14_B_GRL_SIGN.md`, `PHASE14_B1.md`.

---

## Item C — one pre-registered toy revision

`PREREGISTER_v2.json` was written before any v2 fit. Locked MLP, BN **off** (primary), ID-only heavy tails (class-dependent τ log-uniform [0.3, 3], two rare classes n=80, 3% contamination at τ=8). BN-on reported separately. Not retuned. No third revision.

| Criterion | Primary (BN off) |
|---|---|
| C1 λ=0 ≥ 0.5 | **yes — 0.693 ± 0.027** |
| C2 sub-chance at λ≥0.25 | **no — λ=0.25 0.699, λ=2 0.786** |
| C3 detectors together | no |
| C5 reweight strengthens | no (0.784 vs 0.786) |
| C7 third domain inverts | no (0.676) |

C1 is the legal start v1 never had. From that start, AUROC **rises** with λ. The opposite of the derm cliff. BN-on control also starts legal (0.599) and does not invert (λ=2: 0.675).

**The mechanism question closes.** The paper reports a robust phenomenon with six refuted explanations — image memorisation, class composition, domain specificity, dimensional collapse, latent compression, and an ID-tail / score-tail account — plus an explicit statement that a controlled synthetic reproduction failed under a pre-registered grid (v1: BN artifact; v2: legal start, no inversion). That is a publishable position.

`phase13_5/PREREGISTER_v2.json`, `phase13_5/v2/PHASE13_5_V2_REPORT.md`.

---

## What the paper now says

1. **Lead.** Adversarial training makes the model more confident, and worse calibrated, on data it has never seen (OOD ECE 0.247 → 0.746; ID ECE ~0.10).
2. **Cliff.** Maha AUROC 0.86 → 0.51 at λ=0.25, then below chance. No safe operating point.
3. **Not mediated by leakage.** λ causes both. Linear domain-decodability can stay high while OOD detection collapses (13.2). Plot vs λ, never vs leakage.
4. **Mechanism open, six accounts closed.** The inversion is a bulk location shift of the OOD score distribution, not a tail, not compression, not dimensional collapse, not class mix, not memorisation, not PAD-specific, and not reproduced by a locked linear-Gaussian or heavy-tailed toy.
5. **Camelyon.** Not a result. The online adversary never trained. Phase 12 is Limitations, not Results.

## Files

- D: `phase14/CAUSAL_CLAIM_AUDIT.md`; corrections in `MASTER_REPORT.md`, `phase12/PHASE12_1B_AUDIT.md`, `phase10/table3.md`, `phase13/PHASE13_REPORT.md`, `scripts/aggregate_final_package.py`
- A: `phase14/tails/`
- B: `phase14/camelyon_b1/`; note on `phase12/PHASE12_1B_REPORT.md`
- C: `phase13_5/PREREGISTER_v2.json`, `phase13_5/v2/`
