# WORK ORDER — Phase 12: does the cliff exist outside dermatology?

Paper B's derm result is locked. This phase tests whether the central phenomenon is a property of **domain-adversarial invariance training** or a property of the ISIC↔PAD setup.

**Timebox: 3 weeks.** If 12.1 has not produced a verdict by then, stop, write it up as future work, and ship the dermatology paper to MedIA. Do not let this phase hold the paper hostage.

---

## What is being tested

The derm finding, restated in dataset-independent terms:

> As the domain-adversarial weight λ_adv rises from zero, covariate-shift OOD detection collapses from strong to chance-or-below, almost entirely within the first small nonzero value — while the domain leakage probe improves and in-distribution accuracy does not degrade. Every monitored metric improves while the OOD gate fails.

Two weaknesses in the derm study that this phase exists to patch:

1. **Two domains, one modality contrast.** ISIC is dermoscopy, PAD and Fitzpatrick17k are clinical photographs. A reviewer can argue the "domain" construct is really modality. Dermatology data cannot answer this.
2. **Generality asserted, not shown.** One task, one modality family, one architecture family.

## Predictions (report PASS/FAIL, as in every prior phase)

| # | Prediction |
|---|---|
| P12.1 | On Camelyon17, OOD AUROC collapses from high at λ_adv=0 to ≈0.5 or below at the smallest nonzero λ tested |
| P12.2 | ID accuracy does not degrade across the collapse region |
| P12.3 | The collapse co-occurs with the drop in domain-probe accuracy |
| P12.4 | On iWildCam, the cliff appears with **many** domains, not just two — removing the modality objection entirely |
| P12.5 | Cross-domain (OOD-split) accuracy does not improve enough to justify the loss of OOD detection |

A FAIL is a finding. If the cliff does not replicate, that is itself important: it would mean the effect depends on something specific to the derm setup, and the paper's claim must narrow accordingly. **Report it; do not retune toward a prediction.** This rule has held for every phase and holds here.

---

## Phase 12.0 — Dataset discovery (blocking gate, no compute)

The datasets may already be on this cluster. Check before downloading anything.

1. Search shared data locations for existing WILDS data: `camelyon17_v1.0`, `iwildcam_v2.0`. Look under `/data2/`, any shared dataset roots, scratch, and group directories. Check `module avail` and any site dataset registry.
2. Check whether the `wilds` Python package is installed in the project venv; if not, install it (`pip install wilds`).
3. If not present locally, get sizes before downloading: Camelyon17 ≈ 10 GB, iWildCam ≈ 90 GB. **Confirm quota before starting a 90 GB download.**
4. Record licences and citations alongside the existing ISIC / PAD / Fitzpatrick17k entries: WILDS itself, Camelyon17 (CAMELYON17 challenge terms), iWildCam (its own terms).
5. Verify the official WILDS splits load correctly and report the split sizes and domain counts you actually get.

Report to `results/paperB/phase12/PHASE12_0_DISCOVERY.md` and **stop for confirmation before downloading anything over 10 GB.**

---

## Phase 12.1 — Camelyon17 (primary, gates everything else)

Closest analogue to the derm setup: medical, but a completely different modality (histopathology), with hospital as the domain and stain variation as the shift.

### Setup

- **Architecture: standard single-encoder DANN**, not the dual-encoder. The claim under test is about invariance training in general, so plain DANN is both the cleaner test and the easier result to accept. Use the WILDS reference backbone (DenseNet-121) so numbers sit alongside the public leaderboard.
- **Domain head:** multi-class over the training hospitals, with GRL, exactly as in the derm setup.
- **Splits:** official WILDS splits. ID = `id_val`/`id_test` (training hospitals); OOD = the held-out hospital test split. Do not invent splits.
- **Seeds:** 3 per λ point, 5 at the endpoints and at the located transition.

### Locate the cliff before sampling it densely

λ values do **not** transfer across setups — loss magnitudes differ. Do not assume the transition sits in {0, 0.25, …, 8}.

1. **Coarse scan, 1 seed:** λ_adv ∈ {0, 0.1, 1, 10}. Find the interval where OOD AUROC falls.
2. **Dense sampling, 3 seeds:** at least 5 points inside and around that interval, including λ=0 and one value well above the transition.
3. If the collapse is already complete at the smallest nonzero λ in the coarse scan, extend **downward** (0.05, 0.01, 0.001) until you find where it begins — that boundary is the result, and a collapse at λ=0.001 would be a stronger finding than one at λ=0.25.

### Measure per λ point

Identical protocol to the derm study:

- **Detectors:** MSP, Energy, cosine-to-nearest-centroid, Mahalanobis (class-conditional, shared covariance), kNN (k=50). All statistics fit on the **training split only** — assert this in code, as before.
- **Domain leakage probe:** logistic probe on frozen features predicting hospital; report **balanced accuracy and AUROC** against the majority baseline.
- **ID accuracy** on the in-distribution split.
- **Cross-domain accuracy** on the OOD split. WILDS's headline metric is exactly this, which answers "what does the invariance buy you?" natively on an established benchmark.
- ECE.

**Output:** the same four-column table as the derm paper — leakage, ID accuracy, OOD AUROC, cross-domain accuracy — per λ. → `results/paperB/phase12/camelyon17/`

**Gate: if P12.1 fails, stop and report before touching iWildCam.**

---

## Phase 12.2 — iWildCam (gated on 12.1; the many-domain result)

Non-medical, and — the reason it is here — **hundreds of domains rather than two.** This is what finally retires the "your domains are just two modalities" objection, which no dermatology dataset can answer.

- Backbone: WILDS reference ResNet-50.
- Domains: camera trap locations. Training has ~240 locations; OOD splits use locations never seen.
- **Domain head with ~240 classes may destabilise GRL.** Try the full multi-class head first. If training is unstable, cluster locations into ~10–20 groups and use those as domain labels. **Report whichever you used and why** — it is a design decision reviewers will ask about, not an implementation detail to bury.
- Same λ-location procedure as 12.1: coarse scan, then dense sampling around the transition.
- Same measurements. Seeds may be reduced to 3 throughout given the larger images; state the reduction.

**Additional analysis unique to this dataset:** does the cliff depend on the number of domains? Retrain with domain subsets of size {2, 5, 20, all}, at λ=0 and at one post-cliff λ. If the cliff appears even at 2 domains and persists at 240, the phenomenon is about invariance itself, not about how many or what kind of domains exist.

→ `results/paperB/phase12/iwildcam/`

---

## Phase 12.3 — Optional: does the monitor survive here too?

Only if 12.1 and 12.2 both replicate and time remains.

On Camelyon17 only, train the dual-encoder variant (lesion/context analogue: a task branch with the adversarial constraint and a context branch supervised on domain) across the same λ grid. Test whether the context branch retains OOD detection at every λ, as `z_context` did in derm.

This would extend the architectural claim beyond dermatology. It is secondary — the phenomenon matters more than the proposed mitigation, and **a domain monitor is cheap from any representation** (frozen ImageNet ResNet-50 reached 0.998 in Phase 1.6), so do not over-invest here.

---

## Phase 12.4 — Integration

Produce `results/paperB/phase12/PHASE12_REPORT.md` with:

- The PASS/FAIL table for P12.1–P12.5
- A three-panel cliff figure per dataset, matching the derm Figure 1 layout exactly so the three can be read side by side
- A combined figure: normalised λ on the x-axis, OOD AUROC on the y-axis, one line per dataset (derm / Camelyon17 / iWildCam). **If all three show the same cliff shape, this becomes the paper's new Figure 1** and the dermatology result becomes one instance of a general phenomenon
- The many-domain result from 12.2, stated as the answer to the modality objection
- Cross-domain accuracy across λ for all three datasets — the cost-benefit answer on established benchmarks

### Consequence for positioning

- **All three replicate cleanly** → the paper is about invariance training, not dermatology. Reconsider venue then, not before. NeurIPS/ICLR become viable; MedIA remains a strong choice with a much stronger paper.
- **Camelyon17 replicates, iWildCam does not** → the effect may be medical-imaging-specific. Interesting, and the claim narrows to that.
- **Neither replicates** → the derm finding stands alone, the claim narrows to the ISIC↔PAD setting, and this phase is reported as a negative result in the limitations. Ship to MedIA.

**Decide the venue after seeing these numbers, not before.**

---

## Execution rules (unchanged)

1. 12.0 and 12.1 are gates. Stop and report at each.
2. Nothing under `results/paperB/` from prior phases may be modified; new work goes under `results/paperB/phase12/`.
3. Long runs go to SLURM as array jobs with resume.
4. Every script takes `--dry_run`.
5. Report failures as findings. Four predictions have already failed in this project and were reported as such; that record is why the result is credible. Do not break it here.
6. Keep the reporting conventions: balanced accuracy for probes, bootstrap CIs, effect sizes over p-values, and never a bare `1.0000`.
