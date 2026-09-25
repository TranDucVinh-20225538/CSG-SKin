# AMENDMENT — Phase 12.1b: the manipulation did not take

**Do not submit the proposed dense grid `{0, 0.1, 1, 2, 4, 10}`.** It would densely sample a range in which the independent variable never moved.

## Reading of the coarse scan

Leakage (3-class hospital probe) across the whole λ range: 0.977 → 0.969 → 0.970 → 0.963. The 3-class majority baseline is ≈ 0.33–0.40. **The adversary never removed hospital information at any λ tested.**

For comparison, derm at λ=0.25 moved leakage from 0.915 to 0.553 — close to its majority baseline of 0.688.

The derm claim is *"when invariance is achieved, covariate-shift OOD detection collapses."* This run never achieved invariance, so P12.1 and P12.3 are **not yet testable**. This is not a FAIL; it is an experiment that has not reached the regime under test.

**The λ=10 Mahalanobis drop (0.671 → 0.482) is not a replication.** The derm signature was every detector collapsing together — MSP, Energy, cosine and Mahalanobis all reached ~0.41. Here Mahalanobis moved while kNN stayed flat (0.830 → 0.816) and leakage stayed at 0.963. One detector moving while the others do not, with the representation still fully hospital-identifiable, indicates a covariance-conditioning artifact at high λ, not a representational change. Do not report it as the derm phenomenon.

**Confirmed so far:** P12.2 holds (ID acc 0.987 → 0.981). P12.5 holds (transfer acc 0.874 / 0.832 / 0.888 / 0.864 — flat and noisy).

---

## 12.1b — find where leakage actually drops

New objective for this round: **not to locate the cliff, but to locate the point where invariance is induced at all.** Leakage is the primary readout; OOD AUROC is secondary until leakage moves.

### Step 1 — audit the adversary against the derm implementation (no compute)

Compare the Camelyon17 DANN configuration line by line against `src/models/csg_lite.py` and `scripts/train_csg.py`:

| Item | Derm value | Camelyon17 value | Match? |
|---|---|---|---|
| `adv_lr_multiplier` | **30.0** | ? | |
| GRL λ schedule | constant vs DANN ramp `2/(1+exp(-10p))-1` | ? | |
| Adversary head architecture | Linear→BN→ReLU→Dropout(0.5)→Linear→BN→ReLU→Dropout(0.5)→Linear | ? | |
| Domain loss weighting / batch composition | ? | ? | |
| Where GRL is inserted | on the normalised latent | ? | |

`adv_lr_multiplier = 30.0` is the prime suspect: the derm setup ran a deliberately strong adversary. An adversary at the base learning rate may simply be too weak to push back regardless of λ. Report every mismatch before running anything.

### Step 2 — extend λ upward, 1 seed

λ_adv ∈ {10, 30, 100, 300}, with whatever adversary configuration Step 1 establishes as matching derm.

Report per λ, with **leakage first**: leakage (3-class balanced accuracy, against the majority baseline), ID accuracy, then Mahalanobis and kNN on hospitals 1 and 2, then transfer accuracy.

**Target:** leakage balanced accuracy approaching the 3-class majority baseline. That is the precondition. Until it is met, no statement about the cliff is possible.

### Step 3 — classify the outcome

| Outcome | Meaning | Next step |
|---|---|---|
| **(a)** Leakage falls toward majority and OOD detection collapses with it | Derm form replicates on a different λ scale | Dense-sample around that transition; the phase proceeds as planned |
| **(b)** Leakage falls only where ID accuracy also collapses | **Invariance is unattainable here without destroying the task.** Stain colour and tumour signal are entangled beyond separation | Report as a boundary condition. This narrows the derm claim to *"when invariance is achieved"* — more precise, not weaker. A real finding |
| **(c)** Leakage never falls at any λ, with a derm-matched adversary | Implementation difference remains | Report the audit findings and stop; do not keep raising λ indefinitely |

Record ID accuracy at every point so (a) and (b) can be told apart. Plot leakage and ID accuracy against λ on shared axes — that pair of curves is the deliverable of this round.

### Hard stop

If λ=300 with a derm-matched adversary still leaves leakage above ~0.8, **stop.** Do not continue escalating. Report outcome (c) with the configuration audit, and treat Camelyon17 as a setting where this adversarial formulation does not induce invariance.

---

## Framing note for whichever outcome lands

Outcome (b) is not a failure of the project. It says: *the derm phenomenon requires invariance to actually be achieved, and there exist realistic medical settings where adversarial training cannot achieve it without destroying the task.* That is a useful and publishable boundary condition, and it is more informative than a clean replication would be about when practitioners should expect the safety failure.

Only outcome (c) is uninformative, and it is a bug to be fixed rather than a result.

**iWildCam stays gated.** Do not start it until Camelyon17 reaches (a) or (b).

Standing rules unchanged: leakage reported as balanced accuracy against its majority baseline, failures reported as findings, no retuning toward a prediction, outputs under `results/paperB/phase12/`.
