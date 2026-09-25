# WORK ORDER — Phase 13.5: toy model for the sub-chance inversion

**Priority: highest remaining item.** Fourteen phases of ablation have refuted four mechanisms without producing one. This is the attempt to derive it rather than ablate toward it.

Compute is negligible (CPU or one small GPU allocation, seconds to minutes per run). The cost is thought, not hardware. Do not let it queue behind Phase 12.

---

## 1. What must be explained

The toy model is not chasing one number. It must reproduce a **set** of independent observations from the derm study. A model that reproduces only the headline is weak evidence; one that reproduces several independent facts is strong.

| # | Observation | Source |
|---|---|---|
| F1 | AUROC 0.863 at λ=0 → 0.508 at λ=0.25 → 0.413 at λ=2. A **cliff**, not a gradient | Phase 3 |
| F2 | **Below 0.5**, robustly: OOD ranked as *more* in-distribution than ID | Phases 1, 2, 2.5 |
| F3 | MSP, Energy, cosine and Mahalanobis all collapse **together** to ~0.41 | cbm_revision |
| F4 | Restricting ID to the 6 classes shared with OOD does **not** fix it (0.414 vs 0.409) | Phase 2.5b |
| F5 | Reweighting ID to the OOD class mix makes the inversion **stronger** (0.240) | Phase 2.5b |
| F6 | Holds on a third domain never in `L_adv` (Fitzpatrick17k 0.399) | Phase 2.5a |
| F7 | Holding the adversarial domain out of training changes almost nothing (0.427 vs 0.407) | Phase 2 |

F4, F5 and F7 are the discriminating ones — they are why the obvious explanations died.

## 2. The analytical baseline — derive this first, before any code

For a **linear encoder on Gaussian data**, the answer is closed-form and it is a useful negative result.

Let `x | y=k, d ~ N(μ_k + β_d·u, σ²I)` and `z = Wx`. Then:

```
z | y=k, d  ~  N(W μ_k + β_d·Wu,  σ² W Wᵀ)
```

Adversarial pressure drives `Wu → 0`. At `Wu = 0` the class-conditional latent distributions of the two domains are **identical**. A Mahalanobis score fit on domain 0 therefore has identical score distributions for both domains, giving **AUROC = 0.5 exactly** when class priors match.

Two consequences to state explicitly in the report:

1. **A linear encoder on equal-covariance Gaussian data cannot produce sub-chance AUROC**, except through class-composition differences — which F4 and F5 have already excluded empirically.
2. Therefore **F2 requires some ingredient outside the linear-Gaussian picture.** Identifying the minimal such ingredient is the result of this phase.

Write this derivation out properly (it is short) — it belongs in the paper as the setup that motivates everything after it.

## 3. Data generating process

Implement exactly; every parameter below is a knob the ablation grid will move.

- Input dimension `D = 50`
- `K` classes, means `μ_k` drawn once on a random orthonormal basis of a label subspace, fixed across runs by seed
- Domain direction `u`, orthogonal to the label subspace
- Domain 0 (ID): `β_0 = 0`, `n_0 = 20000`
- Domain 1 (OOD): `β_1 = δ`, `n_1 = 2000` — **deliberately mirrors the ISIC/PAD size ratio**
- Per-domain input noise scale `τ_d`: `x = μ_k + β_d·u + τ_d·ε`, `ε ~ N(0, I)`

**Class priors:** domain 0 carries all `K` classes; domain 1 carries a subset `K' < K`. This mirrors PAD lacking DF and VASC, and lets the toy reproduce (or fail to reproduce) F4 and F5 internally.

**Note on `τ_d`:** if `τ_1 < τ_0`, even a linear encoder yields sub-chance — but that would be a static data property, visible at λ=0 too. The real data shows 0.863 at λ=0, so a pure input-variance explanation is already excluded. Include `τ_d` as a knob anyway, to confirm the toy reproduces that exclusion.

## 4. Models

**Encoder** `f: R^D → R^m`
- `linear`: single matrix `W`
- `mlp`: 2 hidden layers, ReLU
- Latent dim `m ∈ {2, 4, 8, 16, 64}` — 16 matches derm
- Optional `BatchNorm1d(m)` on the latent, matching `lesion_bn`

**Classifier** `g: R^m → R^K`, cross-entropy **on domain 0 only** — this mirrors `ignore_index=-1` for PAD and is likely essential, so it must be a switchable flag, not hard-coded.

**Adversary** `h: R^m → R^2` behind a GRL with weight λ
- `linear` adversary vs `mlp` adversary (2 hidden layers, BN, dropout — the derm head template)
- Adversary LR multiplier as a knob, default **30.0** to match derm

**Training:** AdamW, batches balanced 50/50 across domains (as in derm), λ swept over `{0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8}`, 5 seeds per cell.

## 5. Evaluation — mirror the real protocol exactly

Fit **on domain-0 train latents only** (assert it):
- Class-conditional Mahalanobis, shared covariance — primary
- MSP, Energy, cosine-to-nearest-centroid, kNN (k=50) — for the F3 check

Score domain-0 test vs domain-1. Report AUROC, plus:
- `Var(z | domain 1) ÷ Var(z | domain 0)`
- Median distance to nearest class centroid, each domain
- Distance to the **global** centroid, each domain
- Participation ratio of the latent covariance (ties to 13.1)
- `‖Wu‖` or its MLP analogue — the measured residual domain component
- Domain probe accuracy on the latent, as the toy's "leakage"

## 6. The ablation grid — which ingredient is necessary?

This is the core of the phase. Ablate **one at a time** from the full configuration and record whether sub-chance survives:

| Ablation | Hypothesis being tested |
|---|---|
| linear vs MLP encoder | Does sub-chance need non-linearity? |
| CE on domain 0 only **vs both domains** | Is the label asymmetry essential? |
| latent dim `m` 2 → 64 | Does sub-chance need a compressed latent? |
| adversary capacity: linear vs MLP | Does a weak adversary fail to induce it? |
| adversary LR multiplier 1 vs 30 | Is it the strength of the adversary or λ itself? |
| BatchNorm on/off | Does the latent BN contribute? |
| `n_1/n_0` ratio 0.1 → 1.0 | Does minority-domain size matter? |
| `K' = K` vs `K' < K` | Reproduces F4/F5 internally |
| training to convergence vs early stopping | **Is sub-chance an optimisation-path artifact rather than a property of the optimum?** |

The last row matters more than it looks. The idealised adversarial optimum is `p(z|d=0) = p(z|d=1)`, which gives exactly 0.5. Sub-chance is a *deviation* from that optimum, so it is likely a property of where training actually lands — not of where it is heading.

### Leading hypothesis to test (do not assume it)

> With a capacity-constrained encoder, matching the **mode** of the ID latent distribution is cheaper than matching its **tails**. The encoder satisfies the adversary by mapping minority-domain inputs into the high-density core, while the classification loss simultaneously **spreads domain-0 samples outward** into separated class clusters. ID acquires spread; OOD does not. OOD becomes more typical than typical.

If true, `Var(z|d=1) / Var(z|d=0)` falls below 1 and keeps falling with λ, and removing the label asymmetry (CE on both domains) should weaken or abolish the inversion.

## 7. Validation against the independent facts

For the configuration that reproduces sub-chance, check how many of F1–F7 it also reproduces **without further tuning**:

- F1: is the collapse a cliff at small λ, or a gradient?
- F3: do all four detectors collapse together?
- F4: does restricting ID to shared classes leave it unchanged?
- F5: does reweighting ID to the OOD class mix make it stronger?
- F7: add a second, never-adversarially-seen OOD domain (`β_2 = δ'`) — does it invert too?

**Report the count.** Reproducing F2 alone is weak. Reproducing F2 plus F3, F4, F5 and F7 with no extra fitting is strong evidence the mechanism is right.

## 8. Reporting

→ `results/paperB/phase13_5/` and `PHASE13_5_REPORT.md`:

1. The analytical linear-Gaussian derivation and its conclusion
2. The ablation table: which ingredients are **necessary**, which **sufficient**, which irrelevant
3. The minimal configuration that reproduces sub-chance
4. The F1–F7 validation count for that configuration
5. AUROC and variance-ratio curves vs λ, toy and derm on shared axes
6. A plain-language statement of the mechanism, or an explicit statement that none was found

## 9. Honesty requirements

**Do not tune the toy model until it produces the target curve.** A toy fitted to its target explains nothing. Fix the grid in advance, run it, report what the grid says.

**A negative result here is valuable and must be reported as such.** If no configuration reproduces sub-chance, that means the effect requires something absent from this model — real image statistics, a pretrained backbone, optimisation dynamics at scale — and that **narrows the search**, which is worth more than a fitted curve. The derm paper then reports a robust phenomenon with four refuted explanations and an explicit statement that a controlled synthetic reproduction failed. That is an honest and publishable position.

**If the linear case saturates at exactly 0.5 and only the MLP goes below**, say so precisely — that identifies non-linearity as necessary and is itself a clean finding.

## 10. Why this is the priority

It is the only route to a **general** claim that does not depend on Phase 12 replicating. If the mechanism follows from the training objective rather than from dermatology, generality is *argued* rather than *asserted* — and that holds even if Camelyon17 lands on outcome (c).

It is also the cheapest item remaining, by a wide margin.
