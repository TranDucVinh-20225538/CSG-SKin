# Phase 13.5 — toy model for the sub-chance inversion

Grid locked in `GRID_LOCKED.json` before any fit. Not retuned. Failures are findings.

## Analytical baseline (linear encoder, equal-covariance Gaussians)

Let \(x \mid y=k,\, d \sim \mathcal{N}(\mu_k + \beta_d u,\, \tau^2 I)\) and \(z = Wx\). Then

\[ z \mid y=k,\, d \;\sim\; \mathcal{N}\big(W\mu_k + \beta_d\, Wu,\; \tau^2 WW^\top\big). \]

A perfect adversary equalises the domain-conditional latents. With a shared covariance this requires the class-conditional means to match across domains for every \(k\):

\[ W\mu_k + \beta_0 Wu = W\mu_k + \beta_1 Wu \quad\Rightarrow\quad (\beta_1-\beta_0)\,Wu = 0. \]

If \(\beta_1 \neq \beta_0\), necessarily \(Wu \to 0\). At \(Wu=0\) the two domains have **identical** class-conditional Gaussians. A class-conditional Mahalanobis score fit on domain 0 then has identical score distributions on both domains, so **AUROC \(= 0.5\) exactly** when class priors match.

Two consequences:

1. A linear encoder on equal-covariance Gaussian data **cannot** produce sub-chance AUROC except through class-composition differences. F4 and F5 have already excluded composition in the real data (6-class restriction 0.414 vs 0.409; reweighting *strengthens* the inversion to 0.240).
2. Therefore F2 (robust AUROC \(< 0.5\)) requires an ingredient outside the linear-Gaussian picture. Identifying the minimal such ingredient is the result of this phase.

If instead \(\tau_1 < \tau_0\), even a linear map yields \(\Sigma_d = \tau_d^2 WW^\top\), so OOD is more concentrated and Mahalanobis inverts **at \(\lambda=0\)**. The derm number at \(\lambda=0\) is 0.863, so a static input-variance gap is already excluded. The `tau1_half` cell exists only to confirm the toy reproduces that exclusion.

## Locked full configuration

MLP encoder (128–128 ReLU), \(m=16\), BatchNorm on, \(L_\mathrm{cls}\) on domain 0 only, derm-template adversary, `adv_lr_multiplier=30`, \(n_0=20000\), \(n_1=2000\), \(K=8\), \(K'=6\), \(\tau_0=\tau_1=1\), 40 epochs, 5 seeds, \(\lambda\in\{0,0.05,0.1,0.25,0.5,1,2,4,8\}\). Ablations move one knob.

## Full configuration vs λ

| λ | Maha | MSP | Energy | cosine | kNN-50 | leak bal | var OOD/ID | PR |
|---:|---|---|---|---|---|---|---|---|
| 0 | 0.327 ± 0.010 | 0.578 ± 0.009 | 0.646 ± 0.008 | 0.530 ± 0.014 | 0.549 ± 0.014 | 0.513 ± 0.021 | 0.618 ± 0.010 | 7.175 ± 0.036 |
| 0.05 | 0.324 ± 0.013 | 0.575 ± 0.011 | 0.642 ± 0.013 | 0.528 ± 0.013 | 0.543 ± 0.014 | 0.498 ± 0.001 | 0.628 ± 0.023 | 7.181 ± 0.034 |
| 0.1 | 0.337 ± 0.015 | 0.578 ± 0.014 | 0.642 ± 0.014 | 0.537 ± 0.013 | 0.556 ± 0.014 | 0.493 ± 0.002 | 0.636 ± 0.032 | 7.194 ± 0.055 |
| 0.25 | 0.381 ± 0.009 | 0.556 ± 0.015 | 0.600 ± 0.015 | 0.528 ± 0.013 | 0.539 ± 0.012 | 0.501 ± 0.005 | 0.730 ± 0.029 | 7.186 ± 0.023 |
| 0.5 | 0.436 ± 0.008 | 0.534 ± 0.009 | 0.553 ± 0.012 | 0.521 ± 0.012 | 0.522 ± 0.012 | 0.500 ± 0.002 | 0.857 ± 0.018 | 7.181 ± 0.035 |
| 1 | 0.454 ± 0.011 | 0.520 ± 0.006 | 0.535 ± 0.008 | 0.511 ± 0.007 | 0.514 ± 0.006 | 0.503 ± 0.005 | 0.898 ± 0.017 | 7.149 ± 0.048 |
| 2 | 0.399 ± 0.015 | 0.546 ± 0.004 | 0.587 ± 0.010 | 0.519 ± 0.005 | 0.529 ± 0.008 | 0.500 ± 0.004 | 0.763 ± 0.032 | 7.212 ± 0.062 |
| 4 | 0.396 ± 0.021 | 0.546 ± 0.005 | 0.587 ± 0.008 | 0.525 ± 0.009 | 0.538 ± 0.008 | 0.506 ± 0.009 | 0.757 ± 0.019 | 7.355 ± 0.077 |
| 8 | 0.414 ± 0.012 | 0.546 ± 0.016 | 0.580 ± 0.017 | 0.525 ± 0.013 | 0.541 ± 0.013 | 0.524 ± 0.008 | 0.782 ± 0.031 | 7.792 ± 0.029 |

## F1–F7 validation on the full configuration

**2/6** independent facts reproduced on the locked criteria. That count overstates the case: F2 is not the derm F2. See below.

| Fact | Pass? | Detail |
|---|---|---|
| F1 cliff | no | {"auc0": 0.327186125, "auc025": 0.38146815, "auc2": 0.399201225} |
| F2 sub-chance | yes | {"auc2": 0.399201225} |
| F3 detectors together | no | {"detectors_at_2": {"maha": 0.399201225, "MSP": 0.5460597875000002, "Energy": 0.586731475, "cosine": 0.51910895, "knn50": 0.52869445}} |
| F4 shared-class restriction | yes | {"auc_all": 0.399201225, "auc_shared": 0.40440943534830176} |
| F5 reweight to OOD mix | no | {"auc_all": 0.399201225, "auc_reweight": 0.399734575} |
| F6/F7 held-out domain | no | {"auc_domain2": 0.5472147749999999} |


## Ablation table (sub-chance at λ=2)

| Cell | Maha λ=0 | Maha λ=2 | sub-chance at 2? | var OOD/ID at 2 | leak at 2 |
|---|---|---|---|---|---|
| full | 0.327 ± 0.010 | 0.399 ± 0.015 | yes | 0.763 ± 0.032 | 0.500 ± 0.004 |
| linear_encoder | 0.510 ± 0.008 | 0.534 ± 0.010 | no | 0.997 ± 0.010 | 0.596 ± 0.038 |
| ce_both_domains | 0.480 ± 0.009 | 0.482 ± 0.016 | no | 1.056 ± 0.035 | 0.533 ± 0.024 |
| m2 | 0.491 ± 0.066 | 0.476 ± 0.018 | yes | 0.684 ± 0.193 | 0.500 ± 0.000 |
| m4 | 0.439 ± 0.007 | 0.449 ± 0.016 | yes | 0.676 ± 0.098 | 0.500 ± 0.001 |
| m8 | 0.326 ± 0.007 | 0.391 ± 0.018 | yes | 0.714 ± 0.041 | 0.500 ± 0.001 |
| m64 | 0.398 ± 0.016 | 0.444 ± 0.011 | yes | 0.838 ± 0.024 | 0.576 ± 0.025 |
| linear_adversary | 0.327 ± 0.010 | 0.522 ± 0.013 | no | 0.992 ± 0.036 | 0.507 ± 0.012 |
| adv_lr_1 | 0.327 ± 0.010 | 0.520 ± 0.015 | no | 0.996 ± 0.028 | 0.508 ± 0.003 |
| bn_off | 0.524 ± 0.012 | 0.538 ± 0.010 | no | 1.030 ± 0.026 | 0.526 ± 0.021 |
| balanced_n | 0.347 ± 0.010 | 0.438 ± 0.021 | yes | 0.824 ± 0.051 | 0.501 ± 0.002 |
| Kprime_eq_K | 0.343 ± 0.012 | 0.391 ± 0.022 | yes | 0.745 ± 0.051 | 0.500 ± 0.001 |
| early_stop_5ep | 0.364 ± 0.010 | 0.529 ± 0.019 | no | 1.006 ± 0.022 | 0.525 ± 0.025 |
| tau1_half | 0.035 ± 0.005 | 0.055 ± 0.009 | yes | 0.331 ± 0.029 | 0.497 ± 0.020 |

Sub-chance on full died when: linear_encoder, ce_both_domains, linear_adversary, adv_lr_1, bn_off, early_stop_5ep.

Linear saturates at or above chance; only the MLP went below. Non-linearity is necessary. Clean finding.

`tau1_half` at λ=0 is Maha **0.035 ± 0.005**. A static input-variance gap inverts without an adversary, as the derivation said. The full config has τ₁=τ₀ and *also* inverts at λ=0 (0.327), so the full cell failed the same exclusion the derm number (0.863 at λ=0) imposes.

## What the grid actually showed (do not overclaim)

The full MLP+BN cell is **already inverted at λ=0** (Maha 0.327 ± 0.010). AUROC then *rises* toward chance as λ grows (0.327 → 0.381 at 0.25 → 0.399 at 2). That is the opposite of F1. Derm starts at 0.863 and collapses. This toy never occupied that regime.

Turning BatchNorm off restores λ=0 to 0.524 and λ=2 to 0.538. The λ=0 inversion is a **BatchNorm-on-mixed-batches** artifact: 50/50 domain-0/1 batches feed `BatchNorm1d` even when λ=0, so OOD already sits in the ID core before any GRL gradient exists. `linear_adversary` and `adv_lr_1` share the same λ=0 number as full (0.327) and recover to ~0.52 at λ=2 — the adversary was not the cause of the inversion.

Linear encoder: 0.510 at λ=0, 0.534 at λ=2. Sits at chance. Matches the closed-form: no sub-chance without an extra ingredient. The extra ingredient on this grid was BN+MLP, not λ.

Detectors do not collapse together (F3 fails): at λ=2, Maha 0.399 vs MSP 0.546 / Energy 0.587 / cosine 0.519 / kNN 0.529. Reweighting does nothing (F5: 0.400 vs 0.399). A never-trained second domain does not invert (F7: 0.547).

## Mechanism statement

**None that explains the derm facts was found.**

The locked grid produced a BatchNorm-induced sub-chance score that is already present at λ=0. That is a different phenomenon from the derm cliff (high AUROC at λ=0, collapse of every detector at small λ, stronger inversion under class-mix reweighting, inversion on a held-out third domain). Reproducing “AUROC < 0.5 at λ=2” under those conditions is the weak form of F2 the work order warned against.

What the grid *did* establish, and what belongs in the paper:

1. Linear + equal-covariance Gaussians do not invert (analytic and numeric). F2 requires something outside that picture.
2. A static τ₁<τ₀ inverts at λ=0 and is excluded by the derm λ=0 number.
3. On this synthetic DGP, BatchNorm on mixed domain batches is sufficient to invert at λ=0. That is a toy-specific artifact, not a candidate derm mechanism.
4. A controlled reproduction of the derm *set* of facts failed. The derm paper reports a robust phenomenon, four refuted explanations, and an explicit statement that this synthetic grid did not recover it. Not retuned.

Figures: `phase13_5/figures/fig_13_5_*.{png,pdf}`.
