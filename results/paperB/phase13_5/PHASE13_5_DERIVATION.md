# Phase 13.5 — analytical baseline (written before the grid ran)

Let \(x \mid y=k,\, d \sim \mathcal{N}(\mu_k + \beta_d u,\, \tau^2 I)\) and \(z = Wx\). Then

\[
z \mid y=k,\, d \;\sim\; \mathcal{N}\big(W\mu_k + \beta_d\, Wu,\; \tau^2 WW^\top\big).
\]

A perfect adversary equalises the domain-conditional latents. With a shared covariance this requires the class-conditional means to match across domains for every \(k\):

\[
W\mu_k + \beta_0 Wu = W\mu_k + \beta_1 Wu \quad\Rightarrow\quad (\beta_1-\beta_0)\,Wu = 0.
\]

If \(\beta_1 \neq \beta_0\), necessarily \(Wu \to 0\). At \(Wu=0\) the two domains have identical class-conditional Gaussians. A class-conditional Mahalanobis score fit on domain 0 then has identical score distributions on both domains, so **AUROC \(= 0.5\) exactly** when class priors match.

Two consequences, independent of any numerical run:

1. A linear encoder on equal-covariance Gaussian data **cannot** produce sub-chance AUROC except through class-composition differences. F4 and F5 have already excluded composition in the real data (6-class restriction 0.414 vs 0.409; reweighting strengthens the inversion to 0.240).
2. Therefore F2 (robust AUROC \(< 0.5\)) requires an ingredient outside the linear-Gaussian picture. Identifying the minimal such ingredient is the result of this phase.

If instead \(\tau_1 < \tau_0\), even a linear map yields \(\Sigma_d = \tau_d^2 WW^\top\), so OOD is more concentrated and Mahalanobis inverts **at \(\lambda=0\)**. The derm number at \(\lambda=0\) is 0.863, so a static input-variance gap is already excluded. The `tau1_half` cell exists only to confirm the toy reproduces that exclusion.

The numerical grid is locked in `GRID_LOCKED.json` and is not retuned toward a target curve.
