# Paper B R12 — fixed-image adversary reweighting + ERM-init dual

Locked before first training run. Append deviations only; never edit rules after a run starts.
Not public preregistration. Run manifest frozen in `results/paperB/r12_reweight/MANIFEST.json`.

## Questions

1. Does the class-prior intervention recover OOD detection when image exposure is held fixed (reweight, not resample)?
2. Does inversion persist when starting from a competent ERM encoder while actually achieving domain invariance?
3. Do pooled metrics conceal class-specific / seed effects? (answered by stratified + seed tables)

## Recipe (verified against code)

`r6_common.CURRENT`: latent=16, cls_lr_mult=0.2, max_epochs=40, balance=none.
Checkpoint policy: `last.ckpt` (`CKPT_POLICY`). Three-stream DANN (classifier ISIC + ISIC adv + pad_adv).
GRL schedule / adv LR ×30 / λ_ctx = λ_orth = 1 as main recipe.
Primary detector: class-conditional Mahalanobis on `z_lesion_norm`.
Bootstrap: `PREREGISTER_BOOTSTRAP.json` (2000 draws, rng 20261005).
Rule A / degraded / V1 as R11 shared rules.

## Block A — fixed-image adversary reweighting (primary)

Keep task stream and source/PAD adversary **image schedules** identical across prior arms for each seed
(same three-stream draws; no class-prior resampling). Change only source-adversary domain-loss weights:

  w(c) = π_requested(c) / π_source(c)   (0 if π_requested=0)
  Normalize so mean weight over the full fixed source-adversary stream is 1
  (analytic identity under w=π_req/π_src when π_src>0 everywhere used).

π_requested ∈ {prior100=π_ISIC, prior50=midpoint, prior0=π_pad_adv}.
π_pad_adv from pad_adv labels at run start only (disclose: prior estimation uses those labels;
not label-free adaptation). PAD labels never enter the supervised task loss.

Implementation: three-stream with `adv_mix=None` (no resampling); `align=iwdan_oracle` with
`iw_cls=False` and frozen `oracle_w` = normalized w — **DANN + fixed source-adv reweight**,
not R2 IWDAN. Documented as such in every config.

Runs: λ∈{0.5,2}×{prior100,50,0}×seeds{42,52,62,72,82}=30 + λ=0×5 seeds (one baseline/seed)=35.
Prior-specific λ=0 rows are **reference** to the same baseline, not independent replicates.

Invalid if a requested class has zero source support (not expected for these priors).

## Block B — ERM-initialized trainable dual

If compatible: load ERM EffB3 backbone weights into **both** `lesion_backbone` and `context_backbone`
(shape-matched 572 tensors). Heads (projector, BN, classifier, domain) randomly initialized.
Trainable encoders (not frozen X3). Core: λ∈{0,2}×5 seeds, prior100 fixed-image pipeline.
Matched from-scratch controls = Block A λ∈{0,2} prior100 (same pipeline).

## Endpoints (defined before outcomes)

Per eligible cell: pooled AUROC; ID-class-stratified AUROC (each ID class vs same OOD panel) +
unweighted macro over fixed 8-class list; FPR@95TPR = share OOD accepted as ID (score≤τ, τ=95th ID);
ID bal_acc; V1 held-out leakage + class-matched leakage; seed mean±sd; cluster bootstrap CI with units.
Primary unseen: Fitzpatrick17k, MED-NODE, SD-198. pad_heldout reported separately.
Paired prior0−prior100 contrasts across seeds for Block A.

## Reading (pre-specified)

Block A: recovery on unseen under fixed exposure → prior-weighting effect; recovery on unseen but not
pad_heldout → limited for aligned-domain headline; no recovery → weakens prior-only explanation.
Block B: apply R11 competent/invariant/Rule A definitions; if ID improves without invariance, does not
answer competent-invariant inversion; retain negative findings.

## Outputs

`results/paperB/r12_reweight/` (non-overwriting). No manuscript edits. No R11 X3–X5 reruns.

## Deviations

_(append only)_

- 2026-11-09: Train jobs submitted as Slurm array `73217` (0–44%5); score job `73218` afterok. Custom `r12_reweight_train.py` (does not call `R.run`, which hardcodes pad oracle for `iwdan_oracle`). λ=0 uses `align=dann` (no IW).

- 2026-11-09: Resubmitted array with TimeLimit=04:00:00 (train) / 06:00:00 (score) after QOS/time audit; prior array cancelled (~2 min into task 0). New jobs 73220 / 73221.
