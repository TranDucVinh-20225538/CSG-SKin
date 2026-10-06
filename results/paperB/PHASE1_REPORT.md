# Paper B — Phase 1 report

Inference only. No retraining. Checkpoints selected by Phase 0.1 policy
(newest `best-*.ckpt` per run). Statistics (means, covariances, class
centroids, kNN bank) were fit on the **ISIC training split only**; a
runtime assertion rejects val / test / PAD leakage into the fit.

## Honesty (non-negotiable)

`z_context` detectors, `backbone_raw_context` detectors, and `context_predictor_softmax` are **domain-supervised**: the context branch and its predictor were trained with an explicit ISIC-vs-PAD domain label (`L_ctx`). The baseline Mahalanobis score is unsupervised. This asymmetry is labelled in every CSV row via the `supervision` column. The defence against “of course it works, you supervised it” is Phase 4 (semantic OOD, domain held constant) and Phase 11 (unseen third domain), not spin.

## P1 verdict

Prediction P1: Mahalanobis AUROC on `z_context` > 0.95, vs ~0.41 on `z_lesion`, same `runB_orth1` checkpoints.

| Quantity | Value | n seeds |
|---|---|---:|
| runB_orth1 `z_context` Maha AUROC (domain-supervised) | **> 0.9999** | 5 |
| runB_orth1 `z_lesion` Maha AUROC | 0.4089 | 5 |
| baseline_soft `backbone_raw` Maha AUROC | 0.8679 | 5 |

**P1: PASS (z_context Maha AUROC > 0.95)**

Gate: Phase 2 authorised

## Required table

Probe acc / ID bal acc in the left and right columns are the published n=5 figures from `results/cbm_revision` / `results/effb3_control` / `leakage.json` (not re-probed here). **Domain probe acc is plain accuracy** (`check_leakage.py` `accuracy_score`); its majority floor is **0.688**. It is not the Phase 3 dial column, which is 2-class **balanced** accuracy with floor **0.5** (see `LEAKAGE_FLOOR_AUDIT.md`). Maha AUROC for CSG branches and ResNet-50 `backbone_raw` is from this pass. **EffNet-B3 is the primary control** (16-d `z` Maha 0.7259 from `results/effb3_control/`, not the 1536-d `backbone_raw` 0.824 from this pass).

| Branch | Domain probe acc | OOD AUROC (Maha) | ID Bal Acc | supervision |
|---|---|---|---|---|
| Baseline (entangled, ResNet-50) | 0.979 | 0.868 | 0.655 | unsupervised |
| EffNet-B3 single-encoder (primary control) | 0.802 | 0.726 | 0.683 | unsupervised |
| z_lesion (invariant) | 0.724 | 0.409 | 0.702 | unsupervised |
| z_context (leaky) | 0.9997 | > 0.9999 | n/a | **domain-supervised** |

Footnote: never report 1.0000 ± 0.0000. `runB_orth1` n=5 mean Maha AUROC = 0.9999986 ± 9e-7. Seed 42 has **2 discordant pairs / 11.6M** (1 ID outlier above 2 OOD images). A bare 1.0000 reads as a leakage bug; the pair count is the diligence.

ID balanced accuracy gap vs the primary control: **0.683 → 0.702 = +1.9pp** (not vs ResNet-50's 0.655). Recomputed on this pass (ISIC test):
- baseline_soft: 0.6553 ± 0.0129
- runB_orth1: 0.7017 ± 0.0026

## Outputs

- `/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/ood_dual_branch_per_seed.csv`
- `/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/ood_dual_branch_aggregate.csv`

## What ran

- baseline_soft seed=42 kind=baseline id_acc=0.7677 id_bal=0.6570 n_train=16211 n_id=5067 n_ood=2298
- baseline_soft seed=52 kind=baseline id_acc=0.7928 id_bal=0.6762 n_train=16211 n_id=5067 n_ood=2298
- baseline_soft seed=62 kind=baseline id_acc=0.7926 id_bal=0.6555 n_train=16211 n_id=5067 n_ood=2298
- baseline_soft seed=72 kind=baseline id_acc=0.7780 id_bal=0.6517 n_train=16211 n_id=5067 n_ood=2298
- baseline_soft seed=82 kind=baseline id_acc=0.7823 id_bal=0.6359 n_train=16211 n_id=5067 n_ood=2298
- effb3_control seed=42 kind=effb3 id_acc=0.8159 id_bal=0.6939 n_train=16211 n_id=5067 n_ood=2298
- effb3_control seed=52 kind=effb3 id_acc=0.7936 id_bal=0.6892 n_train=16211 n_id=5067 n_ood=2298
- effb3_control seed=62 kind=effb3 id_acc=0.8013 id_bal=0.6857 n_train=16211 n_id=5067 n_ood=2298
- effb3_control seed=72 kind=effb3 id_acc=0.8131 id_bal=0.6887 n_train=16211 n_id=5067 n_ood=2298
- effb3_control seed=82 kind=effb3 id_acc=0.7948 id_bal=0.6584 n_train=16211 n_id=5067 n_ood=2298
- runA_grl seed=42 kind=csg id_acc=0.8097 id_bal=0.7041 n_train=16211 n_id=5067 n_ood=2298
- runA_grl seed=52 kind=csg id_acc=0.8190 id_bal=0.6854 n_train=16211 n_id=5067 n_ood=2298
- runA_grl seed=62 kind=csg id_acc=0.8009 id_bal=0.7095 n_train=16211 n_id=5067 n_ood=2298
- runA_grl seed=72 kind=csg id_acc=0.8099 id_bal=0.6989 n_train=16211 n_id=5067 n_ood=2298
- runA_grl seed=82 kind=csg id_acc=0.7963 id_bal=0.6800 n_train=16211 n_id=5067 n_ood=2298
- runB_orth1 seed=42 kind=csg id_acc=0.8206 id_bal=0.7042 n_train=16211 n_id=5067 n_ood=2298
- runB_orth1 seed=52 kind=csg id_acc=0.8021 id_bal=0.6995 n_train=16211 n_id=5067 n_ood=2298
- runB_orth1 seed=62 kind=csg id_acc=0.8046 id_bal=0.6983 n_train=16211 n_id=5067 n_ood=2298
- runB_orth1 seed=72 kind=csg id_acc=0.8129 id_bal=0.7050 n_train=16211 n_id=5067 n_ood=2298
- runB_orth1 seed=82 kind=csg id_acc=0.8003 id_bal=0.7016 n_train=16211 n_id=5067 n_ood=2298
