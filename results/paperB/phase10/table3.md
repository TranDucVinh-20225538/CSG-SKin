| Prediction | Test | Result | Rules out |
|---|---|---|---|
| Image memorisation of PAD in L_adv | Phase 2: retrain with pad_heldout never in L_ctx/L_adv (716 / 412 patients) | pad_heldout z_lesion Maha 0.427 ± 0.025 vs pad_adv 0.407 ± 0.021 (Δ ≈ 0.02) | Inversion is not memorisation of adversarial images |
| PAD-specific mapping (would be ~0.50 on a third domain) | Phase 2.5a: Fitzpatrick17k, never in any train branch, n=3887 | z_lesion Maha 0.399 ± 0.039, same inversion as PAD 0.409 | Not PAD-specific; structural of the invariant branch |
| Class-mix / missing DF,VASC drives AUROC | Phase 2.5b: 6-class-restricted ID and ID reweighted to PAD mix | 6-class 0.414 vs 0.409; reweight 0.240 (inversion strengthens) | Class composition is not the driver |
| Clean 2×2: z_context detects domain not class; z_lesion the reverse | Phase 4: 4a hold {DF,VASC}, 4b hold {SCC} | 4a z_lesion 0.648 vs z_context 0.641 vs baseline 0.687; 4b z_lesion 0.806 vs z_context 0.607 | No double dissociation. Semantic OOD is graded, 4a=4b equal prominence, null 2×2 |
| z_context 4a is a VASC-colour detector (high VASC, ~0.50 DF) | Phase 4.5b: class-conditional Maha on 4a | z_context DF 0.61, VASC 0.67 — both above 0.50 | VASC/DF hypothesis not confirmed |
| 4b win is near-OOD, 4a loss is far-OOD | Phase 4.5c: cosine-to-nearest-keep on DF/VASC/SCC | cosines 0.570 / 0.566 / 0.565; Pearson(advantage, cosine)=−0.12 | No near/far axis. 4a and 4b simply differ |
| Leakage↔AUROC is a continuous trade-off (SPS) | Phase 3 λ sweep: leakage and AUROC between λ=0.25 and λ=2 | leakage 0.553 → 0.584 while AUROC 0.508 → 0.413. Linearity is an artefact of anchoring at λ=0 | SPS withdrawn. Not a tunable trade-off. There is no safe operating point |
| OOD collapse is mediated by reduced linear domain leakage | Phase 3 λ=0.25→2 + Phase 13.2 backbone | leakage 0.553→0.584 while AUROC 0.508→0.413; backbone leak 0.982→0.941 while backbone Maha 0.749→0.475 | Mediation rejected. Joint effects of λ, not leakage→OOD |
| Inversion is an ID-tail effect | Phase 14.A2: drop pooled upper tail | λ=2 0.427 → 0.412 (p90). Inversion persists in the bulk | Tail hypothesis wrong |
| Heavy-tailed ID toy reproduces the derm F-set | Phase 13.5 v2, pre-registered, BN off | λ=0 0.693; λ=2 0.786. No inversion | Synthetic reproduction failed. No third revision |
| OOD collapse is because z_lesion is only 16-d (original manuscript) | Phase 1 EffB3 16-d control + Phase 3 λ=0 (same 16-d CSG) | EffB3 16-d Maha 0.726; CSG λ=0 Maha 0.863; CSG λ=0.25 Maha 0.508 | Dimensionality is not the cause. λ_adv is the cause |
