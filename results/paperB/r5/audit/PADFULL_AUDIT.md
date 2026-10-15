# pad_full audit — paper/midl/paper_b.tex (branch paperb/reviewer-r1)

Read-only audit; nothing in the manuscript was changed. "pad_full" means all 2,298 PAD images, including the 1,582 `pad_adv` images the adversarial/context branches trained on; `pad_heldout` is the 716-image patient-disjoint partition. Methods (l.210-212) state: "All OOD and cross-domain numbers reported in the main text use `pad_heldout`."

Partition codes: **probe** = domain-leakage probe trained/tested on ISIC test ∪ pad_full (`leakage.z_lesion.bal_acc_mean`, `depths_pad_full.*.leakage_bal_acc_mean`, `leakage.pad_full.*`, phase15 objectives `leakage.bal_acc_mean`); **OOD** = OOD-detection AUROC scored on pad_full; **heldout** = scored on pad_heldout (listed only for contrast).

No pad_heldout leakage probe exists for any dual-encoder run (phase3_sweep, phase13, r2/item4, r3/c); pad_heldout probes exist only for the single-encoder DANN (phase15 `leakage.pad_heldout`) and ResNet-50 (reviewer_r1).

## 1. Single-encoder table (tab:single, l.990-996) and the claims that rest on it

Source: `phase15/single_dann/single_dann_ladv*_s*/summary.json`, seed means. Every OOD cell = `ood.pad_full.*`; the leakage column = `leakage.pad_full.bal_acc_mean`. pad_heldout values = `ood.pad_heldout.*` (identical to `r4/r4_results.json` `grl_arm[λ].pad_heldout`) and `leakage.pad_heldout.bal_acc_mean`. The caption (l.982-984) and table do not mention pad_full.

| λ | n | Leakage printed (pad_full) / heldout | Maha | kNN | MSP | cosine | Energy |
|---|---|---|---|---|---|---|---|
| 0 | 5 | 0.970 / 0.952 | 0.700 / 0.700 | 0.944 / 0.935 | 0.868 / 0.857 | 0.884 / 0.875 | 0.911 / 0.901 |
| 0.25 | 3 | 0.763 / 0.684 | 0.525 / 0.539 | 0.528 / 0.548 | 0.515 / 0.548 | 0.509 / 0.530 | 0.507 / 0.536 |
| 0.5 | 3 | 0.795 / 0.742 | 0.510 / 0.524 | 0.517 / 0.539 | 0.501 / 0.543 | 0.493 / 0.519 | 0.507 / 0.543 |
| 1 | 3 | 0.795 / 0.744 | 0.548 / 0.555 | 0.521 / 0.541 | 0.462 / 0.517 | 0.482 / 0.517 | 0.448 / 0.498 |
| 2 | 5 | 0.807 / 0.760 | 0.509 / 0.521 | 0.516 / 0.542 | 0.479 / 0.540 | 0.468 / 0.514 | 0.470 / 0.529 |
| 4 | 3 | 0.798 / 0.756 | 0.502 / 0.513 | 0.486 / 0.515 | 0.471 / 0.519 | 0.454 / 0.490 | 0.449 / 0.501 |
| 8 | 3 | 0.787 / 0.738 | 0.520 / 0.525 | 0.532 / 0.558 | 0.519 / 0.566 | 0.490 / 0.526 | 0.500 / 0.551 |

(λ=0 Maha agrees to 3 d.p. on both partitions by coincidence: per-seed values differ, e.g. s42 0.6626 vs 0.6651.) ID acc and X-dom columns are partition-free / pad_heldout and are correct.

**"Four of five below chance by λ = 4"** (l.533, caption l.984, l.1089): true on pad_full (kNN 0.486, MSP 0.471, cosine 0.454, Energy 0.449; Maha 0.502). On pad_heldout only cosine is below chance (0.490); Maha 0.513, kNN 0.515, MSP 0.519, Energy 0.5007 (rounds to 0.501). Over the whole pad_heldout table, only two cells are below 0.5: λ=1 Energy 0.498 and λ=4 cosine 0.490. **The claim does not hold on pad_heldout.**

## 2. All pad_full-sourced numbers

| line | printed | what | source | partition | labelled_in_text | pad_heldout value | note |
|---|---|---|---|---|---|---|---|
| 990-996 | 35 detector cells (see §1) | single-encoder Maha/kNN/MSP/cosine/Energy | phase15 single_dann `ood.pad_full.*` | OOD | no | see §1 | the table says nothing about pad_full; Methods says all OOD numbers use pad_heldout |
| 990-996 | 0.970, 0.763, 0.795, 0.795, 0.807, 0.798, 0.787 | single-encoder leakage column | phase15 single_dann `leakage.pad_full.bal_acc_mean` | probe | no (caption only says "balanced accuracy") | 0.952, 0.684, 0.742, 0.744, 0.760, 0.756, 0.738 | |
| 533, 984, 1089 | "four of the five sit below chance by λ=4" | claim | tab:single pad_full row | OOD | no | 1 of 5 below chance (cosine 0.490) | does not hold on pad_heldout |
| 532, 983, 1086 | "every detector collapses at λ=0.25" | claim | tab:single | OOD | no | 0.530-0.548 | still near chance on pad_heldout, but higher |
| 1086-1088 | kNN 0.944→0.528, energy 0.911→0.507, cosine 0.884→0.509, MSP 0.868→0.515 | single-encoder λ0→0.25 | phase15 `ood.pad_full.*` | OOD | no | kNN 0.935→0.548, energy 0.901→0.536, cosine 0.875→0.530, MSP 0.857→0.548 | |
| 1090 | 0.454 ± 0.014 | single-encoder cosine, λ=4 | phase15 `ood.pad_full.cosine_max` (sd 0.0145) | OOD | no | 0.490 ± 0.018 | above it, the "below chance" example only works on pad_full |
| 567 | 0.509 | single-encoder Maha λ=2 ("0.427 versus 0.509") | phase15 `ood.pad_full.mahalanobis_classcond` | OOD | no | 0.521 | 0.427 beside it is the image-level dual-encoder pad_heldout value, so the comparison mixes partitions |
| 705 | 0.525 | single-encoder Maha λ=0.25 ("reaches chance") | phase15 `ood.pad_full.mahalanobis_classcond` | OOD | no | 0.539 | |
| 561-562 | 0.76-0.81 | single-encoder leakage plateau | phase15 `leakage.pad_full` | probe | no | 0.74-0.76 | |
| 562 | 0.571 | dual-encoder leakage "on that split" | phase3_sweep/phase13 image-level λ=0.25, 5 seeds | probe | no | none exists | |
| 506-507 | 0.51 (lower end of "0.51-0.55") | GRL λ=0.25, range of 5 detectors | pad_full Energy 0.507 | OOD (mixed) | **mislabelled**: l.504 says the detectors are scored on pad_heldout | pad_heldout range 0.53-0.55 (cosine 0.530 … kNN 0.548) | upper end 0.55 is the pad_heldout value; the range mixes partitions |
| 505-506 | 0.970 → 0.763 | GRL leakage | phase15 `leakage.pad_full` | probe | **yes** (l.504-505) | 0.952 → 0.684 | |
| 507-509 | 0.900, 0.836 | MMD leakage w100 / w10000 | r4 `mmd_arm.*.leakage` (pad_full probe stored at training) | probe | yes (l.504-505) | none exists | |
| 600-601 | 0.763 ("0.77 versus 0.763") | single-encoder leakage λ=0.25 | phase15 `leakage.pad_full` | probe | no | 0.684 | |
| 579 | 0.970 ("the 0.970 that ISIC↔PAD reaches") | ISIC-PAD separability from ERM features | most likely phase15 single_dann λ0 `leakage.pad_full` 0.9705 (objectives ERM is 0.973) | probe | no | 0.952 | source inferred from the value; I could not tell for certain which run is meant |
| 960-970 | Maha col: ERM 0.838, CORAL 0.999, MMD1 0.813, MMD100 0.716, MMD3000 0.748, MMD10000 0.665, GRL 0.525, IRM 0.729, GroupDRO 0.868 | tab:obj Mahalanobis AUROC | phase15/objectives `ood_pad_full.mahalanobis_classcond`; phase15b mmd_aggregate `ood_pad_full.*`; phase15 single_dann (GRL) | OOD | **only ERM** (§ footnote); the footnote implies the other rows are not pad_full | ERM 0.834, CORAL n/a, MMD1 0.811, MMD100 0.724, MMD3000 0.747, MMD10000 0.664, GRL 0.539, IRM n/a, GroupDRO n/a (r4_results.json mmd_arm / grl_arm) | prose l.508 cites the pad_heldout 0.811→0.724 for the cells the table prints as 0.813/0.716 |
| 960-970 | Leakage col: 0.973, 0.998, 0.967, 0.900, 0.893, 0.836, 0.763, 0.950, 0.976 | tab:obj leakage | phase15/objectives / phase15b `leakage.bal_acc_mean`; single_dann `leakage.pad_full` | probe | no (caption: "balanced accuracy, floor 0.5") | GRL 0.684; others none exist | |
| 491-499 | 0.967 → 0.836, 0.13, 0.950, 0.998 | MMD/IRM/CORAL leakage | same as tab:obj | probe | no | none exists | |
| 494-495 | Mahalanobis 0.665, kNN 0.738, energy 0.694 | MMD w10000 detectors | phase15b `mmd_aggregate.json` `ood_pad_full.*` | OOD | no | 0.664, 0.740, 0.696 (r4 mmd_arm.mmd_w10000.pad_heldout) | |
| 491-495 | "leaves every detector well above chance" | claim | phase15b ood_pad_full | OOD | no | holds on pad_heldout too (min 0.633) | |
| 509, 512 | "at or above 0.66"; 0.63-0.74 | MMD w100 / w10000 detector ranges | r4 mmd_arm pad_heldout | heldout | yes | — | pad_full rounds the same; listed for completeness |
| 398-399, 1014 | 0.914 → 0.553 | z_lesion leakage paired with participation ratio | phase13/per_run `depths_pad_full.z_lesion_norm.leakage_bal_acc_mean`: λ0 5-seed 0.91449, λ0.25 seeds 42/52/62 0.55257 | probe | no | none exists | also stale: λ0.25 5-seed = 0.571; λ0 is 5 runs, not "the same three" |
| 441-442 | 0.982 → 0.941 | backbone leakage, "flat across the sweep" | phase13/per_run (image-level) `depths_pad_full.backbone_raw_lesion.leakage_bal_acc_mean`, λ0 0.98237 → λ8 0.94093 | probe | no | none exists | section is lesion-level per l.613; lesion-level (r3/c, plotted in Fig 5) is 0.982 → 0.943 |
| 442 | 0.749 → 0.475 | backbone Mahalanobis AUROC | phase13/per_run (image-level) `depths_pad_full.backbone_raw_lesion.mahalanobis_eps1e-3`, λ0 0.7490 → λ2 0.4750 | OOD | no | image-level pad_heldout 0.749 → 0.489 (phase14/tails `spaces.backbone_raw_lesion.a2.pad_heldout.full_auroc`); no lesion-level pad_heldout backbone value | lesion-level pad_full (r3/c, Fig 5) 0.683 → 0.446; prose and figure disagree |
| Fig 5 (l.939-944) | both panels | backbone leakage; backbone Maha AUROC | `scripts/r3_c_lesion_figures.py` → `plot_manuscript_figures.fig5_no_mediation`, r3/c/per_run `depths_pad_full.backbone_raw_lesion.{leakage_bal_acc_mean, mahalanobis_eps1e-3}` | probe + OOD | no | right panel: none at lesion level | the right panel is pad_full OOD AUROC, captioned without a partition |
| Fig 1a (l.324) | leakage panel | dual-encoder leakage | r2/item4 summaries `leakage.z_lesion.bal_acc_mean` | probe | no | none exists | Figs 1b-d, 2, 3, 4 and 6b use pad_heldout / Fitzpatrick / ID; Fig 6a is the HAM probe, not PAD |
| 60-61 | 0.927, 0.555, 0.64-0.66 | abstract leakage | r4 lesion_table1 leakage (r2/item4 summaries) | probe | no | none exists | |
| 311-317 | 0.927, 0.614, 0.581, 0.563, 0.555, 0.636, 0.663 (± s.d.) | tab:dial leakage column | r4 `lesion_table1[λ].leakage` | probe | no (caption: "balanced accuracy (chance 0.5)") | none exists | |
| 336-339, 377-380, 435-438, 715-717 | 0.927, 0.555, 0.636, 0.663, 0.563 | leakage in prose | r4 lesion_table1 | probe | no | none exists | |
| 1044-1050 | 0.915, 0.571, 0.555, 0.591, 0.584, 0.548, 0.625 (± s.d.) | tab:imagelevel leakage column | phase3_sweep `leakage.z_lesion.bal_acc_mean` | probe | no | none exists | |
| 1052-1053 | 0.802, 0.979 | control-row leakage (EffB3, ResNet-50) | phase10/table1.md; Phase 1 `check_leakage.py` | probe | no | none exists | **also the wrong metric**: plain accuracy (floor 0.688), in a balanced-accuracy column (LEAKAGE_FLOOR_AUDIT.md l.21, 61) |
| 1052-1053 | 0.726 ± 0.021, 0.868 ± 0.019 | control-row OOD AUROC (EffB3 16-d control, ResNet-50 baseline) | 0.868: `ood_dual_branch_aggregate.csv` baseline_soft/backbone_raw/mahalanobis_classcond (n_ood 2298); 0.726: Phase 1 `results/effb3_control/` (cited in MASTER_REPORT.md / PHASE1_REPORT.md, n_ood 2298) | OOD | no (the column's other rows are pad_heldout) | none found | |
| 582, 1124 | 0.876 | ISIC↔PAD hand-crafted probe "ceiling" | phase1_5/trivial_summary.json probes[4].auroc_mean (counts.pad 2298) | probe (AUROC) | no | none | also a metric mismatch: AUROC compared with a balanced accuracy (0.901) |
| 565 | >0.9999 | domain probe on the context branch | phase1_6 (supervised_head / concentration: n_ood 2298) | probe/OOD on pad_full | no | none found | the context branch was trained on pad_adv domain labels, so 1,582 of the 2,298 images were training data |
| 1143-1150 | 0.998, 0.997, 0.991, 84.5%, >0.9999, 11.6M pairs, 0.55-0.70 | domain heads and PC1 AUROC | phase1_6/supervised_head.json, concentration.json, phase10/table2.md (n_ood 2298; 11.6M = 5,067 × 2,298) | probe/OOD on pad_full | no | none found | as above for z_context |
| 212-213 | 0.413 | earlier-draft λ=2 value | phase3 pad_full Maha (0.4128) | OOD | **yes** | 0.427 | correctly disclosed |

## 3. Numbers that look PAD-based but are pad_heldout (checked, not pad_full)

l.373-374 tail truncation and medians; l.394 Ledoit–Wolf 0.427 (cleanup/ledoit_pad_heldout.json; the phase13 pad_full LW value is 0.413); l.396 9.4e-4 (reviewer_r1/maha_stability.json, pad_heldout); l.544-545 ResNet-50 leakage 0.965→0.939 and EffB3 0.742 (pad_heldout probe, labelled "same partition"); tab:dial OOD AUROC / OOD ECE / X-dom; tab:imagelevel OOD AUROC dual-encoder rows; l.1011-1012 refuted-table pad_heldout/pad_adv values.
