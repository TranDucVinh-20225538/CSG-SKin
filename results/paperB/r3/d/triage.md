# R3 D — triage of the 37 free-scan mismatches

Commit of the scan: `7a7333f` (manuscript at `0c9a15e`, before the lesion sweep finished). Re-run at the commit of this file with the extended manifest and the allowlist:

`python3 scripts/verify_manuscript_numbers.py --manifest paper/number_manifest.json --allowlist paper/number_allowlist.json --fail-on manifest`

→ manifest 46 entries, 0 failing, exit 0; free scan 437 match / 5 allowlisted / 26 last-digit / 32 mismatch. Of the 32, 27 are manifest-verified (the free scan does not read the manifest; rows 6–7 are among them and also carry a (b) wording issue), 4 are (b) with no manifest entry (rows 1, 8, 21, 22), and 1 is pending B1 (row 36).

**Verdict:** every printed number has a source; the manuscript agent should fix the text at rows 1, 6–7, 8, 20 and 21 (λ or OOD set not named) and at row 22 (a substantive comparison of different metrics).

Categories: (a) constant or derived → `paper/number_allowlist.json`; (b) manuscript ambiguity → manuscript agent; (c) source was not indexed → aggregate JSON written, manifest entry added; (d) markdown-only → manifest entry on the A3 JSON; (e) source already indexed, but the free scan matched the number to a different candidate (wrong λ, split or method) → manifest entry pinning the key. (e) is not in the work order's list; it is the most common case.

Aggregate JSONs written by `scripts/r3_d_aggregates.py` (mean, s.d., n over seeds of existing per-run files; no new training): `r2/item2/item2_aggregate.json`, `r2/item4/item4_aggregate.json`, `phase3_sweep/sweep_aggregate.json`, `phase15/single_dann/single_dann_aggregate.json`, `phase15b/mmd/mmd_aggregate.json`; `r2/item5/exploratory_results.json` gained an `aggregate` block. Manifest entries added by `scripts/r3_d_manifest_add.py`.

| # | Line | Printed | Cat. | Resolution | Source / reason |
|---:|---:|---|---|---|---|
| 1 | 62 | 0.553 | (b) | manuscript agent | λ = 0.25 leakage (`phase13/per_run` λ0.25 `z_lesion_norm` 0.5526); the abstract sentence names no λ |
| 2 | 64 | 0.707 | (e) | manifest `img_idbal_l2` | `phase3_sweep/sweep_aggregate.json` runB_orth1_ladv2 `id_balanced_acc` mean, n = 5 (0.7073) |
| 3 | 64 | 60.0% | (d) | manifest `img_overlap_pct` | `r2/item4/image_level_overlap.json` `pct_sharing` (60.016) |
| 4 | 350 | 0.068 | (e) | manifest `img_fitz_maha_sd_l2` | `sweep_aggregate.json` runB_orth1_ladv2 Fitzpatrick Mahalanobis s.d., n = 5 (0.0677) |
| 5 | 388 | 0.843 | (e) | manifest `maha_pad_l0` (existing) | `reviewer_r1/bootstrap_results.json` pad_heldout λ0 Mahalanobis seed mean |
| 6 | 422 | 4.879 | (d) + (b) | manifest `pr_l0`; manuscript agent | `phase13/participation_ratio.json` λ0 mean. "across the sweep" but the pair is λ = 0 → 0.25; at λ = 2 the ratio is 4.737 |
| 7 | 422 | 4.500 | (d) + (b) | manifest `pr_l025`; as row 6 | `participation_ratio.json` λ0.25 mean |
| 8 | 423 | 0.553 | (b) | manuscript agent | λ0.25, unnamed. Also: λ = 0 leakage is 0.915 at l.62 (Phase 3) and 0.914 at l.423 (Phase 13, 0.9145) |
| 9 | 434 | 0.386 | (e) | manifest `fitz_ci_lo_l2` | `bootstrap_results.json` fitzpatrick17k λ2 Mahalanobis seed-mean CI lo |
| 10 | 437 | 0.994 | (e) | manifest `resnet50_fitz` | `phase2_5/phase25_aggregate.json` `baseline_soft.fitz_maha.mean` |
| 11 | 454 | 0.167 | (a) | allowlist | six-class chance floor, 1/6 |
| 12 | 455 | 0.298 | (e) | manifest `xfer_effb3_bal` | `phase6_xfer/diagnosis_aggregate.json` `controls.effb3_control.bal.mean` |
| 13 | 456 | 0.184 | (e) | manifest `xfer_acc_l2` | `diagnosis_aggregate.json` `csg[4]` (λ2) `acc.mean` |
| 14 | 456 | 0.385 | (e) | manifest `xfer_majority` | `diagnosis_aggregate.json` `floors.plain_majority_always_BCC` |
| 15 | 458 | 0.581 | (e) | manifest `xfer_auc_l2` (+ `xfer_auc_l0` 0.605) | `diagnosis_aggregate.json` `csg[4].auc.mean` |
| 16 | 483 | 0.25 | (a) | allowlist | a λ value |
| 17 | 500 | 0.589 | (c) | manifest `item5_cos_l1` | `r2/item5/exploratory_results.json` `aggregate.1.0.Cosine.mean` |
| 18 | 501 | 0.648 | (c) | manifest `item5_knn_l1` | `aggregate.1.0.kNN.mean` |
| 19 | 501 | 0.589 | (c) | manifest `item5_msp_l1` | `aggregate.1.0.MSP.mean` |
| 20 | 541 | 0.665 | (e) | manifest `mmd_maha` (+ `mmd_knn`, `mmd_energy`) | `phase15b/mmd/mmd_aggregate.json` mmd_w10000 `ood_pad_full` mean, n = 3. The OOD set is pad_full; the sentence names none |
| 21 | 603 | 0.553 | (b) | manuscript agent | λ0.25, unnamed |
| 22 | 625 | 0.876 | (b), substantive | manuscript agent | Different metrics are compared: 0.901 is HAM↔BCN colour-histogram **balanced accuracy** (`phase16/b0/summary.json`), 0.876 is ISIC↔PAD Laplacian **AUROC** (`phase1_5/trivial_summary.json` probe 4). Like for like: AUROC 0.967 vs 0.876; balanced accuracy 0.901 vs 0.770. "close to" does not hold |
| 23 | 628 | 0.532 | (e) | manifest `hambcn_laplacian` | `phase16/b0/summary.json` `handcrafted_site_unmasked.laplacian_stats.bal_acc_mean` |
| 24 | 635 | 0.963 | (e) | manifest `hambcn_leak_l0` | `phase16/b1/dense3seed_aggregate.json` `rows[0].leakage_mean` |
| 25 | 637 | 0.576 | (e) | manifest `hambcn_cos_l0` | `dense3seed_aggregate.json` `rows[0].cosine_mean` |
| 26 | 638 | 0.970 | (e) | manifest `single_leak_l0` (no label) | `phase15/single_dann/single_dann_aggregate.json` ladv0 `leakage.pad_full.bal_acc_mean`, n = 5. Label omitted: 0.970 also prints MSP (l.410) and Phase 16 leakage (l.622) |
| 27 | 638 | 0.763 | (e) | manifest `single_leak_l025` | same file, ladv0p25, n = 3 |
| 28 | 655 | 0.699 | (e) | manifest `hambcn_ci_hi` (+ `hambcn_ci_lo` 0.666) | `reviewer_r1/bootstrap_phase16_ham_bcn_ladv1.json` `mean_across_seeds_bootstrap_ci` |
| 29 | 701 | 1.4 | (a), **pending B1** | allowlist | derived 0.040 / 0.028 (`r2/item4/item4_aggregate.json`, n = 3) |
| 30 | 701 | 0.040 | (a), **pending B1** | allowlist | derived 0.504 − 0.464, lesion-level id_bal λ0 − λ2; the 0.028 also changes at n = 5 |
| 31 | 721 | 0.531 | (c) | manifest `nv_nearest_id_l2` (+ `nv_nearest_id_l0` 0.528) | `r2/item2/item2_aggregate.json` `z_lesion_norm|pad_heldout` λ2 `frac_nv_nearest_id` |
| 32 | 748 | 6.63 | (c) | manifest `ratio_nv_l0` | `item2_aggregate.json` λ0 `ratio_NV` |
| 33 | 748 | 2.37 | (c) | manifest `ratio_nv_l2` | `item2_aggregate.json` λ2 `ratio_NV` |
| 34 | 978 | 0.11 | (e) | manifest `camelyon_adv_ce_l0` (existing) | `r2/item5/coarse_decision_primary.json` `coarse.0.adv_min_ce` (0.106, primary seed) |
| 35 | 981 | 0.10 | (a) | allowlist | pre-committed threshold, `PRECOMMIT_CAMELYON2.json` `leakage_moved` |
| 36 | 1015 | 0.693 | (c), **pending B1** | manifest after B1 | `item4_aggregate.json` `by_split.image.0.id_bal` (seeds 42/52/62); the 0.504 and 0.19 on the same line move to n = 5 |
| 37 | 1019 | 16,577 | (e) | manifest `fitz_n_rows` | `phase11_third_domain/fitzpatrick17k_label_map.json` `n_rows`; the manifest now parses `{,}` |

Caveats: the allowlist matches on the printed text and a context regex, not on line numbers, so it survives the page cut. A manifest label is checked against the nearest dataset mention in the .tex; 12 entries pass with a "no dataset label nearby" warning (table cells and unlabelled sentences). Verifier changes are additive: `--allowlist`, and `{,}` accepted in manifest `written`.
