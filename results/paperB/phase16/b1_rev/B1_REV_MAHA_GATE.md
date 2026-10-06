# B1-rev pre-registered Maha gate (λ=0 only)

**Orientation:** ID=HAM, OOD=BCN heldout (30% lesions), Mahalanobis fit HAM train only.

| Outcome | Rule |
|---------|------|
| **Continue** | Maha unrestricted @ λ=0 **≥ 0.65** → remaining coarse λ + dense 3-seed |
| **Stop** | Maha @ λ=0 **≤ ~0.5** → do not run rest; narrative = same-modality collapse repeats under either ID/OOD orientation |

Decision read from `ham_bcn_ladv0_s42/summary.json` → `rev_maha_gate` **before** submitting `phase16_b1_rev_coarse.sbatch`.
