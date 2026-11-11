# Paper B R12 reweight — inventory (pre-run)

Repo: `/data2/cmdir/home/toandq/CSG-Skin-paperB` (writable). Tip: see `CODE_VERSION.txt`.
Dirty tree: `DIRTY_STATUS.txt` / `DIRTY_DIFFSTAT.txt` (untracked r11 leftovers only at audit time).
Split hash (ISIC lesion + PAD paths): `SPLIT_HASH.txt`.

## Recipe verification

| item | expected | found |
|---|---|---|
| CURRENT latent | 16 | 16 (`r6_common.CURRENT`) |
| cls_lr_mult | 0.2 | 0.2 |
| max_epochs | 40 | 40 |
| balance | none | none |
| CKPT_POLICY | last | last |
| Seeds | 42–82 step 10 | `R.SEEDS` |
| Bootstrap | 2000 / rng 20261005 | `PREREGISTER_BOOTSTRAP.json` |

Historical R6 R1 uses **resampling** of the ISIC adversary stream (`adv_mix` / `class_weighted_probs`).
R12 Block A deliberately uses **fixed exposure + loss reweight** — cross-protocol vs R1.

## Assets present

- ERM refs: `checkpoints/r5_p8/erm_ref/single_dann_ladv0_s{42,52,62,72,82}/last.ckpt` + `REFERENCE.json` (ID bal mean 0.558).
- Dual Table-1 path ckpts under `checkpoints/phase3/runB_orth1_*` (for rescoring where features exist).
- R6 R1 cur features/ckpts under `results/paperB/r6/r1/cur/` and `checkpoints/r6_r1/cur/` (resampling protocol).
- R11 X2–X5 results under `results/paperB/r11/` — preserved; not rerun.
- OOD CSVs: `results/paperB/r5/p7a/sets/{mednode,sd198,scin}.csv`; Fitz via `FITZ_DIR`.
- pad_adv / pad_heldout from Phase 2 split (n=1582 / 716).

## ERM → dual init (Block B audit)

| component | ERM key prefix | Dual key prefix | shapes | action |
|---|---|---|---|---|
| EffB3 backbone | `net.backbone.features.*` (572) | `model.lesion_backbone.features.*` | match | copy |
| EffB3 backbone | same | `model.context_backbone.features.*` | match | copy (same ERM weights) |
| feat BN / classifier | `net.feat_bn`, `net.classifier` (1536→8) | `lesion_bn` (16-d), `lesion_projector`, `lesion_classifier` | **mismatch** | random init |
| domain head | `net.domain_head` on 1536-d | `domain_classifier_adv` on 16-d latent | **mismatch** | random init |

Block B is **not blocked**: safe partial load of both backbones only; heads re-init. Documented in every Block B config.

## Missing / out of scope

- No far-OOD non-skin panel pre-staged → mark unavailable (no download).
- Do not use BCN/HAM as unseen under overlapping ISIC training.
- R11 X3–X5 not rerun.

## Prior disclosure

π_pad_adv estimated once from pad_adv label column (existing R1 protocol). Used only to set requested adversary weights / historical resampling proportions. **Not** label-free adaptation.
