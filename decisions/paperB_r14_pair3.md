# Paper B R14 — ISIC 2019 → MILK10k dermoscopic (pair 3)

Locked design under `results/paperB/r14_pair3/`. Does not modify `r12_reweight/`, `r13_pair2/`,
or the submitted r12c manuscript. Append deviations only; never edit rules after a run starts.

## Why this pair

Same source, same split, same recipe as the main ISIC→PAD-UFES-20 result; only the adversary's
target changes. Target = MILK10k dermoscopic images (Vienna, Ankara, Queensland GP, MSK, Skopje;
CC-BY-NC 4.0; https://api.isic-archive.com/doi/milk10k/). Label shift has the same direction and
size as ISIC→PAD: after mapping and overlap removal, NV 0.508→0.081, BCC 0.131→0.568,
TV(ISIC 2019, MILK) ≈ 0.557 (TV ISIC→PAD ≈ 0.583). R13 (DERM12345 → MILK10k clinical close-up)
changed the source and the modality at once, never got V1 ≤ 0.60 and had two degenerate seeds;
R14 keeps the source/recipe that already reaches V1 ≈ 0.50.

## Data

- Source: ISIC 2019, lesion-disjoint split seed 42 (same as Table 1 / main recipe).
- Target: MILK10k **dermoscopic only** (5,240 images, 1 per lesion). Do not redistribute images.
- Download (no login): milk10k.zip + milk10k.csv + supplements/training_gt.csv from
  `https://isic-archive.s3.amazonaws.com/dois/10.34970-648456/`. Record sha256 in DATA_MANIFEST.json.
- Labels (lesion-level, training_gt.csv): BCC→BCC, NV→NV, MEL→MEL, BKL→BKL, DF→DF, VASC→VASC,
  SCCKA→SCC, AKIEC→AK. Drop INF (50), BEN_OTH (44), MAL_OTH (9).
- Known mapping difference: MILK AKIEC includes SCC in situ/Bowen's, whereas ISIC 2019 files
  Bowen's under SCC.
- Overlap with ISIC 2019 (mandatory): for every MILK lesion_id query
  `https://api.isic-archive.com/api/v2/lesions/{lesion_id}/` and collect all isic_ids; any lesion
  with an image in ISIC_2019_Training_GroundTruth.csv is removed from MILK (all splits).
  Expected (box check 2026-11-16): 859 of 5,240 lesions, 1,895 ISIC 2019 images, all HAM10000
  (HAM_ lesion ids); remaining mapped ≈ 4,278 (MEL 230, NV 347, BCC 2,429, AK 293, BKL 442,
  DF 33, VASC 31, SCC 473). Report any difference before continuing.
  API: ≤ 2 req/s; cache under `results/paperB/r14_pair3/cache/isic_api/`; generic User-Agent
  (`python-requests`); no email or personal name in headers.
- Patient overlap: MILK Vienna may share patients with HAM10000 with no patient id to remove them;
  §2.3 removes only shared lesions. milk_heldout is not part of the verdict, so the verdict is
  unaffected, but AUROC on milk_heldout may be biased toward ID.
- Near-duplicate: pHash 64-bit Hamming ≤ 6 of MILK dermoscopic vs all ISIC 2019 and vs
  PAD-UFES-20, Fitzpatrick17k, MED-NODE, SD-198. Drop MILK matches to ISIC 2019; list (do not
  drop) matches to OOD sets. Write OVERLAP.md + overlap_lesions.csv.
- Split MILK by lesion_id (seed 42): milk_adv 70 %, milk_heldout 30 %. No patient id: lesion-
  disjoint = image-disjoint; patient overlap between milk_adv and milk_heldout cannot be excluded.

## Recipe / grid (20 runs)

Main recipe exactly as Table 1 / R12 architecture: dual EffNet-B3, z_l 16-d BN, z_c 64-d,
40 epochs, cls_lr_mult as main (`r6_common.CURRENT`), lesion-disjoint ISIC 2019 split seed 42,
last.ckpt primary, best.ckpt also scored.
Streams: isic_cls, isic_adv (**resampled to prior**), milk_adv.
Seeds 42, 52, 62, 72, 82.
- λ = 0 × 5 seeds (prior100)
- λ = 2 × {prior100, prior50, prior0} × 5 seeds
Tags: `r14_ladv0_prior100_s{seed}`, `r14_ladv2_prior{100,50,0}_s{seed}`.
Checkpoints: `checkpoints/r14_pair3/<tag>/`.
Run JSON: `results/paperB/r14_pair3/runs/<tag>/summary.json`.
NaN guard: abort a run and mark it `degenerate` if any feature or loss is NaN/Inf; keep its log.

## Early-abort stage

Stage 1 first: `r14_ladv0_prior100_s42`, `r14_ladv2_prior100_s42`, `r14_ladv2_prior100_s52`
(+ `prior0_s42` if a GPU is free). Score immediately (V1 + Mahalanobis only). Write STAGE1.md:

- STOP-gap: V1 > 0.70 at λ = 2 prior100 on both s42 and s52 → stop, no further jobs.
- STOP-headroom: at λ = 0, Mahalanobis AUROC < 0.65 on all of Fitzpatrick17k, MED-NODE, SD-198 → stop.
- STOP-degenerate: ≥ 1 of the stage-1 runs degenerate or ID bal acc < 0.25 → stop and report the trace.
- Otherwise submit the remaining 16–17 runs (array %2). Do not look at any AUROC beyond these rules to retune.

## Scoring

Features from z_l^norm (dual latent). Detectors fit on ISIC train only: Mahalanobis class-
conditional (+1e-3·I) primary; kNN-50, cosine, MSP, energy reported. ID = ISIC test.
OOD: milk_heldout (aligned, like pad_heldout), PAD-UFES-20 all 2,298 (now unseen),
Fitzpatrick17k, MED-NODE, SD-198, SCIN.
Leakage V1: linear domain probe on held-out images, ISIC test ∪ milk_heldout, GroupShuffleSplit
by lesion (Table 1 held-out V1). Also class-aware (within-class) leakage on shared classes
(same probes as R6 class_aware_heldout).
Bootstrap 2,000 draws, rng 20261005; units: lesion (milk_heldout), patient (PAD), P13 cluster
(Fitzpatrick17k), cluster_union (SD-198), image (MED-NODE), case (SCIN).
Rule A = pooled CI upper < 0.5 and ≥ 3 seeds with their own CI upper < 0.5.
Degraded = ID test balanced accuracy < 0.25: report, exclude from means/CI/verdict.
Also per run: ID bal acc, nearest-class share and NV-nearest fraction on each OOD set,
predicted-majority share on milk_heldout. V1 for every cell including λ = 0 and prior0/prior50.
If a site field exists, Mahalanobis AUROC on milk_heldout per site (descriptive only).

## Reading rules (λ = 2)

Invariant = V1 ≤ 0.60 at prior100 (seed mean). Competent = ID bal acc within 0.03 of this pair's
λ = 0 cell (also report vs ERM 0.558 on the same split). Not invariant → Case C, settles nothing.
Small-gap check: if V1 at λ = 0 (seed mean) is already ≤ 0.60, say so first in REPORT.md and state
that the invariance gate says nothing about what the adversary did in this pair; the verdict below
is still computed.

- Rule A on ≥ 2 of the 3 labelled unseen sets (Fitzpatrick17k, MED-NODE, SD-198) at prior100 and
  on none at prior0: inversion reproduced, label shift necessary in this pair — read this way only
  if V1 at λ = 2 prior0 (seed mean) is also ≤ 0.60. If prior0 is not invariant, the verdict is
  "reproduced; necessity not separated from invariance depth" (consistent with appendix E of the
  paper; R12 prior 0 % leakage rose to 0.757, X4 shift 0 % to 0.671).
- Met at prior100 and also at prior0: reproduced, label shift not necessary here.
- Met on < 2 at prior100: not reproduced in this pair.
- milk_heldout, PAD-UFES-20 and SCIN reported, not counted. prior50 reported as the dose step.

Report every run and number as found, nulls included. No wording beyond these rules.

## Outputs

`results/paperB/r14_pair3/` only: DATA_MANIFEST.json, OVERLAP.md, PRIORS.json, STAGE1.md,
STATUS.md, CELLS.json, RESULTS.json, REPORT.md. Commit and push as each item finishes.
Scope: rebuttal / camera-ready / arXiv; submitted r12c manuscript is not edited.

## Deviations

_(append only)_

- 2026-11-18: Prepare job 74360 (after cancelled 74344). ISIC API overlap exact match to box check (859 / 1895 / remaining counts). Parallel cache fill ≤2 req/s (ThreadPool 4). pHash dropped 67 additional MILK dermoscopic images vs ISIC 2019 (227 hash hits); listed 7 PAD / 0 Fitz / 0 MED / 6 SD-198. PRIORS refreshed after pHash drops. TV(ISIC train, milk_adv)=0.563. attribution uniform — no per-site AUROC. Stage-1 array 74361.

- 2026-11-18: Stage-1 array 74361 COMPLETED. Gate **STOP-degenerate**: λ=0 prior100 s42 ID bal 0.185 < 0.25 (also prior0 s42 ID 0.152). λ=2 prior100 s42/s52 ID≈0.45, V1≈0.616 (no STOP-gap). No further jobs. REPORT/CELLS/RESULTS written for abort.

- 2026-11-18: CPU diagnosis (DIAG_cpu_20261118.md): no I/O or light_extract bug. R14 λ0/prior0 diverge (loss_cls↑ loss_ctx↑); R12 λ0 and R14 λ2 prior100 healthy on same recipe/MILK files. Before any GPU rerun: schedule one determinism redo of r14_ladv0_prior100_s42 only; if ID bal still <0.25 → stop R14 as "pipeline unstable on this pair, not a test". Thresholds unchanged.

- 2026-11-18: Determinism redo λ0 s42 job **74376**: ID bal **0.475** (first run 0.185), traj matches R12 (loss_cls/ctx↓). Not degenerate → STAGE1 **CONTINUE**. Thresholds unchanged. Archived first λ0 + degraded prior0_s42; submitting remaining train array %2.

- 2026-11-18 (locked **before** reading train array 74399 or score job 74400; jobs not cancelled):

  1. **Rerun rule (every run in the 20-run grid).** Any run with ID test balanced accuracy < 0.25 is rerun exactly once with the same seed. Decision uses ID only — never V1 or OOD. Report both attempts (first marked "degenerate, superseded"). If the rerun is also < 0.25, exclude under the degraded rule; that cell then has 4 seeds and Rule A still needs k ≥ 3. Explicitly under this rule: `r14_ladv0_prior100_s42` (0.185 → 0.475) and `r14_ladv2_prior0_s42` (0.152, archived; retrained by the `--skip_done` array). Do not rerun any run for any other reason.

  2. **Determinism.** Same seed does not give the same result on this cluster (λ0 s42: 0.185 vs 0.475). REPORT.md must include: "Seeds are not bitwise reproducible (GPU non-determinism)." Report the first-attempt degenerate count over the whole grid (stage 1: 2 of 4) and, for each degenerate run, whether it diverged (loss_ctx/loss_cls trajectory).

  3. **Honest stage-1 stop.** REPORT.md must say: the stage-1 STOP-degenerate rule triggered; R14 continued only under the 2026-11-18 deviation after a CPU diagnosis and a determinism redo. Thresholds unchanged. Do not describe stage 1 as passed.

  4. **Reading order (strict).** Verdict by reading rules in this order only: (a) V1 at λ=0; (b) invariance at λ=2 prior100 (seed mean ≤ 0.60, else Case C, no reading of AUROC); (c) V1 at prior0 for the "label shift necessary" clause; (d) Rule A. Report every run as found.

- 2026-11-19: Backfill λ0 s42 pad_ufes20/scin (job 74481); full rescore → λ=0 cell n=5. REPORT includes honest stage-1 stop, determinism, rerun rule (prior0 s42 0.152→0.480), reading order (a–d). Verdict Case C unchanged.
