# CONFIRMATION — Phase 12.0 accepted, 12.1 authorised with amendments

Phase 12.0 report accepted. Split sizes, hospital assignments and class counts match the published WILDS figures exactly, so the local copies are the correct releases.

## Confirmations

1. **Use the existing DST-Skin copies.** No re-download. Two conditions: record the absolute data root and the `RELEASE_*` marker contents in the run config JSON so results stay reproducible if that tree moves, and treat it strictly read-only. It belongs to another project and must not be modified.
2. **Camelyon ID eval = `id_val`.** Confirmed — there is no `id_test`. Do not construct one. See amendment A.
3. **Checkpoint on `id_val`, never on `val`.** Confirmed, and your reasoning is right: selecting on hospital 1 would peek at an unseen hospital and contaminate the very measurement this phase exists to make. See amendment D.
4. **GPU priority.** See amendment E — the coarse scan starts now, the dense sweep waits.

---

## Amendment A — split `id_val` in two

`id_val` is currently doing two jobs: checkpoint selection and the ID reference distribution for OOD scoring. A checkpoint chosen to perform well on `id_val` gives those samples lower Mahalanobis distances, which inflates ID/OOD separation.

The bias is conservative — it pushes AUROC *up*, and the finding is that AUROC collapses — but it is free to remove. Split `id_val` (33,560, ample) into two disjoint halves, stratified by hospital and label:

- `id_val_select` — checkpoint selection only
- `id_val_score` — ID reference for all OOD detectors and ID accuracy

Fix the split with a recorded seed. Report both halves' sizes.

## Amendment B — Camelyon17 is binary; expect the detector suite to behave differently

The derm task had 8 classes. Camelyon17 has 2. This changes what each detector can do:

- **MSP and Energy are weak on a binary task** — MSP is confined to [0.5, 1] and carries much less information than over 8 classes. Do not read a flat or noisy MSP curve as a failure to replicate.
- **Lead with Mahalanobis (class-conditional, shared covariance) and kNN (k=50)** for Camelyon17. These are the detectors the derm conclusion actually rested on.
- Report MSP/Energy for completeness with an explicit note that the binary label space limits them.

State this in the report before presenting the numbers, so the comparison to derm is read correctly.

## Amendment C — build the class-composition control into iWildCam from the start

Phase 2.5b ruled out class composition as the driver of the derm inversion. That confound returns on iWildCam and must be handled in the first run, not bolted on later:

- `id_test` has 87 classes present; OOD `test` has 102. The class distributions differ.
- Compute OOD AUROC **both** unrestricted **and** restricted to classes present in both splits, exactly as in Phase 2.5b.

Camelyon17 does not need this — train, `id_val` and `test` are all balanced 50/50, so the composition is matched by construction. That makes Camelyon17 the cleaner of the two; say so.

Also for iWildCam: with 182 classes, many sparsely populated, per-class covariance estimates will be poorly conditioned. Keep the shared covariance as specified, and add **class-agnostic (single-Gaussian) Mahalanobis** as a co-primary detector alongside kNN.

## Amendment D — two never-adversarial domains, and a comparability note

**The free bonus:** the adversary sees only hospitals 0, 3, 4. Hospitals 1 (`val`) and 2 (`test`) are both domains it never saw. This reproduces the Fitzpatrick17k control — where `z_lesion` scored 0.399 on a domain never in `L_adv` — natively inside the official splits, and gives two such domains rather than one.

Report OOD AUROC and cross-domain accuracy for **both** hospitals as separate columns at every λ. If the cliff tracks on both, that is the Camelyon17 analogue of the Fitz result on a different modality.

**Comparability caveat:** published WILDS Camelyon17 baselines select checkpoints on the OOD `val` split. You are deliberately not doing that. Your accuracy numbers will therefore not be directly comparable to the leaderboard. This is the correct choice for this experiment, but it must be stated explicitly wherever leaderboard context is given — do not present the numbers as leaderboard-comparable.

## Amendment E — GPU priority

The derm final package (array `60585`) has priority; it is what makes the paper shippable.

**Exception: launch the 4-run coarse λ scan now**, sharing the QOS cap. It is one seed over `{0, 0.1, 1, 10}`, it is the highest-information and lowest-cost step in the phase, and it determines the entire dense grid. It will not meaningfully delay the derm array.

**Hold the dense sweep** until `60585` releases its cards.

## Amendment F — verify all file paths before launching an array

The 200-file spot check is thin against 455,956 and 217,644 files. Before submitting any multi-hour array, run `os.path.exists` over every path in both metadata CSVs — a few minutes of I/O that prevents a missing-file crash hours into a job. Report the count of any missing files.

---

## Launch order

1. Amendment F path verification (minutes).
2. Amendment A `id_val` split, recorded.
3. Coarse λ scan on Camelyon17, 1 seed, `{0, 0.1, 1, 10}` — **start now**.
4. Report where the transition sits, then stop for the dense-grid design.

Extend the grid **downward** if the collapse is already complete at λ=0.1. A cliff at λ=0.01 is a stronger result than one at λ=0.25, not a weaker one.

Standing rules unchanged: report failures as findings, no retuning, outputs only under `results/paperB/phase12/`.
