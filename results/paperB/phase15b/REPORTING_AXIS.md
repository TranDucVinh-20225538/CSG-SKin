# Phase 15b — reporting axis

Written before Item A and Item B numbers. Nominal weights are not comparable across losses.

## X axis

Plot and tabulate OOD AUROC against **achieved leakage**, not against the penalty weight or λ.

## Two leakage numbers when K > 2

| Probe | Metric | Floor | Use |
|---|---|---|---|
| K-way domain probe on the training domains | balanced accuracy | 1/K | What the adversary is fighting |
| Binary train-vs-OOD-domain probe | balanced accuracy | 0.5 | Comparable to dermatology; this is the x-axis of the cross-setting figure |

Dermatology stays 2-class, so the two probes coincide and the floor is 0.5.

## Two supervision blocks, never one table

| Block | Objectives | PAD labels |
|---|---|---|
| Unlabeled invariance | ERM, CORAL, MMD, DANN | never in L_cls |
| Labeled environments | IRM, GroupDRO | pad_adv labels enter the risk |

IRM/GroupDRO cross-domain accuracy 0.60–0.64 against 0.32–0.40 in the unlabeled block is label access, not generalisation. Do not put them in one block.
