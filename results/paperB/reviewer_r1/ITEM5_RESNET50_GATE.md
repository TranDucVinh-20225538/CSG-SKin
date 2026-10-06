# ResNet-50 Item 5 — adversary gate (rescored from logs)

- `resnet50_single_dann_ladv0_s42`: **valid_control** — λ=0 control. Adversary loss weight is zero, so the head is not trained. Not a null experiment.
- `resnet50_single_dann_ladv0_s52`: **valid_control** — λ=0 control. Adversary loss weight is zero, so the head is not trained. Not a null experiment.
- `resnet50_single_dann_ladv0_s62`: **valid_control** — λ=0 control. Adversary loss weight is zero, so the head is not trained. Not a null experiment.
- `resnet50_single_dann_ladv0p5_s42`: **valid** — adversary learned: best-epoch CE 0.4146 (epoch 1) is below ln K − 0.02.
- `resnet50_single_dann_ladv0p5_s52`: **valid** — adversary learned: best-epoch CE 0.4683 (epoch 1) is below ln K − 0.02.
- `resnet50_single_dann_ladv0p5_s62`: **valid** — adversary learned: best-epoch CE 0.5469 (epoch 1) is below ln K − 0.02.
- `resnet50_single_dann_ladv2_s42`: **inconclusive** — adversary never left chance: best-epoch CE 0.6864 is not below ln K − 0.02 = 0.6731. Null experiment, not a scientific negative.
- `resnet50_single_dann_ladv2_s52`: **inconclusive** — adversary never left chance: best-epoch CE 0.6882 is not below ln K − 0.02 = 0.6731. Null experiment, not a scientific negative.
- `resnet50_single_dann_ladv2_s62`: **inconclusive** — adversary never left chance: best-epoch CE 0.6860 is not below ln K − 0.02 = 0.6731. Null experiment, not a scientific negative.
