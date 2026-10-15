#!/bin/bash
# Paper B R5 Q3 launcher (decisions/paperB_r5_q3.md). Prepared under the compute HOLD; not executed.
# Usage: GO=1 [RECIPE=current|p8] [VIT=dinov2_vits14|dinov2_vitb14] bash scripts/launch_r5_q3.sh [AFTER_JOBID]
# Arms: dual ViT (10), single ResNet-50 λ ∈ {0, 2} (10; written into p9/stage2 and shared with P9
# stage 2: P9's selected rung if SELECTED_resnet50.json exists, else the baseline config) -> report.
# ConvNeXt-T arm dropped by the user (2026-10-15): ResNet-50 is the second CNN; the open question is a transformer.
# DINOv2-S/14 weights are not in the torch hub cache. Fetch them once on the login node before launching:
#   .venv/bin/python -c "import torch; torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth')"
# or run with VIT=dinov2_vitb14 (cached) and log the substitution as a deviation.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs
if [[ "${GO:-0}" != "1" ]]; then echo "compute HOLD: set GO=1 to submit"; exit 1; fi
RECIPE="${RECIPE:-current}"
VIT="${VIT:-dinov2_vits14}"
CK="${HOME}/.cache/torch/hub/checkpoints"
for W in "${VIT}_pretrain.pth"; do
  [[ -f "${CK}/${W}" ]] || { echo "missing pretrained weights ${CK}/${W}; see the header of this script"; exit 1; }
done
DEP=()
if [[ -n "${1:-}" ]]; then DEP=(--dependency=afterany:$1); fi
V=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-9%2 --export=ALL,STAGE=q3,BACKBONE=${VIT},RECIPE=${RECIPE} slurm/r5_dual_gpu.sbatch)
S=$(sbatch --parsable --dependency=afterany:${V} --array=0-9%2 --export=ALL,STAGE=q3_r50,BACKBONE=resnet50,RECIPE=${RECIPE} slurm/r5_single_gpu.sbatch)
RP=$(sbatch --parsable --dependency=afterany:${S} --export=ALL,CMD=q3_report,RECIPE=${RECIPE} slurm/r5_cpu.sbatch)
echo "Q3 ${VIT}=${V} resnet50=${S} report=${RP}"
