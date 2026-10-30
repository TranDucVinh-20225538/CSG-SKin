#!/bin/bash
# Paper B R7 T1 launcher (decisions/paperB_r7_t1.md).
# Usage: GO=1 [VIT=dinov2_vits14|dinov2_vitb14] bash scripts/launch_r7_t1.sh
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs results/paperB/r7/t1
if [[ "${GO:-0}" != "1" ]]; then echo "set GO=1 to submit"; exit 1; fi
RECIPE="${RECIPE:-current}"
VIT="${VIT:-dinov2_vits14}"
CK="${HOME}/.cache/torch/hub/checkpoints"
[[ -f "${CK}/${VIT}_pretrain.pth" ]] || { echo "missing ${CK}/${VIT}_pretrain.pth"; exit 1; }
DUAL=$(sbatch --parsable --array=0-9%2 --export=ALL,STAGE=t1_dual,BACKBONE=${VIT},RECIPE=${RECIPE} slurm/r7_dual_gpu.sbatch)
ERM=$(sbatch --parsable --array=0-4%2 --export=ALL,STAGE=t1_erm,BACKBONE=${VIT},RECIPE=${RECIPE} slurm/r7_single_gpu.sbatch)
echo "T1 dual=${DUAL} erm=${ERM} backbone=${VIT} (ERM 5 seeds)"
