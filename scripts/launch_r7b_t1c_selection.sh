#!/bin/bash
# Paper B R7b T1c selection: 4 GPU runs (ERM×{0.3,1.0} + dual λ0×{0.3,1.0}), seed 42.
# Usage: GO=1 bash scripts/launch_r7b_t1c_selection.sh
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs results/paperB/r7b/t1c
if [[ "${GO:-0}" != "1" ]]; then echo "set GO=1 to submit"; exit 1; fi
RECIPE="${RECIPE:-current}"
VIT="${VIT:-dinov2_vits14}"
DUAL=$(sbatch --parsable --array=0-1%2 --export=ALL,STAGE=t1c_sel,BACKBONE=${VIT},RECIPE=${RECIPE} slurm/r7b_dual_gpu.sbatch)
ERM=$(sbatch --parsable --array=0-1%2 --export=ALL,STAGE=t1c_sel,BACKBONE=${VIT},RECIPE=${RECIPE} slurm/r7b_single_gpu.sbatch)
echo "T1c selection dual=${DUAL} erm=${ERM}"
echo "After both finish: .venv/bin/python scripts/r7b_t1c_select.py && GO=1 bash scripts/launch_r7b_t1c_main.sh"
