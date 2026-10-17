#!/bin/bash
#SBATCH --job-name=paperB-p10-pilot
#SBATCH --array=0-4%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_pilot_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_pilot_%A_%a.err
# Camelyon adversary ladder pilot (decisions/paperB_r5_p10_ladder_pilot.md). One GPU; stop via --skip inside script.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
RUNGS=(P_L1 P_L2 P_L3 P_L4 P_L5)
RUNG=${RUNGS[${SLURM_ARRAY_TASK_ID}]}
echo "=== P10 pilot rung=${RUNG} task=${SLURM_ARRAY_TASK_ID} $(date) ==="
"${ROOT}/.venv/bin/python" scripts/train_r5_p10_camelyon_toolbox.py \
  --pilot --rung "${RUNG}" --seed 42 --recipe "${RECIPE:-current}" --skip_done --num_workers 6
echo "=== done $(date) ==="
