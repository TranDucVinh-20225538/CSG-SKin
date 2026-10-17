#!/bin/bash
#SBATCH --job-name=paperB-r5-p10
#SBATCH --array=0-9%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_shift_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_shift_%A_%a.err
# P10 label-shift arms (decisions/paperB_r5_p10.md amendment 2026-10-17).
# tasks 0-4 induced_l2 seeds 42..82; 5-9 natural_l2 seeds 42..82. One GPU (%1).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
T=${SLURM_ARRAY_TASK_ID}
SEEDS=(42 52 62 72 82)
if (( T < 5 )); then RUNG=induced_l2; SEED=${SEEDS[$T]}; else RUNG=natural_l2; SEED=${SEEDS[$((T-5))]}; fi
echo "=== P10 shift_arm rung=${RUNG} seed=${SEED} task=${T} $(date) ==="
"${ROOT}/.venv/bin/python" scripts/train_r5_p10_camelyon_toolbox.py \
  --rung "${RUNG}" --seed "${SEED}" --shift_arm --recipe "${RECIPE:-current}" \
  --skip_done --num_workers 6
echo "=== done $(date) ==="
