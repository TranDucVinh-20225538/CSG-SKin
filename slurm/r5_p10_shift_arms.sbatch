#!/bin/bash
#SBATCH --job-name=paperB-r5-p10
#SBATCH --array=0-9%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_shift_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_shift_%A_%a.err
# P10 label-shift arms (decisions/paperB_r5_p10.md amendment 2026-10-17).
# Interleaved by seed: natural s42, induced s42, natural s52, induced s52, ... (both GPUs %2).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
T=${SLURM_ARRAY_TASK_ID}
SEEDS=(42 52 62 72 82)
SEED=${SEEDS[$((T / 2))]}
if (( T % 2 == 0 )); then RUNG=natural_l2; else RUNG=induced_l2; fi
echo "=== P10 shift_arm rung=${RUNG} seed=${SEED} task=${T} $(date) ==="
"${ROOT}/.venv/bin/python" scripts/train_r5_p10_camelyon_toolbox.py \
  --rung "${RUNG}" --seed "${SEED}" --shift_arm --recipe "${RECIPE:-current}" \
  --skip_done --num_workers 6
echo "=== done $(date) ==="
