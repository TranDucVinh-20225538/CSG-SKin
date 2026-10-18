#!/bin/bash
#SBATCH --job-name=pB-p10-last
#SBATCH --array=0-14%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_posthoc_last_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_posthoc_last_%A_%a.err
# Retrain pilot (0-4) + shift_arms (5-14) with --force; save last.pt + summary.posthoc_last.
# Descriptive only; does not change locked gate / best-epoch selection rule.
# Prefer afterany on remaining R2 so this does not steal vip2 GPUs from 68562.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
T=${SLURM_ARRAY_TASK_ID}
PY="${ROOT}/.venv/bin/python"
if (( T <= 4 )); then
  RUNGS=(P_L1 P_L2 P_L3 P_L4 P_L5)
  RUNG=${RUNGS[$T]}
  echo "=== posthoc last pilot rung=${RUNG} $(date) ==="
  "$PY" scripts/train_r5_p10_camelyon_toolbox.py \
    --pilot --rung "${RUNG}" --seed 42 --recipe current --force --num_workers 6
else
  SEEDS=(42 52 62 72 82)
  U=$((T - 5))
  SEED=${SEEDS[$((U / 2))]}
  if (( U % 2 == 0 )); then RUNG=natural_l2; else RUNG=induced_l2; fi
  echo "=== posthoc last shift_arm rung=${RUNG} seed=${SEED} $(date) ==="
  "$PY" scripts/train_r5_p10_camelyon_toolbox.py \
    --shift_arm --rung "${RUNG}" --seed "${SEED}" --recipe current --force --num_workers 6
fi
echo "=== done $(date) ==="
