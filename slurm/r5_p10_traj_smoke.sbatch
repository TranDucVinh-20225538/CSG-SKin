#!/bin/bash
#SBATCH --job-name=pB-p10-traj2
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_traj2_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_traj2_%A_%a.err
# Two-run trajectory smoke BEFORE remaining R2: (0) natural_l2 s42, (1) pilot P_L4 s42.
# Per-epoch fast site probe + ID bal; last.pt + posthoc_last. Does not change locked gate.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
T=${SLURM_ARRAY_TASK_ID}
if (( T == 0 )); then
  echo "=== traj smoke natural_l2 s42 $(date) ==="
  "$PY" scripts/train_r5_p10_camelyon_toolbox.py \
    --shift_arm --rung natural_l2 --seed 42 --recipe current \
    --force --epoch_traj --num_workers 6
else
  echo "=== traj smoke pilot P_L4 s42 $(date) ==="
  "$PY" scripts/train_r5_p10_camelyon_toolbox.py \
    --pilot --rung P_L4 --seed 42 --recipe current \
    --force --epoch_traj --num_workers 6
fi
echo "=== done $(date) ==="
