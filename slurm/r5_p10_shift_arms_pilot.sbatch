#!/bin/bash
#SBATCH --job-name=paperB-p10-arms
#SBATCH --array=0-9%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_arms_pilot_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_arms_pilot_%A_%a.err
# Full Camelyon 10 runs at the pilot-selected level (5 seeds × natural/induced).
# Requires SELECTED_RUNG=P_L* (export from after-pilot dispatch).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
if [[ -z "${SELECTED_RUNG:-}" ]]; then
  echo "SELECTED_RUNG unset" >&2
  exit 1
fi
T=${SLURM_ARRAY_TASK_ID}
SEEDS=(42 52 62 72 82)
SEED=${SEEDS[$((T / 2))]}
if (( T % 2 == 0 )); then ARM=natural; else ARM=induced; fi
RUNG="${ARM}_${SELECTED_RUNG}"
echo "=== P10 pilot arms rung=${RUNG} seed=${SEED} task=${T} $(date) ==="
"${ROOT}/.venv/bin/python" scripts/train_r5_p10_camelyon_toolbox.py \
  --rung "${RUNG}" --seed "${SEED}" --shift_arm --recipe "${RECIPE:-current}" \
  --skip_done --num_workers 6
echo "=== done $(date) ==="
