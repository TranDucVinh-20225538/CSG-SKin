#!/bin/bash
#SBATCH --job-name=paperB-p16b1
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase16_b1_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase16_b1_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"
cd "${ROOT}"
LAMBDAS=(0 0.1 0.25 0.5 1 2 4 8)
LADV="${LAMBDAS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Phase 16 B1 coarse λ=${LADV} task=${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
"${PY}" "${ROOT}/scripts/phase16_b1_coarse.py" \
  --lambda_adv "${LADV}" \
  --seed 42 \
  --image_size 448 \
  --batch_size 32 \
  --num_workers 4 \
  --max_epochs 40 \
  --skip_done
echo "=== Phase 16 B1 λ=${LADV} done $(date) ==="
