#!/bin/bash
#SBATCH --job-name=paperB-r6-r2
#SBATCH --array=0-39%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6r2_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6r2_%A_%a.err
# R6 R2 CDAN/IWDAN/IWCDAN (30) + IWDAN oracle (10) = 40 runs. RECIPE (env): current | p8 | auto.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
RECIPE="${RECIPE:-auto}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R6 R2 recipe=${RECIPE} task=${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" scripts/train_r6_iwdan.py --recipe "${RECIPE}" --task_id "${SLURM_ARRAY_TASK_ID}" \
  --resume --skip_done --num_workers 6
echo "=== done $(date) ==="
