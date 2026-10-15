#!/bin/bash
#SBATCH --job-name=paperB-r6-r1
#SBATCH --array=0-34%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6r1_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6r1_%A_%a.err
# R6 R1 dose test, 35 runs (scripts/train_r6_dose.py --list). RECIPE (env): current | p8 | auto (default auto).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
RECIPE="${RECIPE:-auto}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R6 R1 recipe=${RECIPE} task=${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" scripts/train_r6_dose.py --recipe "${RECIPE}" --task_id "${SLURM_ARRAY_TASK_ID}" \
  --resume --skip_done --num_workers 6
echo "=== done $(date) ==="
