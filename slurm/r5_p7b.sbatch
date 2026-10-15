#!/bin/bash
#SBATCH --job-name=paperB-r5-p7b
#SBATCH --array=0-14%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p7b_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p7b_%A_%a.err
# R5 P7(b) PAD held out of the adversary (BCN vs HAM), 15 runs (scripts/train_r5_p7b.py --list). RECIPE (env).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
RECIPE="${RECIPE:-auto}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R5 P7b recipe=${RECIPE} task=${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" scripts/train_r5_p7b.py --recipe "${RECIPE}" --task_id "${SLURM_ARRAY_TASK_ID}" \
  --resume --skip_done --num_workers 6
echo "=== done $(date) ==="
