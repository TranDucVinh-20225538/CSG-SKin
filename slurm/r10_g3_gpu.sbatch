#!/bin/bash
#SBATCH --job-name=paperB-r10-g3
#SBATCH --array=0-4%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r10g3_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r10g3_%A_%a.err
# R10 G3: IWDAN n50 λ=0.5 × seeds 42–82.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R10 G3 task=${SLURM_ARRAY_TASK_ID} start $(date) ==="
"${ROOT}/.venv/bin/python" scripts/train_r10_g3.py --task_id "${SLURM_ARRAY_TASK_ID}" --resume --skip_done --num_workers 6
echo "=== done $(date) ==="
