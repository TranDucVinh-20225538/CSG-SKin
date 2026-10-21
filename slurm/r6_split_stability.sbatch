#!/bin/bash
#SBATCH --job-name=paperB-ss
#SBATCH --array=0-17%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6ss_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6ss_%A_%a.err
# R6 split-stability: 18 runs (decisions/paperB_r6_split_stability.md).
# Seed-42 hash gate runs inside each task before train.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R6 split_stability task=${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"${ROOT}/.venv/bin/python" scripts/r6_split_stability.py verify42
"${ROOT}/.venv/bin/python" scripts/r6_split_stability.py train --task_id "${SLURM_ARRAY_TASK_ID}" --skip_done --resume --num_workers 6
echo "=== done $(date) ==="
