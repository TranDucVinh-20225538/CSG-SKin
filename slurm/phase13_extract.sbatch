#!/bin/bash
#SBATCH --job-name=paperB-p13ext
#SBATCH --array=0-26
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p13ext_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p13ext_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== P13 extract task ${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
${PY} ${ROOT}/scripts/extract_phase13_features.py --task_id "${SLURM_ARRAY_TASK_ID}" --skip_done --cpu --num_workers 4 --batch_size 32
echo "=== P13 extract task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
