#!/bin/bash
#SBATCH --job-name=paperB-r5-s-extract
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5s_extract_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5s_extract_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R5 S extract worker ${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" scripts/r5_s_extract.py "${SLURM_ARRAY_TASK_ID}" 2
echo "=== done $(date) ==="
