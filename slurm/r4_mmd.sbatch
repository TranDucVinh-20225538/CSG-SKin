#!/bin/bash
#SBATCH --job-name=paperB-r4-mmd
#SBATCH --array=0-23
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r4mmd_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r4mmd_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== R4 MMD task ${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" scripts/r4_mmd_pad_heldout.py "${SLURM_ARRAY_TASK_ID}"
echo "=== done $(date) ==="
