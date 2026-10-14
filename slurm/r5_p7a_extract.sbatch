#!/bin/bash
#SBATCH --job-name=paperB-r5-p7a
#SBATCH --array=0-34%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p7a_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p7a_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R5 P7a extract task ${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" scripts/r5_p7a_extract.py "${SLURM_ARRAY_TASK_ID}"
echo "=== done $(date) ==="
