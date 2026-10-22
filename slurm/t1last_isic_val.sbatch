#!/bin/bash
#SBATCH --job-name=paperB-t1val
#SBATCH --array=0-9%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/t1val_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/t1val_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
mkdir -p logs
"${ROOT}/.venv/bin/python" scripts/extract_t1last_isic_val.py --task_id "${SLURM_ARRAY_TASK_ID}" --device cuda --num_workers 4
