#!/bin/bash
#SBATCH --job-name=paperB-p13agg
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p13agg_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p13agg_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== P13 analyze start $(date) ==="
${PY} ${ROOT}/scripts/analyze_phase13.py
echo "=== P13 analyze done $(date) ==="
