#!/bin/bash
#SBATCH --job-name=paperB-p13toy
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p13toy_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p13toy_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== P13 toy start $(date) ==="
${PY} ${ROOT}/scripts/train_phase13_toy.py
echo "=== P13 toy done $(date) ==="
