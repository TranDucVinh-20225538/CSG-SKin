#!/bin/bash
#SBATCH --job-name=paperB-r2-r50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r2_r50_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r2_r50_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R2 ImageNet R50 extract start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" scripts/r2_extract_imagenet_r50.py
echo "=== done $(date) ==="
