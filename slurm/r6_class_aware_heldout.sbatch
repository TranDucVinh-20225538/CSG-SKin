#!/bin/bash
#SBATCH --job-name=pB-clsAware
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6_class_aware_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6_class_aware_%j.err
# Class-aware held-out probes on existing features (partial oracle OK).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== class-aware heldout start $(date) commit=$(git rev-parse --short HEAD) ==="
"${ROOT}/.venv/bin/python" scripts/r6_class_aware_heldout_probes.py
echo "=== done $(date) ==="
