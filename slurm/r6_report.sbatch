#!/bin/bash
#SBATCH --job-name=paperB-r6-report
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6report_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6report_%j.err
# ITEM (env): p8 | r1 | r2 | p7b.  RECIPE (env): current | p8 | auto (ignored for p8).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== report item=${ITEM} recipe=${RECIPE:-auto} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" scripts/r6_report.py --item "${ITEM}" --recipe "${RECIPE:-auto}"
echo "=== done $(date) ==="
