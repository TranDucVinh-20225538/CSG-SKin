#!/bin/bash
#SBATCH --job-name=paperB-finalagg
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/finalagg_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/finalagg_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== final aggregate start $(date) ==="
${PY} ${ROOT}/scripts/aggregate_final_package.py
echo "=== final aggregate done $(date) ==="
