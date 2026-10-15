#!/bin/bash
#SBATCH --job-name=paperB-r5-p8-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p8_cpu_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p8_cpu_%j.err
# STAGE (env): selA | selB (recipe selection, seconds) | report (P8 REPORT.md + RECIPE_DECISION.json).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== R5 P8 cpu stage=${STAGE} start $(date) host=$(hostname) ==="
if [[ "${STAGE}" == "report" ]]; then
  "${ROOT}/.venv/bin/python" scripts/r6_report.py --item p8
else
  "${ROOT}/.venv/bin/python" scripts/train_r5_p8.py --stage "${STAGE}"
fi
echo "=== done $(date) ==="
