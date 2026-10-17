#!/bin/bash
#SBATCH --job-name=pB-r1-abc
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r1_after_s82_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r1_after_s82_%j.err
# After R1 task 17 (shift0 λ0.5 s82): release R2 hold, refresh R1 report, run leakage-matched (a)/(b)/(c).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"

echo "=== release R2 68562 hold $(date) ==="
scontrol release 68562 || true

echo "=== R1 report $(date) ==="
${PY} scripts/r6_report.py --item r1

echo "=== leakage-matched (a)/(b)/(c) $(date) ==="
${PY} scripts/r6_r1_leakage_matched.py

echo "=== done $(date) commit=$(git rev-parse --short HEAD) ==="
