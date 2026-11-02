#!/bin/bash
#SBATCH --job-name=paperB-r9-e1-fin
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r9e1_fin_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r9e1_fin_%j.err
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
"${ROOT}/.venv/bin/python" scripts/r9_e1_report.py
# within-class held-out probe (a/b/c) including E1 cells
"${ROOT}/.venv/bin/python" scripts/r6_class_aware_heldout_probes.py || true
"${ROOT}/.venv/bin/python" scripts/r9_finalize.py
echo "E1 finalize done $(date)"
