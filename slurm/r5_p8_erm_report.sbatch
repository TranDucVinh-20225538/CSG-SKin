#!/bin/bash
#SBATCH --job-name=pB-p8ermR
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p8_erm_report_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p8_erm_report_%j.err
# Bootstrap + erm_ref/REPORT.md, then regenerate P8 REPORT.md (competence note + 5-seed ERM).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== P8 ERM report start $(date) ==="
"${ROOT}/.venv/bin/python" scripts/r5_erm_lesion_ref.py report
"${ROOT}/.venv/bin/python" scripts/r6_report.py --item p8
echo "=== done $(date) ==="
