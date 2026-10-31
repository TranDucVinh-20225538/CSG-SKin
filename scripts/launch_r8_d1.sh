#!/bin/bash
# Paper B R8 D1 launcher. Usage: GO=1 bash scripts/launch_r8_d1.sh
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs results/paperB/r8/d1
if [[ "${GO:-0}" != "1" ]]; then echo "set GO=1 to submit"; exit 1; fi
RECIPE="${RECIPE:-current}"
J=$(sbatch --parsable --array=0-26%2 --export=ALL,RECIPE=${RECIPE} slurm/r8_single_gpu.sbatch)
echo "R8 D1=${J} (27 tasks, seed 42, cap 30 GPU-h)"
echo "After: .venv/bin/python scripts/r8_d1_report.py"
