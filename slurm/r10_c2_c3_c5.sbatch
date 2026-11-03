#!/bin/bash
#SBATCH --job-name=paperB-r10-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r10cpu_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r10cpu_%j.err
set -euo pipefail
ROOT=/data2/cmdir/home/toandq/CSG-Skin-paperB
export PYTHONUNBUFFERED=1 PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export OMP_NUM_THREADS=32
cd "$ROOT"
"${ROOT}/.venv/bin/python" scripts/r10_c2_c3_c5.py --items c2,c3,c5
echo DONE $(date)
