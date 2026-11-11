#!/bin/bash
#SBATCH --job-name=paperB-r12sc
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r12_reweight_score_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r12_reweight_score_%j.err
set -euo pipefail
ROOT=/data2/cmdir/home/toandq/CSG-Skin-paperB
cd "$ROOT"
mkdir -p logs results/paperB/r12_reweight
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
.venv/bin/python scripts/r12_reweight_score.py --write_report
echo DONE_R12_SCORE $(date)
