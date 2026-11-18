#!/bin/bash
#SBATCH --job-name=r14-score
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r14_pair3_score_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r14_pair3_score_%j.err
set -euo pipefail
ROOT=/data2/cmdir/home/toandq/CSG-Skin-paperB
cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
.venv/bin/python scripts/r14_pair3_score.py
echo DONE_R14_SCORE $(date)
