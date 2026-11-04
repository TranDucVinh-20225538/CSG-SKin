#!/bin/bash
#SBATCH --job-name=paperB-r10b-b1
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r10b_b1_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r10b_b1_%j.err
set -euo pipefail
cd /data2/cmdir/home/toandq/CSG-Skin-paperB
mkdir -p logs results/paperB/r10b
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python scripts/r10b_b1_shift50_boots.py
echo DONE_B1 $(date)
