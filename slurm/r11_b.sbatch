#!/bin/bash
#SBATCH --job-name=paperB-r11-b
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r11_b_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r11_b_%j.err
set -euo pipefail
cd /data2/cmdir/home/toandq/CSG-Skin-paperB
mkdir -p logs results/paperB/r11/b
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python scripts/r11_b_redraw.py
echo DONE_R11_B $(date)
