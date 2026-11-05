#!/bin/bash
#SBATCH --job-name=paperB-r11-x1
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r11_x1_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r11_x1_%j.err
set -euo pipefail
cd /data2/cmdir/home/toandq/CSG-Skin-paperB
mkdir -p logs results/paperB/r11/x1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv/bin/python scripts/r11_x1_label_space.py
echo DONE_R11_X1 $(date)
