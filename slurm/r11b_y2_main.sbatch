#!/bin/bash
#SBATCH --job-name=paperB-r11b-y2m
#SBATCH --array=0-24%4
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b/results/paperB/r11b/logs/y2_main_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b/results/paperB/r11b/logs/y2_main_%A_%a.err
set -euo pipefail
WT=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b
MAIN=/data2/cmdir/home/toandq/CSG-Skin-paperB
cd "$WT"
mkdir -p results/paperB/r11b/logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${WT}/scripts:${MAIN}/scripts
"$MAIN/.venv/bin/python" scripts/r11b_y2_x3_train.py --phase main --task_id "${SLURM_ARRAY_TASK_ID}" --skip_done
echo DONE_Y2_MAIN_${SLURM_ARRAY_TASK_ID} $(date)
