#!/bin/bash
#SBATCH --job-name=paperB-r11b-y3
#SBATCH --array=0-29%4
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b/results/paperB/r11b/logs/y3_train_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b/results/paperB/r11b/logs/y3_train_%A_%a.err
set -euo pipefail
WT=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b
MAIN=/data2/cmdir/home/toandq/CSG-Skin-paperB
cd "$WT"
mkdir -p results/paperB/r11b/logs results/paperB/r11b/y3
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${WT}/scripts:${MAIN}/scripts
"$MAIN/.venv/bin/python" scripts/r11b_y3_x5_train.py --task_id "${SLURM_ARRAY_TASK_ID}" --skip_done
echo DONE_Y3_${SLURM_ARRAY_TASK_ID} $(date)
