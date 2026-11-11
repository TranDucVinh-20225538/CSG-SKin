#!/bin/bash
#SBATCH --job-name=paperB-r11b-y1
#SBATCH --array=0-3
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b/results/paperB/r11b/logs/y1_slurm_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b/results/paperB/r11b/logs/y1_slurm_%A_%a.err
set -euo pipefail
WT=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b
MAIN=/data2/cmdir/home/toandq/CSG-Skin-paperB
cd "$WT"
mkdir -p results/paperB/r11b/logs results/paperB/r11b/y1
export CUDA_VISIBLE_DEVICES=
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${WT}/scripts:${MAIN}/scripts
"$MAIN/.venv/bin/python" -u scripts/r11b_y1_diagnose.py --device cpu \
  --shard "${SLURM_ARRAY_TASK_ID}" --n_shards 4 --skip_done
echo DONE_Y1_SHARD_${SLURM_ARRAY_TASK_ID} $(date)
