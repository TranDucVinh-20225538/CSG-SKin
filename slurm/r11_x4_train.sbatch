#!/bin/bash
#SBATCH --job-name=paperB-r11-x4
#SBATCH --array=0-14%2
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r11_x4_train_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r11_x4_train_%A_%a.err
set -euo pipefail
ROOT=/data2/cmdir/home/toandq/CSG-Skin-paperB
cd "$ROOT"
mkdir -p logs results/paperB/r11/x4
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
.venv/bin/python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
.venv/bin/python scripts/r11_x4_train.py --task_id "${SLURM_ARRAY_TASK_ID}" --skip_done --num_workers 4
echo DONE_X4_TRAIN_${SLURM_ARRAY_TASK_ID} $(date)
