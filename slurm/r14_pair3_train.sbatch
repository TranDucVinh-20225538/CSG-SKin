#!/bin/bash
#SBATCH --job-name=r14-train
#SBATCH --array=0-19%2
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r14_pair3_train_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r14_pair3_train_%A_%a.err
set -euo pipefail
ROOT=/data2/cmdir/home/toandq/CSG-Skin-paperB
cd "$ROOT"
mkdir -p logs results/paperB/r14_pair3 checkpoints/r14_pair3
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
.venv/bin/python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
.venv/bin/python scripts/r14_pair3_train.py --task_id "${SLURM_ARRAY_TASK_ID}" --skip_done
echo DONE_R14_TRAIN_${SLURM_ARRAY_TASK_ID} $(date)
