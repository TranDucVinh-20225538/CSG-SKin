#!/bin/bash
#SBATCH --job-name=r14-redo-l0
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r14_pair3_redo_l0_s42_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r14_pair3_redo_l0_s42_%j.err
set -euo pipefail
ROOT=/data2/cmdir/home/toandq/CSG-Skin-paperB
cd "$ROOT"
mkdir -p logs results/paperB/r14_pair3 checkpoints/r14_pair3
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
.venv/bin/python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
# fresh train: no --skip_done; light extract OK (does not affect train)
.venv/bin/python scripts/r14_pair3_train.py --tag r14_ladv0_prior100_s42 --light
echo DONE_REDO_L0_S42 $(date)
