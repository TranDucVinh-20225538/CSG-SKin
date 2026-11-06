#!/bin/bash
#SBATCH --job-name=paperB-r11-x5fwd
#SBATCH --array=0-2
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r11_x5_fwd_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r11_x5_fwd_%A_%a.err
set -euo pipefail
ROOT=/data2/cmdir/home/toandq/CSG-Skin-paperB
cd "$ROOT"
mkdir -p logs results/paperB/r11/x5
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
.venv/bin/python -c "import torch; print('CUDA', torch.cuda.is_available())"
.venv/bin/python scripts/r11_x5_synth_forward.py --task_id "${SLURM_ARRAY_TASK_ID}" --num_workers 4
echo DONE_X5_FWD_${SLURM_ARRAY_TASK_ID} $(date)
