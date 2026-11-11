#!/bin/bash
#SBATCH --job-name=paperB-r11b-y4s
#SBATCH --array=0-3%2
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b/results/paperB/r11b/logs/y4_screen_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b/results/paperB/r11b/logs/y4_screen_%A_%a.err
set -euo pipefail
WT=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b
MAIN=/data2/cmdir/home/toandq/CSG-Skin-paperB
cd "$WT"
mkdir -p results/paperB/r11b/logs results/paperB/r11b/y4
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${WT}/scripts:${MAIN}/scripts
"$MAIN/.venv/bin/python" -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"$MAIN/.venv/bin/python" scripts/r11b_y4_x4_train.py --phase screen --task_id "${SLURM_ARRAY_TASK_ID}" --skip_done
echo DONE_Y4_SCREEN_${SLURM_ARRAY_TASK_ID} $(date)
