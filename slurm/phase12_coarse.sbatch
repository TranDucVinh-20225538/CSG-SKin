#!/bin/bash
#SBATCH --job-name=paperB-p12c
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p12coarse_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p12coarse_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
LAMS=(0 0.1 1 10)
TID=${SLURM_ARRAY_TASK_ID}
LAM="${LAMS[$TID]}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== P12 coarse lambda_adv=${LAM} seed=42 start $(date) host=$(hostname) ==="
${PY} -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
${PY} ${ROOT}/scripts/train_phase12_camelyon_dann.py --lambda_adv "${LAM}" --seed 42 --skip_done --num_workers 4 --batch_size 32 --epochs 10
echo "=== P12 coarse lambda_adv=${LAM} seed=42 done $(date) ==="
