#!/bin/bash
#SBATCH --job-name=paperB-r9-e2s
#SBATCH --array=0-5%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r9e2s_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r9e2s_%A_%a.err
# R9 E2 stage S: A4/A7 80ep ×2 + ERM 80ep ×2 (decisions/paperB_r9_e2.md).
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R9 E2 S task=${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"${ROOT}/.venv/bin/python" scripts/train_r9_e2.py --stage S --task_id "${SLURM_ARRAY_TASK_ID}" --resume --skip_done --num_workers 6
echo "=== done $(date) ==="
