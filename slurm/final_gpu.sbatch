#!/bin/bash
#SBATCH --job-name=paperB-final
#SBATCH --array=0-36
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/finalgpu_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/finalgpu_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== final gpu task ${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
${PY} -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
${PY} ${ROOT}/scripts/eval_final_gpu.py --task_id "${SLURM_ARRAY_TASK_ID}" --skip_done --num_workers 4 --batch_size 64
echo "=== final gpu task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
