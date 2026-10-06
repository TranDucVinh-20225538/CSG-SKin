#!/bin/bash
#SBATCH --job-name=paperB-p2
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase2_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase2_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
SEEDS=(42 52 62 72 82)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== Phase 2 seed=${SEED} start $(date) host=$(hostname) CUDA=${CUDA_VISIBLE_DEVICES:-} ==="
${PY} -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
${PY} ${ROOT}/scripts/train_phase2_pad_holdout.py --seed "${SEED}" --resume --skip_done --num_workers 4 --max_epochs 40
echo "=== Phase 2 seed=${SEED} done $(date) ==="
