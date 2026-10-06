#!/bin/bash
#SBATCH --job-name=paperB-r1-i4
#SBATCH --array=0-3%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/reviewer_r1_item4_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/reviewer_r1_item4_%A_%a.err
#
# Reviewer Item 4 (priority before Item 3): raise n seeds to 5 at cliff λ.
# Runs: λ=0.25 × {72,82}, λ=1 × {72,82} — dual-encoder runB_orth1, same recipe as Phase 3.
# Prefer reviewer_r1_item4_serial.sbatch + reviewer_r1_enqueue_gpu.sh (no watching).
# Array limited to %1 concurrent task if you use this file.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
LAMBDAS=(0.25 0.25 1 1)
SEEDS=(72 82 72 82)
TID=${SLURM_ARRAY_TASK_ID}
LAM="${LAMBDAS[$TID]}"
SEED="${SEEDS[$TID]}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== Reviewer Item4 lambda_adv=${LAM} seed=${SEED} start $(date) host=$(hostname) ==="
${PY} -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
${PY} ${ROOT}/scripts/train_phase3_sweep.py --lambda_adv "${LAM}" --seed "${SEED}" --resume --skip_done --num_workers 4 --max_epochs 40
echo "=== Reviewer Item4 lambda_adv=${LAM} seed=${SEED} done $(date) ==="
