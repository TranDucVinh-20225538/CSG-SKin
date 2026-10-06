#!/bin/bash
#SBATCH --job-name=paperB-r1-i5
#SBATCH --array=0-8%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/reviewer_r1_item5_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/reviewer_r1_item5_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
LAMBDAS=(0 0 0 0.5 0.5 0.5 2 2 2)
SEEDS=(42 52 62 42 52 62 42 52 62)
TID=${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
mkdir -p logs
echo "=== Item5 ResNet-50 task=${TID} lam=${LAMBDAS[$TID]} seed=${SEEDS[$TID]} start $(date) ==="
${PY} "${ROOT}/scripts/train_reviewer_r1_item5_resnet50.py" \
  --lambda_adv "${LAMBDAS[$TID]}" --seed "${SEEDS[$TID]}" \
  --resume --skip_done --num_workers 4 --max_epochs 40
echo "=== Item5 task=${TID} done $(date) ==="
