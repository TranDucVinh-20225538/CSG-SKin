#!/bin/bash
#SBATCH --job-name=paperB-r1-i3p
#SBATCH --array=0-11%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/reviewer_r1_item3_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/reviewer_r1_item3_%A_%a.err
#
# Item 3 parallel: up to 4 GPUs (user QOS cap). 12 runs = 3 waves × ~1.5 h ≈ 4–5 h wall.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
MULTS=(10 10 10 10 10 10 100 100 100 100 100 100)
LAMBDAS=(0.25 0.25 0.25 2 2 2 0.25 0.25 0.25 2 2 2)
SEEDS=(42 52 62 42 52 62 42 52 62 42 52 62)
TID=${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== Item3 parallel task=${TID} mult=${MULTS[$TID]} lam=${LAMBDAS[$TID]} seed=${SEEDS[$TID]} start $(date) ==="
${PY} "${ROOT}/scripts/train_phase3_sweep.py" \
  --lambda_adv "${LAMBDAS[$TID]}" --seed "${SEEDS[$TID]}" \
  --adv_lr_multiplier "${MULTS[$TID]}" \
  --resume --skip_done --num_workers 4 --max_epochs 40
echo "=== Item3 task=${TID} done $(date) ==="
