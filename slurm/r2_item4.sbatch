#!/bin/bash
#SBATCH --job-name=paperB-r2-i4
#SBATCH --array=0-5%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r2_item4_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r2_item4_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
LAMBDAS=(0 0 0 2 2 2)
SEEDS=(42 52 62 42 52 62)
TID=${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R2 Item4 lam=${LAMBDAS[$TID]} seed=${SEEDS[$TID]} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" scripts/train_r2_item4_lesion_split.py \
  --lambda_adv "${LAMBDAS[$TID]}" --seed "${SEEDS[$TID]}" \
  --resume --skip_done --num_workers 4 --max_epochs 40
echo "=== done $(date) ==="
