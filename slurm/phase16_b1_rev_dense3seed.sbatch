#!/bin/bash
#SBATCH --job-name=paperB-p16b1rev-dense
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase16_b1_rev_dense_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase16_b1_rev_dense_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"
cd "${ROOT}"
LAMBDAS=(0 0.25 1 0 0.25 1)
SEEDS=(52 52 52 62 62 62)
LADV="${LAMBDAS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
if [ "$SEED" = "42" ]; then echo "skip seed 42"; exit 0; fi
"${PY}" "${ROOT}/scripts/phase16_b1_rev_coarse.py" \
  --lambda_adv "${LADV}" \
  --seed "${SEED}" \
  --image_size 448 \
  --batch_size 32 \
  --num_workers 4 \
  --max_epochs 40 \
  --skip_done
