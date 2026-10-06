#!/bin/bash
#SBATCH --job-name=paperB-r1-i4q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/reviewer_r1_item4_serial_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/reviewer_r1_item4_serial_%j.err
#
# Item 4 — one GPU, four runs back-to-back (skip_done). No array / no manual slot-watching.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"

LAMBDAS=(0.25 0.25 1 1)
SEEDS=(72 82 72 82)

echo "=== Item4 serial queue start $(date) host=$(hostname) ==="
${PY} -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

for i in "${!LAMBDAS[@]}"; do
  LAM="${LAMBDAS[$i]}"
  SEED="${SEEDS[$i]}"
  echo "--- Item4 step $((i + 1))/4 lambda_adv=${LAM} seed=${SEED} $(date) ---"
  ${PY} "${ROOT}/scripts/train_phase3_sweep.py" \
    --lambda_adv "${LAM}" --seed "${SEED}" --resume --skip_done --num_workers 4 --max_epochs 40
done

echo "=== Item4 serial queue done $(date) ==="
