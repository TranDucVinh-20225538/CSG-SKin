#!/bin/bash
#SBATCH --job-name=paperB-r1-i3q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/reviewer_r1_item3_serial_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/reviewer_r1_item3_serial_%j.err
#
# Item 3 — adv LR sensitivity: multipliers {10,100} × λ {0.25,2} × seeds {42,52,62}, one GPU serial queue.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"

MULTS=(10 100)
LAMBDAS=(0.25 2)
SEEDS=(42 52 62)
N=$(( ${#MULTS[@]} * ${#LAMBDAS[@]} * ${#SEEDS[@]} ))
step=0

echo "=== Item3 serial queue start $(date) host=$(hostname) n=${N} ==="
${PY} -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

for MULT in "${MULTS[@]}"; do
  for LAM in "${LAMBDAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      step=$((step + 1))
      echo "--- Item3 step ${step}/${N} lambda_adv=${LAM} adv_lr_mult=${MULT} seed=${SEED} $(date) ---"
      ${PY} "${ROOT}/scripts/train_phase3_sweep.py" \
        --lambda_adv "${LAM}" --seed "${SEED}" --adv_lr_multiplier "${MULT}" \
        --resume --skip_done --num_workers 4 --max_epochs 40
    done
  done
done

echo "=== Item3 serial queue done $(date) ==="
