#!/bin/bash
#SBATCH --job-name=paperB-p16b1rev-l0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase16_b1_rev_l0_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase16_b1_rev_l0_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"
cd "${ROOT}"
echo "=== B1-rev λ=0 GATE (Maha before rest) $(date) ==="
"${PY}" "${ROOT}/scripts/phase16_b1_rev_coarse.py" \
  --lambda_adv 0 \
  --seed 42 \
  --image_size 448 \
  --batch_size 32 \
  --num_workers 4 \
  --max_epochs 40 \
  --skip_done
echo "=== done $(date) ==="
