#!/bin/bash
#SBATCH --job-name=paperB-p16b0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase16_b0_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase16_b0_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"
cd "${ROOT}"
echo "=== Phase 16 B0 start $(date) host=$(hostname) ==="
"${PY}" "${ROOT}/scripts/phase16_b0.py" \
  --seed 42 \
  --image_size 448 \
  --batch_size 32 \
  --num_workers 4 \
  --max_epochs 40 \
  --handcrafted_workers 8
echo "=== Phase 16 B0 done $(date) ==="
