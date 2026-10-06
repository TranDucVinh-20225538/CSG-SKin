#!/bin/bash
#SBATCH --job-name=cleanup-erm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/cleanup_erm_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/cleanup_erm_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
echo "=== cleanup ERM pad_heldout rescore $(date) host=$(hostname) ==="
${PY} -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
${PY} "${ROOT}/scripts/cleanup_erm_pad_heldout.py"
echo "=== done $(date) ==="
