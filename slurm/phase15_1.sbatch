#!/bin/bash
#SBATCH --job-name=paperB-p151
#SBATCH --array=0-24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase15_1_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase15_1_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
LAMBDAS=(0 0 0 0 0 0.25 0.25 0.25 0.5 0.5 0.5 1 1 1 2 2 2 2 2 4 4 4 8 8 8)
SEEDS=(42 52 62 72 82 42 52 62 42 52 62 42 52 62 42 52 62 72 82 42 52 62 42 52 62)
TID=${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== 15.1 lambda=${LAMBDAS[$TID]} seed=${SEEDS[$TID]} start $(date) host=$(hostname) ==="
${PY} -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
${PY} ${ROOT}/scripts/train_phase15_1_single_dann.py --lambda_adv "${LAMBDAS[$TID]}" --seed "${SEEDS[$TID]}" --resume --skip_done --num_workers 4 --max_epochs 40
echo "=== 15.1 done $(date) ==="
