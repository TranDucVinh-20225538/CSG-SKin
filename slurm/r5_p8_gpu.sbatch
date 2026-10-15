#!/bin/bash
#SBATCH --job-name=paperB-r5-p8
#SBATCH --array=0-17%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p8_%x_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p8_%x_%A_%a.err
# STAGE (env): A (18 tasks: 16 grid + 2 single-encoder ERM refs) | B (2) | main (15) | ext (20, gated).
# Submit through scripts/launch_r5_p8.sh, which sets --array per stage.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R5 P8 stage=${STAGE} task=${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"${ROOT}/.venv/bin/python" scripts/train_r5_p8.py --stage "${STAGE}" --task_id "${SLURM_ARRAY_TASK_ID}" \
  --resume --skip_done --num_workers 6
echo "=== done $(date) ==="
