#!/bin/bash
#SBATCH --job-name=paperB-p15bi
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase15b_iwildcam_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase15b_iwildcam_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
LAMBDAS=(0 0.1 1 10)
SEED=42
TID=${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== 15b iWildCam lambda=${LAMBDAS[$TID]} seed=${SEED} start $(date) host=$(hostname) ==="
${PY} ${ROOT}/scripts/train_phase15b_iwildcam.py \
  --lambda_adv "${LAMBDAS[$TID]}" --seed "${SEED}" \
  --output_root "${ROOT}/results/paperB/phase15b/iwildcam" \
  --ckpt_root "${ROOT}/checkpoints/phase15b/iwildcam" \
  --resume --skip_done --num_workers 2 --max_epochs 12 --batch_size 16 --image_size 448
echo "=== 15b iWildCam done $(date) ==="
