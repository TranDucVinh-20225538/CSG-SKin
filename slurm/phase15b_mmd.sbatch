#!/bin/bash
#SBATCH --job-name=paperB-p15bm
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase15b_mmd_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase15b_mmd_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
WEIGHTS=(300 300 300 1000 1000 1000 3000 3000 3000 10000 10000 10000)
SEEDS=(42 52 62 42 52 62 42 52 62 42 52 62)
TID=${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== 15b MMD w=${WEIGHTS[$TID]} seed=${SEEDS[$TID]} start $(date) host=$(hostname) ==="
${PY} ${ROOT}/scripts/train_phase15_2_objectives.py \
  --objective mmd --weight "${WEIGHTS[$TID]}" --seed "${SEEDS[$TID]}" \
  --output_root "${ROOT}/results/paperB/phase15b/mmd" \
  --ckpt_root "${ROOT}/checkpoints/phase15b/mmd" \
  --resume --skip_done --num_workers 4 --max_epochs 40
echo "=== 15b MMD done $(date) ==="
