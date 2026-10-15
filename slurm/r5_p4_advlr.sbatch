#!/bin/bash
#SBATCH --job-name=paperB-r5-p4-advlr
#SBATCH --array=0-5%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=node002
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p4advlr_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p4advlr_%A_%a.err
# GPU. Paper B R5 P4 (GPU part): lesion-level λ = 2 retrain with adversary LR ×10 / ×100, seeds 42/52/62
# (decisions/paperB_r5_p4.md). 6 runs × ≈1.6 GPU-h. Report: STEP=p4_advlr_report of slurm/paperB_r5r6_cpu.sbatch.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
MULTS=(10 10 10 100 100 100)
SEEDS=(42 52 62 42 52 62)
TID=${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R5 P4 advlr x${MULTS[$TID]} seed=${SEEDS[$TID]} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" scripts/r5_p4_advlr_train.py \
  --seed "${SEEDS[$TID]}" --adv_lr_multiplier "${MULTS[$TID]}" \
  --resume --skip_done --num_workers 4 --max_epochs 40
echo "=== done $(date) ==="
