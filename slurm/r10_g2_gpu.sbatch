#!/bin/bash
#SBATCH --job-name=paperB-r10-g2
#SBATCH --array=0-2%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r10g2_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r10g2_%A_%a.err
# R10 G2: T2 pad_adv 50% λ=0 seeds 42,52,62 (same subsample seed 0).
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
RECIPE="${RECIPE:-current}"
cd "${ROOT}"
SEEDS=(42 52 62)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
echo "=== R10 G2 task=${SLURM_ARRAY_TASK_ID} seed=${SEED} start $(date) ==="
"${ROOT}/.venv/bin/python" scripts/train_r5_dual_isic.py \
  --item t2 --backbone effb3 --lambda_adv 0 --seed "${SEED}" \
  --pad_adv_frac 0.50 --pad_adv_subsample_seed 0 \
  --results_dir results/paperB/r7/t2 --tag_prefix r7_t2_ --ckpt_subdir r7_t2 \
  --recipe "${RECIPE}" --skip_done --num_workers 6
echo "=== done $(date) ==="
