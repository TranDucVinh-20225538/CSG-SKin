#!/bin/bash
#SBATCH --job-name=paperB-r8-d1
#SBATCH --array=0-26%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r8d1_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r8d1_%A_%a.err
# R8 D1 screen (decisions/paperB_r8_d1.md). 27 tasks seed 42; cap 30 GPU-h.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
RECIPE="${RECIPE:-current}"
cd "${ROOT}"
T=${SLURM_ARRAY_TASK_ID}
read -r BB CFG LAM PEN W < <("${ROOT}/.venv/bin/python" scripts/r8_d1_plan.py "${T}")
BB_ARGS=()
if [[ "${BB}" == dinov2_* ]]; then BB_ARGS=(--backbone_lr_mult 0.1); fi
echo "=== R8 D1 task=${T} bb=${BB} cfg=${CFG} lam=${LAM} pen=${PEN} w=${W} start $(date) ==="
"${ROOT}/.venv/bin/python" -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"${ROOT}/.venv/bin/python" scripts/train_r5_single_isic.py \
  --item d1 --stage runs --backbone "${BB}" --config "${CFG}" --lambda_adv "${LAM}" --seed 42 \
  --penalty "${PEN}" --penalty_weight "${W}" \
  --results_dir results/paperB/r8/d1 --tag_prefix r8_d1_ --ckpt_subdir r8_d1 \
  --eval_mode d1 --cap_gpu_h 30 --skip_done --recipe "${RECIPE}" --num_workers 6 \
  "${BB_ARGS[@]+"${BB_ARGS[@]}"}"
echo "=== done $(date) ==="
