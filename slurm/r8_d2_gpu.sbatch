#!/bin/bash
#SBATCH --job-name=paperB-r8-d2
#SBATCH --array=0-4%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r8d2_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r8d2_%A_%a.err
# R8 D2: selected DINOv2 MK-MMD w=30 × seeds 42..82. Seed 42 reuses D1 ckpt + full eval.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
RECIPE="${RECIPE:-current}"
cd "${ROOT}"
SEEDS=(42 52 62 72 82)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
SEL=results/paperB/r8/d1/SELECTED_dinov2_vits14.json
CFG=$(.venv/bin/python -c "import json; print(json.load(open('${SEL}'))['config'])")
PEN=$(.venv/bin/python -c "import json; print(json.load(open('${SEL}'))['penalty'])")
W=$(.venv/bin/python -c "import json; print(json.load(open('${SEL}'))['penalty_weight'])")
LAM=$(.venv/bin/python -c "import json; print(json.load(open('${SEL}'))['lambda_adv'])")
echo "=== R8 D2 seed=${SEED} cfg=${CFG} pen=${PEN} w=${W} lam=${LAM} $(date) ==="
if [[ "${SEED}" == "42" ]]; then
  # Reuse D1 checkpoint; full Table-1 eval into d2/
  .venv/bin/python scripts/r8_d2_rescore_seed42.py
else
  .venv/bin/python scripts/train_r5_single_isic.py \
    --item d2 --stage runs --backbone dinov2_vits14 --config "${CFG}" \
    --lambda_adv "${LAM}" --seed "${SEED}" --penalty "${PEN}" --penalty_weight "${W}" \
    --backbone_lr_mult 0.1 --eval_mode full \
    --results_dir results/paperB/r8/d2 --tag_prefix r8_d2_ --ckpt_subdir r8_d2 \
    --skip_done --recipe "${RECIPE}" --num_workers 6
fi
echo "=== done $(date) ==="
