#!/bin/bash
#SBATCH --job-name=paperB-r5-single
#SBATCH --array=0-4%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5single_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5single_%A_%a.err
# Single-encoder ISIC runs (scripts/train_r5_single_isic.py). Env: STAGE, BACKBONE (resnet50 | effb3), RECIPE.
#   p9_ladder  5 tasks, --array=0-4%1 (sequential: each rung checks the stop rule written by the previous one)
#              rung T ∈ baseline, T1_ganin_warmup, T2_bank_reinit, T3_strong, T4_long; λ=2, seed 42
#   p9_stage2  15 tasks: config 'selected', λ ∈ {0, 0.5, 2} (T/5) × seeds 42..82 (T%5)
#   q3_r50     10 tasks: resnet50, config 'selected', λ ∈ {0, 2} × seeds 42..82; written into p9/stage2 so the
#              cells are shared with P9 stage 2 (--skip_done prevents a second run of the same tag)
#   p9v2_r1|p9v2_r2|p9v2_s3|p9v2_s5  P9 v2 (amendment 2026-10-15): effb3; config / λ / seed from
#              `r5_p9v2.py plan <phase> T` (10 / 6 / 4 / 2 tasks); 30 GPU-h stage cap; a missing plan file = skip
# Submit through scripts/launch_r5_p9.sh or scripts/launch_r5_q3.sh.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
RECIPE="${RECIPE:-current}"
cd "${ROOT}"
T=${SLURM_ARRAY_TASK_ID}
SEEDS=(42 52 62 72 82)
case "${STAGE}" in
  p9_ladder) RUNGS=(baseline T1_ganin_warmup T2_bank_reinit T3_strong T4_long)
             ARGS=(--item p9 --stage ladder --ladder --config "${RUNGS[$T]}" --lambda_adv 2 --seed 42) ;;
  p9_stage2) L=(0 0.5 2); ARGS=(--item p9 --stage stage2 --config selected --lambda_adv "${L[$((T / 5))]}" --seed "${SEEDS[$((T % 5))]}") ;;
  q3_r50)    L=(0 2); BACKBONE=resnet50
             ARGS=(--item p9 --stage stage2 --config selected --lambda_adv "${L[$((T / 5))]}" --seed "${SEEDS[$((T % 5))]}") ;;
  p9v2_*)   PLAN=$("${ROOT}/.venv/bin/python" scripts/r5_p9v2.py plan "${STAGE#p9v2_}" "${T}") || { echo "${PLAN}"; exit 0; }
             read -r CFG LAM SEED <<< "${PLAN}"; BACKBONE=effb3
             ARGS=(--item p9 --stage v2 --cap_gpu_h 30 --config "${CFG}" --lambda_adv "${LAM}" --seed "${SEED}") ;;
  *) echo "unknown STAGE ${STAGE}"; exit 2 ;;
esac
echo "=== R5 single stage=${STAGE} backbone=${BACKBONE} task=${T} args=${ARGS[*]} recipe=${RECIPE} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"${ROOT}/.venv/bin/python" scripts/train_r5_single_isic.py --backbone "${BACKBONE}" "${ARGS[@]}" --recipe "${RECIPE}" \
  --skip_done --num_workers 6
echo "=== done $(date) ==="
