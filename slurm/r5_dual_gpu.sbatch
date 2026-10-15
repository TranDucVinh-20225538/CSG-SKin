#!/bin/bash
#SBATCH --job-name=paperB-r5-dual
#SBATCH --array=0-9%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5dual_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5dual_%A_%a.err
# Dual-encoder ISIC runs (scripts/train_r5_dual_isic.py). Env: STAGE, BACKBONE, RECIPE, and for P11 PENALTY / WEIGHTS.
#   q3         10 tasks: --item q3, BACKBONE (convnext_tiny | dinov2_vits14 | dinov2_vitb14), λ ∈ {0, 2} (T/5) × seeds 42..82
#   p11_control 1 task: --item p11, no penalty, λ_adv = 0, seed 42 (selection reference in the P11 pipeline)
#   p11_scan   tasks over PENALTY_LIST × WEIGHT_LIST (env, space-separated), λ_adv = 0, seed 42, EPOCHS (env, 0 = recipe)
#   p11_seeds  5 tasks: PENALTY, WEIGHT, EPOCHS from env (the selection), λ_adv = 0, seeds 42..82
# Submit through scripts/launch_r5_q3.sh or scripts/launch_r5_p11.sh.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
RECIPE="${RECIPE:-current}"
cd "${ROOT}"
T=${SLURM_ARRAY_TASK_ID}
SEEDS=(42 52 62 72 82)
case "${STAGE}" in
  q3)        L=(0 2); ARGS=(--item q3 --backbone "${BACKBONE}" --lambda_adv "${L[$((T / 5))]}" --seed "${SEEDS[$((T % 5))]}") ;;
  p11_control) ARGS=(--item p11 --backbone effb3 --lambda_adv 0 --seed 42) ;;
  p11_scan)  read -r -a PL <<< "${PENALTY_LIST}"; read -r -a WL <<< "${WEIGHT_LIST}"
             NW=${#WL[@]}
             ARGS=(--item p11 --backbone effb3 --lambda_adv 0 --seed 42 --penalty "${PL[$((T / NW))]}"
                   --penalty_weight "${WL[$((T % NW))]}" --max_epochs "${EPOCHS:-0}") ;;
  p11_seeds) ARGS=(--item p11 --backbone effb3 --lambda_adv 0 --seed "${SEEDS[$T]}" --penalty "${PENALTY}"
                   --penalty_weight "${WEIGHT}" --max_epochs "${EPOCHS:-0}") ;;
  *) echo "unknown STAGE ${STAGE}"; exit 2 ;;
esac
echo "=== R5 dual stage=${STAGE} task=${T} args=${ARGS[*]} recipe=${RECIPE} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"${ROOT}/.venv/bin/python" scripts/train_r5_dual_isic.py "${ARGS[@]}" --recipe "${RECIPE}" --skip_done --num_workers 6
echo "=== done $(date) ==="
