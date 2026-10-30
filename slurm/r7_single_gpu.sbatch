#!/bin/bash
#SBATCH --job-name=paperB-r7-single
#SBATCH --array=0-4%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r7single_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r7single_%A_%a.err
# R7 single-encoder. Env: STAGE, BACKBONE, RECIPE.
#   t1_erm   5 tasks: DINOv2 ERM λ=0 seeds 42..82 → results/paperB/r7/t1/erm
#   t1b      5 tasks: DINOv2 DANN λ=2 seeds 42..82 → results/paperB/r7/t1b/runs
#   t1_erm_extra  override ARRAY for seeds 72,82 only (amendment 2026-10-30)
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
RECIPE="${RECIPE:-current}"
BB="${BACKBONE:-dinov2_vits14}"
cd "${ROOT}"
T=${SLURM_ARRAY_TASK_ID}
SEEDS=(42 52 62 72 82)
case "${STAGE}" in
  t1_erm)
    ARGS=(--item t1 --stage erm --config baseline --backbone "${BB}" --lambda_adv 0 --seed "${SEEDS[$T]}"
          --results_dir results/paperB/r7/t1 --tag_prefix r7_t1_ --ckpt_subdir r7_t1)
    ;;
  t1_erm_extra)
    # tasks 0,1 → seeds 72,82 (amendment; original array was 0-2)
    EXTRA=(72 82)
    ARGS=(--item t1 --stage erm --config baseline --backbone "${BB}" --lambda_adv 0 --seed "${EXTRA[$T]}"
          --results_dir results/paperB/r7/t1 --tag_prefix r7_t1_ --ckpt_subdir r7_t1)
    ;;
  t1b)
    ARGS=(--item t1b --stage runs --config baseline --backbone "${BB}" --lambda_adv 2 --seed "${SEEDS[$T]}"
          --results_dir results/paperB/r7/t1b --tag_prefix r7_t1b_ --ckpt_subdir r7_t1b)
    ;;
  *) echo "unknown STAGE ${STAGE}"; exit 2 ;;
esac
echo "=== R7 single stage=${STAGE} backbone=${BB} task=${T} args=${ARGS[*]} recipe=${RECIPE} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"${ROOT}/.venv/bin/python" scripts/train_r5_single_isic.py "${ARGS[@]}" --recipe "${RECIPE}" --skip_done --num_workers 6
echo "=== done $(date) ==="
