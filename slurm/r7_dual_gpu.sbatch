#!/bin/bash
#SBATCH --job-name=paperB-r7-dual
#SBATCH --array=0-9%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r7dual_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r7dual_%A_%a.err
# R7 dual-encoder. Env: STAGE, BACKBONE, RECIPE, PAD_ADV_FRAC (optional).
#   t1_dual   10 tasks: DINOv2 dual λ∈{0,2}×seeds 42..82 → results/paperB/r7/t1
#   t2_pad    9 tasks: EffB3; T=0..2 λ=0 pad25%; T=3..5 λ=2 pad25%; T=6..8 λ=2 pad50%
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
RECIPE="${RECIPE:-current}"
cd "${ROOT}"
T=${SLURM_ARRAY_TASK_ID}
SEEDS=(42 52 62 72 82)
case "${STAGE}" in
  t1_dual)
    L=(0 2); BB="${BACKBONE:-dinov2_vits14}"
    ARGS=(--item t1 --backbone "${BB}" --lambda_adv "${L[$((T / 5))]}" --seed "${SEEDS[$((T % 5))]}"
          --results_dir results/paperB/r7/t1 --tag_prefix r7_t1_ --ckpt_subdir r7_t1)
    ;;
  t2_pad)
    BB=effb3
    if [[ "${T}" -lt 3 ]]; then LAM=0; FRAC=0.25; SEED="${SEEDS[$T]}"
    elif [[ "${T}" -lt 6 ]]; then LAM=2; FRAC=0.25; SEED="${SEEDS[$((T - 3))]}"
    else LAM=2; FRAC=0.50; SEED="${SEEDS[$((T - 6))]}"
    fi
    ARGS=(--item t2 --backbone "${BB}" --lambda_adv "${LAM}" --seed "${SEED}"
          --pad_adv_frac "${FRAC}" --pad_adv_subsample_seed 0
          --results_dir results/paperB/r7/t2 --tag_prefix r7_t2_ --ckpt_subdir r7_t2)
    ;;
  *) echo "unknown STAGE ${STAGE}"; exit 2 ;;
esac
echo "=== R7 dual stage=${STAGE} task=${T} args=${ARGS[*]} recipe=${RECIPE} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"${ROOT}/.venv/bin/python" scripts/train_r5_dual_isic.py "${ARGS[@]}" --recipe "${RECIPE}" --skip_done --num_workers 6
echo "=== done $(date) ==="
