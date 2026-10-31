#!/bin/bash
#SBATCH --job-name=paperB-r7b-single
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r7bsingle_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r7bsingle_%A_%a.err
# R7b single-encoder. Env: STAGE, BACKBONE, RECIPE, BB_MULT.
#   t1c_sel   2 tasks: ERM λ=0 seed42 × {0.3,1.0}
#   t1c_erm   5 tasks: ERM at BB_MULT × seeds 42..82
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
  t1c_sel)
    MULTS=(0.3 1.0)
    M="${MULTS[$T]}"
    ARGS=(--item t1c --stage erm --config baseline --backbone "${BB}" --lambda_adv 0 --seed 42
          --backbone_lr_mult "${M}"
          --results_dir results/paperB/r7b/t1c --tag_prefix r7b_t1c_ --ckpt_subdir r7b_t1c)
    ;;
  t1c_erm)
    M="${BB_MULT:?}"
    ARGS=(--item t1c --stage erm --config baseline --backbone "${BB}" --lambda_adv 0 --seed "${SEEDS[$T]}"
          --backbone_lr_mult "${M}"
          --results_dir results/paperB/r7b/t1c --tag_prefix r7b_t1c_ --ckpt_subdir r7b_t1c)
    ;;
  *) echo "unknown STAGE ${STAGE}"; exit 2 ;;
esac
echo "=== R7b single stage=${STAGE} task=${T} args=${ARGS[*]} recipe=${RECIPE} start $(date) ==="
"${ROOT}/.venv/bin/python" scripts/train_r5_single_isic.py "${ARGS[@]}" --recipe "${RECIPE}" --skip_done --num_workers 6
if [[ "${STAGE}" == "t1c_sel" ]]; then
  BBTAG=$("${ROOT}/.venv/bin/python" -c "import sys; sys.path.insert(0,'scripts'); import r5_common as R; print(R.lam_tag(${M}))")
  SRC="results/paperB/r7b/t1c/erm/r7b_t1c_single_${BB}_baseline_ladv0_s42_bb${BBTAG}"
  if [[ -d "${SRC}" ]]; then
    mkdir -p results/paperB/r7b/t1c/selection
    ln -sfn "../erm/$(basename "${SRC}")" "results/paperB/r7b/t1c/selection/$(basename "${SRC}")"
  fi
fi
echo "=== done $(date) ==="
