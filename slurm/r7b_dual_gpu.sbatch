#!/bin/bash
#SBATCH --job-name=paperB-r7b-dual
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r7bdual_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r7bdual_%A_%a.err
# R7b dual. Env: STAGE, BACKBONE, RECIPE, BB_MULT, LAMBDA_ADV, SEED (optional overrides).
#   t1c_sel   2 tasks: dual λ=0 seed42 × {0.3,1.0}
#   t1c_main  array set by launcher: dual at chosen BB_MULT × λ∈{0,2} × seeds
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
    ARGS=(--item t1c --backbone "${BB}" --lambda_adv 0 --seed 42 --backbone_lr_mult "${M}"
          --results_dir results/paperB/r7b/t1c --tag_prefix r7b_t1c_ --ckpt_subdir r7b_t1c)
    STAGE_DIR=selection
    ;;
  t1c_main)
    # T = lam_idx*5 + seed_idx; LAMBDA_ADV and BB_MULT required from env
    M="${BB_MULT:?}"
    LAM="${LAMBDA_ADV:?}"
    SEED="${SEEDS[$((T % 5))]}"
    if [[ -n "${SEED_OVERRIDE:-}" ]]; then SEED="${SEED_OVERRIDE}"; fi
    ARGS=(--item t1c --backbone "${BB}" --lambda_adv "${LAM}" --seed "${SEED}" --backbone_lr_mult "${M}"
          --results_dir results/paperB/r7b/t1c --tag_prefix r7b_t1c_ --ckpt_subdir r7b_t1c)
    STAGE_DIR=runs
    ;;
  *) echo "unknown STAGE ${STAGE}"; exit 2 ;;
esac
# train_r5_dual always writes under runs/; for selection, redirect via RESULTS layout:
# dual script uses results_dir/runs/<tag>. Move selection into selection/ via RESULTS_DIR + post, or write to runs and symlink.
# Prefer writing all under runs/ and selection script reads by bb mult; also copy path note.
echo "=== R7b dual stage=${STAGE} task=${T} args=${ARGS[*]} recipe=${RECIPE} start $(date) ==="
"${ROOT}/.venv/bin/python" -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"${ROOT}/.venv/bin/python" scripts/train_r5_dual_isic.py "${ARGS[@]}" --recipe "${RECIPE}" --skip_done --num_workers 6
# if selection, mirror into selection/
if [[ "${STAGE}" == "t1c_sel" ]]; then
  TAG_DIR=$(ls -dt results/paperB/r7b/t1c/runs/r7b_t1c_dual_${BB}_ladv0_s42_bb* 2>/dev/null | head -1 || true)
  # find by bb mult tag
  BBTAG=$("${ROOT}/.venv/bin/python" -c "import sys; sys.path.insert(0,'scripts'); import r5_common as R; print(R.lam_tag(${M}))")
  SRC="results/paperB/r7b/t1c/runs/r7b_t1c_dual_${BB}_ladv0_s42_bb${BBTAG}"
  if [[ -d "${SRC}" ]]; then
    mkdir -p results/paperB/r7b/t1c/selection
    ln -sfn "../runs/$(basename "${SRC}")" "results/paperB/r7b/t1c/selection/$(basename "${SRC}")"
  fi
fi
echo "=== done $(date) ==="
