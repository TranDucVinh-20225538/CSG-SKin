#!/bin/bash
#SBATCH --job-name=paperB-r5-q1
#SBATCH --array=0-2%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5q1_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5q1_%A_%a.err
# STAGE (env): gate (3 tasks) | sweep (12) | dose (9, conditional on INVERSION.json). RECIPE (env, default current).
# Submit through scripts/launch_r5_q1.sh, which sets --array per stage.
#   gate   λ=0 x seeds 42,52,62
#   sweep  λ=0 x 72,82; λ=0.5 x 42..82; λ=2 x 42..82
#   dose   λ=2, adversary-stream shift kept f ∈ {1, 0.5, 0} x seeds 42, 52, 62 (three-stream; 3 seeds, user decision 2026-10-15)

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
RECIPE="${RECIPE:-current}"
cd "${ROOT}"
T=${SLURM_ARRAY_TASK_ID}
SEEDS=(42 52 62 72 82)
SHIFT=""
case "${STAGE}" in
  gate)  LAM=0; SEED=${SEEDS[$T]} ;;
  sweep) if (( T < 2 )); then LAM=0; SEED=${SEEDS[$((T + 3))]}
         elif (( T < 7 )); then LAM=0.5; SEED=${SEEDS[$((T - 2))]}
         else LAM=2; SEED=${SEEDS[$((T - 7))]}; fi ;;
  dose)  SH=(1.0 0.5 0.0); LAM=2; SHIFT=${SH[$((T / 3))]}; SEED=${SEEDS[$((T % 3))]} ;;
  *) echo "unknown STAGE ${STAGE}"; exit 2 ;;
esac
echo "=== R5 Q1 stage=${STAGE} task=${T} lam=${LAM} seed=${SEED} shift=${SHIFT:-none} recipe=${RECIPE} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
D=$(mktemp -d /tmp/r5q1_XXXXXX)
trap 'rm -rf "${D}"' EXIT
"${ROOT}/.venv/bin/python" scripts/r5_q1_cxr_data.py --stage_to "${D}"
ARGS=(--stage "${STAGE}" --lambda_adv "${LAM}" --seed "${SEED}" --root "${D}" --recipe "${RECIPE}" --skip_done --num_workers 8)
if [[ -n "${SHIFT}" ]]; then ARGS+=(--adv_mix_shift "${SHIFT}"); fi
"${ROOT}/.venv/bin/python" scripts/train_r5_q1_cxr.py "${ARGS[@]}"
echo "=== done $(date) ==="
