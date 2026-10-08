#!/bin/bash
#SBATCH --job-name=paperB-r2-i5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r2_item5_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r2_item5_%A_%a.err

# LAMS / SEEDS (and optional PLACES): colon-separated lists indexed by SLURM_ARRAY_TASK_ID.
# ASSIGN: primary | fallback. MAXSTEPS: >0 for the λ-placement diagnostic (log only).
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
IFS=: read -r -a L <<< "${LAMS}"
IFS=: read -r -a S <<< "${SEEDS}"
TID=${SLURM_ARRAY_TASK_ID}
PLACE=grl
if [[ -n "${PLACES:-}" ]]; then IFS=: read -r -a P <<< "${PLACES}"; PLACE=${P[$TID]}; fi
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R2 Item5 assign=${ASSIGN:-primary} lam=${L[$TID]} seed=${S[$TID]} place=${PLACE} maxsteps=${MAXSTEPS:-0} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" scripts/train_r2_item5_camelyon2.py --lambda_adv "${L[$TID]}" --seed "${S[$TID]}" \
  --assign "${ASSIGN:-primary}" --adv_placement "${PLACE}" --max_steps "${MAXSTEPS:-0}" --skip_done --num_workers 6
echo "=== done $(date) ==="
