#!/bin/bash
#SBATCH --job-name=paperB-r6-r1hi
#SBATCH --array=0-9%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6r1hi_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6r1hi_%A_%a.err
# R1-hi main wave: λ∈{4,8}×5 seeds, shift 0%, RECIPE=current, last.ckpt.
# WAVE=main (default) or WAVE=advlr3 (array 0-4).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
WAVE="${WAVE:-main}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R6 R1-hi wave=${WAVE} task=${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
"${ROOT}/.venv/bin/python" scripts/train_r6_r1_hi.py --recipe current --wave "${WAVE}" \
  --task_id "${SLURM_ARRAY_TASK_ID}" --resume --skip_done --num_workers 6
echo "=== done $(date) ==="
