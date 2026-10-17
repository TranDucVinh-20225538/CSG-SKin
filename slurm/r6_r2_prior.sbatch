#!/bin/bash
#SBATCH --job-name=paperB-r2-prior
#SBATCH --array=0-9%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6r2p_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6r2p_%A_%a.err
# R2-prior EM + n50 (decisions/paperB_r6_r2_prior.md). One GPU; parallel with P10.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R2-prior task=${SLURM_ARRAY_TASK_ID} $(date) ==="
"${ROOT}/.venv/bin/python" scripts/train_r6_r2_prior.py --recipe "${RECIPE:-current}" \
  --task_id "${SLURM_ARRAY_TASK_ID}" --resume --skip_done --num_workers 6
echo "=== done $(date) ==="
