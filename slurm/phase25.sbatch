#!/bin/bash
#SBATCH --job-name=paperB-p25
#SBATCH --array=0-14
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase25_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase25_%A_%a.err
# No --gres: do not compete with Phase 4 for GPUs.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
METHODS=(runB_orth1 baseline_soft effb3_control)
SEEDS=(42 52 62 72 82)
TID=${SLURM_ARRAY_TASK_ID}
METHOD="${METHODS[$((TID / 5))]}"
SEED="${SEEDS[$((TID % 5))]}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== Phase 2.5 ${METHOD} seed=${SEED} start $(date) host=$(hostname) ==="
${PY} ${ROOT}/scripts/eval_phase25.py --cpu --num_workers 4 --batch_size 32 --method "${METHOD}" --seed "${SEED}" --skip_done
echo "=== Phase 2.5 ${METHOD} seed=${SEED} done $(date) ==="
