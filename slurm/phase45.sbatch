#!/bin/bash
#SBATCH --job-name=paperB-p45
#SBATCH --array=0-17
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase45_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase45_%A_%a.err
# No --gres: do not compete with Phase 3 for GPUs.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
CONFIGS=(4a 4b)
METHODS=(baseline effb3 runB_orth1)
SEEDS=(42 52 62)
TID=${SLURM_ARRAY_TASK_ID}
CFG="${CONFIGS[$((TID / 9))]}"
METHOD="${METHODS[$(( (TID % 9) / 3 ))]}"
SEED="${SEEDS[$((TID % 3))]}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== Phase 4.5a/b ${CFG} ${METHOD} seed=${SEED} start $(date) host=$(hostname) ==="
${PY} ${ROOT}/scripts/eval_phase45.py --cpu --num_workers 4 --batch_size 32 --config "${CFG}" --method "${METHOD}" --seed "${SEED}" --skip_done
echo "=== Phase 4.5a/b ${CFG} ${METHOD} seed=${SEED} done $(date) ==="
