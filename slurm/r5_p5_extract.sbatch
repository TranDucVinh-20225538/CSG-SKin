#!/bin/bash
#SBATCH --job-name=paperB-r5-p5-extract
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p5x_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p5x_%A_%a.err
# Paper B R5 P5 stage 1 (CPU): features of the Phase 15.1 single-encoder checkpoints.
# Default: the three λ = 4 runs (ONLY_LAMBDA=4, array 0-2). For Q4/R4 coverage of all 25 runs:
#   sbatch --array=0-24%5 --export=ALL,ONLY_LAMBDA= slurm/r5_p5_extract.sbatch

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "${ROOT}"
ONLY_LAMBDA="${ONLY_LAMBDA-4}"
EXTRA=""
if [ -n "${ONLY_LAMBDA}" ]; then EXTRA="--only_lambda ${ONLY_LAMBDA}"; fi
echo "=== R5 P5 extract task ${SLURM_ARRAY_TASK_ID} (only_lambda=${ONLY_LAMBDA:-all}) start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" scripts/r5_p5_single_encoder.py --extract "${SLURM_ARRAY_TASK_ID}" ${EXTRA}
echo "=== done $(date) ==="
