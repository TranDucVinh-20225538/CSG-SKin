#!/bin/bash
#SBATCH --job-name=paperB-r4-x
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r4x_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r4x_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
# tasks 0-3: Phase 13 features, 4-7: Phase 6 cross-domain eval; EXTRA index = task % 4
WHICH=$([[ ${SLURM_ARRAY_TASK_ID} -lt 4 ]] && echo p13 || echo xfer)
IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))
echo "=== R4 extra ${WHICH} ${IDX} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" scripts/r4_extra_phase3_runs.py "${WHICH}" "${IDX}"
echo "=== done $(date) ==="
