#!/bin/bash
#SBATCH --job-name=paperB-r2-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r2_cpu_%x_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r2_cpu_%x_%j.err

# Usage: sbatch -J <name> r2_cpu.sbatch <script.py> [args]
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd "${ROOT}"
echo "=== R2 CPU $* start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" "$@"
echo "=== done $(date) ==="
