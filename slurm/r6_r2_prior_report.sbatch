#!/bin/bash
#SBATCH --job-name=pB-r2p-report
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r2_prior_report_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r2_prior_report_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== R2-prior report $(date) ==="
# Refuse until complete (exit 2); Slurm will show FAILED if still incomplete — re-run after array.
"${ROOT}/.venv/bin/python" scripts/r6_r2_prior_report.py
echo "=== done $(date) ==="
