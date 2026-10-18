#!/bin/bash
#SBATCH --job-name=pB-r2p-n50ca
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6_r2p_n50ca_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6_r2p_n50ca_%j.err
# CPU: n50 class-balanced (b) + within-class (c) probes (cached features).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
cd "${ROOT}"
echo "=== n50 class-aware probes $(date) ==="
"${ROOT}/.venv/bin/python" scripts/r6_r2_prior_n50_class_aware.py
echo "=== done $(date) ==="
