#!/bin/bash
#SBATCH --job-name=pB-p10-nct
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_nct_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_nct_%j.err
# CPU re-score P10 shift-arms checkpoints on NCT-CRC-HE (bug fix; no GPU).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== P10 NCT rescore $(date) ==="
"${ROOT}/.venv/bin/python" scripts/r5_p10_nct_rescore.py --device cpu --num_workers 6
echo "=== done $(date) ==="
