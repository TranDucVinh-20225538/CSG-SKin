#!/bin/bash
#SBATCH --job-name=pB-p10-after
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_after_pilot_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_after_pilot_%j.err
# After pilot array: if SELECTED → Camelyon 10 then R2 release; else release R2 now.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
echo "=== after-pilot dispatch $(date) ==="
"${ROOT}/.venv/bin/python" scripts/r5_p10_after_pilot_dispatch.py
echo "=== done $(date) ==="
