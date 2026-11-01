#!/bin/bash
# Paper B R9 E1 launcher. Usage: GO=1 bash scripts/launch_r9_e1.sh
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs results/paperB/r9/e1
if [[ "${GO:-0}" != "1" ]]; then echo "set GO=1 to submit"; exit 1; fi
J=$(sbatch --parsable slurm/r9_e1_gpu.sbatch)
F=$(sbatch --parsable --dependency=afterok:${J} slurm/r9_e1_finalize.sbatch)
echo "R9 E1=${J} FIN=${F} (10 three-stream runs, ≈21 GPU-h)"
