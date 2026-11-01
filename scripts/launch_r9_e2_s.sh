#!/bin/bash
# Paper B R9 E2 stage S launcher. Usage: GO=1 bash scripts/launch_r9_e2_s.sh
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs results/paperB/r9/e2
if [[ "${GO:-0}" != "1" ]]; then echo "set GO=1 to submit"; exit 1; fi
J=$(sbatch --parsable slurm/r9_e2_s_gpu.sbatch)
F=$(sbatch --parsable --dependency=afterok:${J} slurm/r9_e2_s_finalize.sbatch)
echo "R9 E2S=${J} FIN=${F} (4 dual + 2 ERM at 80ep, ≈13 GPU-h)"
