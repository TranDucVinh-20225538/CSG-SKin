#!/bin/bash
# Launch R10 G1/G2/G3. Usage: GO=1 bash scripts/launch_r10_gpu.sh
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs results/paperB/r10
if [[ "${GO:-0}" != "1" ]]; then echo "set GO=1 to submit"; exit 1; fi
G1=$(sbatch --parsable slurm/r10_g1_gpu.sbatch)
G2=$(sbatch --parsable slurm/r10_g2_gpu.sbatch)
G3=$(sbatch --parsable slurm/r10_g3_gpu.sbatch)
echo "R10 G1=${G1} (4) G2=${G2} (3) G3=${G3} (5)  ≈18 GPU-h"
