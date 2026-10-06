#!/bin/bash
# Enqueue Paper B reviewer GPU work as a dependency chain (1 GPU per job, no watching).
# Usage:
#   ./slurm/reviewer_r1_enqueue_gpu.sh              # Item4 serial → Item3 serial
#   ./slurm/reviewer_r1_enqueue_gpu.sh afterok:63137_0   # wait for a running array task first
#
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
DEP="${1:-}"

I4_ARGS=()
if [[ -n "${DEP}" ]]; then
  I4_ARGS=(--dependency="${DEP}")
fi

I4=$(sbatch --parsable "${I4_ARGS[@]}" slurm/reviewer_r1_item4_serial.sbatch)
I3=$(sbatch --parsable --dependency="afterok:${I4}" slurm/reviewer_r1_item3_serial.sbatch)

echo "Queued Item4 serial job ${I4}"
echo "Queued Item3 serial job ${I3} (afterok:${I4})"
echo "Monitor: squeue -u \"${USER}\" -n paperB-r1-i4q,paperB-r1-i3q"
