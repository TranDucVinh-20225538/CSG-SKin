#!/bin/bash
# Paper B R7 T1b launcher (decisions/paperB_r7_t1b.md). Run after T1.
# Usage: GO=1 [AFTER_JOBID=...] bash scripts/launch_r7_t1b.sh
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs results/paperB/r7/t1b
if [[ "${GO:-0}" != "1" ]]; then echo "set GO=1 to submit"; exit 1; fi
RECIPE="${RECIPE:-current}"
VIT="${VIT:-dinov2_vits14}"
DEP=()
if [[ -n "${AFTER_JOBID:-}" ]]; then DEP=(--dependency=afterany:${AFTER_JOBID}); fi
J=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-2%2 --export=ALL,STAGE=t1b,BACKBONE=${VIT},RECIPE=${RECIPE} slurm/r7_single_gpu.sbatch)
echo "T1b=${J} backbone=${VIT}"
