#!/bin/bash
# Paper B R7 T2 launcher (decisions/paperB_r7_t2.md). Run after T1b; cut first if time runs out.
# Usage: GO=1 [AFTER_JOBID=...] bash scripts/launch_r7_t2.sh
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs results/paperB/r7/t2
if [[ "${GO:-0}" != "1" ]]; then echo "set GO=1 to submit"; exit 1; fi
RECIPE="${RECIPE:-current}"
DEP=()
if [[ -n "${AFTER_JOBID:-}" ]]; then DEP=(--dependency=afterany:${AFTER_JOBID}); fi
J=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-8%2 --export=ALL,STAGE=t2_pad,RECIPE=${RECIPE} slurm/r7_dual_gpu.sbatch)
echo "T2=${J}"
