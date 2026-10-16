#!/bin/bash
# Paper B R6 R2 launcher (decisions/paperB_r6_r2.md). Prepared under the compute HOLD; not executed.
# Usage: GO=1 RECIPE=auto|current|p8 bash scripts/launch_r6_r2.sh [AFTER_JOBID]
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs
if [[ "${GO:-0}" != "1" ]]; then echo "compute HOLD: set GO=1 to submit"; exit 1; fi
RECIPE="${RECIPE:-auto}"
DEP=()
if [[ -n "${1:-}" ]]; then DEP=(--dependency=afterany:$1); fi
J=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-39%2 --export=ALL,RECIPE=${RECIPE} slurm/r6_r2.sbatch)
R=$(sbatch --parsable --dependency=afterany:${J} --export=ALL,ITEM=r2,RECIPE=${RECIPE} slurm/r6_report.sbatch)
echo "R2 train=${J} report=${R} recipe=${RECIPE}"
