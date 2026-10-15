#!/bin/bash
# Paper B R5 P7(b) launcher (decisions/paperB_r5_p7b.md). Prepared under the compute HOLD; not executed.
# Usage: GO=1 RECIPE=auto|current|p8 bash scripts/launch_r5_p7b.sh [AFTER_JOBID]
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs
if [[ "${GO:-0}" != "1" ]]; then echo "compute HOLD: set GO=1 to submit"; exit 1; fi
RECIPE="${RECIPE:-auto}"
DEP=()
if [[ -n "${1:-}" ]]; then DEP=(--dependency=afterany:$1); fi
J=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-14%2 --export=ALL,RECIPE=${RECIPE} slurm/r5_p7b.sbatch)
R=$(sbatch --parsable --dependency=afterany:${J} --export=ALL,ITEM=p7b,RECIPE=${RECIPE} slurm/r6_report.sbatch)
echo "P7b train=${J} report=${R} recipe=${RECIPE}"
