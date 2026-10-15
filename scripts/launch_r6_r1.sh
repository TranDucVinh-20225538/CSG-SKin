#!/bin/bash
# Paper B R6 R1 launcher (decisions/paperB_r6_r1.md). Prepared under the compute HOLD; not executed.
# Usage: GO=1 RECIPE=auto|current|p8 bash scripts/launch_r6_r1.sh [AFTER_JOBID]
# RECIPE=auto resolves at run time from results/paperB/r5/p8/RECIPE_DECISION.json, so chain it after the P8 report.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs
if [[ "${GO:-0}" != "1" ]]; then echo "compute HOLD: set GO=1 to submit"; exit 1; fi
RECIPE="${RECIPE:-auto}"
DEP=()
if [[ -n "${1:-}" ]]; then DEP=(--dependency=afterany:$1); fi
J=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-30%2 --export=ALL,RECIPE=${RECIPE} slurm/r6_r1.sbatch)
R=$(sbatch --parsable --dependency=afterany:${J} --export=ALL,ITEM=r1,RECIPE=${RECIPE} slurm/r6_report.sbatch)
echo "R1 train=${J} report=${R} recipe=${RECIPE}"
