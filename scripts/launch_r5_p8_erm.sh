#!/bin/bash
# Lesion-level single-encoder ERM to 5 seeds + competence-note refresh of P8 REPORT.
# Usage: GO=1 bash scripts/launch_r5_p8_erm.sh [AFTER_JOBID]
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs
if [[ "${GO:-0}" != "1" ]]; then echo "set GO=1 to submit"; exit 1; fi
DEP=()
if [[ -n "${1:-}" ]]; then DEP=(--dependency=afterany:$1); fi
E=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} slurm/r5_p8_erm.sbatch)
R=$(sbatch --parsable --dependency=afterok:${E} slurm/r5_p8_erm_report.sbatch)
echo "P8 ERM train_eval=${E} report=${R}"
echo "last job: ${R}"
