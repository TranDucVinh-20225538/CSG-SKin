#!/bin/bash
# Paper B R5 P10 launcher (decisions/paperB_r5_p10.md).
# Usage: GO=1 [RECIPE=current|p8] bash scripts/launch_r5_p10.sh [AFTER_JOBID]
# GPU order: R1 → R2 → P10 → P9 → … . Chain: ladder (5 rungs %1) → stage2 → report.
# Cap 40 GPU-h; gate = held-out hospital probe drop ≥ 0.10 vs λ=0.
# Camelyon17 uses the WILDS copy in DST-Skin/data/raw/wilds (no staging step). RECIPE=current|p8 (env): only the
# classifier LR multiplier applies to Camelyon (see the decisions file).
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs
if [[ "${GO:-0}" != "1" ]]; then echo "compute HOLD: set GO=1 to submit"; exit 1; fi
DEP=()
if [[ -n "${1:-}" ]]; then DEP=(--dependency=afterany:$1); fi
L=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-4%1 --export=ALL,STAGE=ladder,RECIPE=${RECIPE:-current} slurm/r5_p10.sbatch)
S=$(sbatch --parsable --dependency=afterany:${L} --array=0-1%2 --export=ALL,STAGE=stage2,RECIPE=${RECIPE:-current} slurm/r5_p10.sbatch)
RP=$(sbatch --parsable --dependency=afterany:${S} --export=ALL,CMD=p10_report slurm/r5_cpu.sbatch)
echo "P10 ladder=${L} stage2=${S} report=${RP}"
