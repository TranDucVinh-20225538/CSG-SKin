#!/bin/bash
# Paper B R5 P8 launcher (decisions/paperB_r5_p8.md). Prepared under the compute HOLD; not executed.
# Usage: GO=1 bash scripts/launch_r5_p8.sh [AFTER_JOBID]
# Chain: stage A (16 grid + 2 ERM refs) -> selA -> stage B (2) -> selB -> main (15) -> report -> ext (20, gated on
# RECIPE_DECISION.json inversion_survives) -> report.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs
if [[ "${GO:-0}" != "1" ]]; then echo "compute HOLD: set GO=1 to submit"; exit 1; fi
DEP=()
if [[ -n "${1:-}" ]]; then DEP=(--dependency=afterany:$1); fi
A=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-17%2 --export=ALL,STAGE=A slurm/r5_p8_gpu.sbatch)
SA=$(sbatch --parsable --dependency=afterok:${A} --export=ALL,STAGE=selA slurm/r5_p8_cpu.sbatch)
B=skipped  # 40 epochs fixed (LOCK_40_EPOCHS in train_r5_p8.py)
SB=$(sbatch --parsable --dependency=afterok:${SA} --export=ALL,STAGE=selB slurm/r5_p8_cpu.sbatch)
M=$(sbatch --parsable --dependency=afterok:${SB} --array=0-14%2 --export=ALL,STAGE=main slurm/r5_p8_gpu.sbatch)
RP=$(sbatch --parsable --dependency=afterany:${M} --export=ALL,STAGE=report slurm/r5_p8_cpu.sbatch)
E=$(sbatch --parsable --dependency=afterok:${RP} --array=0-19%2 --export=ALL,STAGE=ext slurm/r5_p8_gpu.sbatch)
RE=$(sbatch --parsable --dependency=afterany:${E} --export=ALL,STAGE=report slurm/r5_p8_cpu.sbatch)
echo "P8 stageA=${A} selA=${SA} stageB=${B} selB=${SB} main=${M} report=${RP} ext=${E} report2=${RE}"
echo "last P8 job: ${RE}"
