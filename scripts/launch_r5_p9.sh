#!/bin/bash
# Paper B R5 P9 launcher (decisions/paperB_r5_p9.md). Prepared under the compute HOLD; not executed.
# Usage: GO=1 [RECIPE=current|p8] bash scripts/launch_r5_p9.sh ladder [AFTER_JOBID]
#        GO=1 [RECIPE=...] BACKBONE=resnet50|effb3 bash scripts/launch_r5_p9.sh stage2 [AFTER_JOBID]
# ladder: two sequential 5-rung arrays (ResNet-50, EfficientNet-B3; one GPU each, %1) -> report (writes SELECTED_*).
# stage2: needs user approval (outside the 20 GPU-h cap); refuses unless SELECTED_<backbone>.json exists (FORCE=1
#         runs the baseline config instead; log as a deviation).
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs
if [[ "${GO:-0}" != "1" ]]; then echo "compute HOLD: set GO=1 to submit"; exit 1; fi
STAGE="${1:?stage: ladder | stage2}"
RECIPE="${RECIPE:-current}"
SUF=""
if [[ "${RECIPE}" != "current" ]]; then SUF="_rec${RECIPE}"; fi
DEP=()
if [[ -n "${2:-}" ]]; then DEP=(--dependency=afterany:$2); fi
case "${STAGE}" in
  ladder)
    A=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-4%1 --export=ALL,STAGE=p9_ladder,BACKBONE=resnet50,RECIPE=${RECIPE} slurm/r5_single_gpu.sbatch)
    B=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-4%1 --export=ALL,STAGE=p9_ladder,BACKBONE=effb3,RECIPE=${RECIPE} slurm/r5_single_gpu.sbatch)
    RP=$(sbatch --parsable --dependency=afterany:${A}:${B} --export=ALL,CMD=p9_report,RECIPE=${RECIPE} slurm/r5_cpu.sbatch)
    echo "P9 ladder resnet50=${A} effb3=${B} report=${RP}" ;;
  stage2)
    BB="${BACKBONE:?set BACKBONE=resnet50|effb3}"
    if [[ "${FORCE:-0}" != "1" && ! -f "results/paperB/r5/p9/SELECTED_${BB}${SUF}.json" ]]; then
      echo "no P9 rung reached the dual-encoder leakage level for ${BB}; stage 2 not launched"; exit 1
    fi
    S=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-14%2 --export=ALL,STAGE=p9_stage2,BACKBONE=${BB},RECIPE=${RECIPE} slurm/r5_single_gpu.sbatch)
    RP=$(sbatch --parsable --dependency=afterany:${S} --export=ALL,CMD=p9_report,RECIPE=${RECIPE} slurm/r5_cpu.sbatch)
    echo "P9 stage2 ${BB}=${S} report=${RP}" ;;
  *) echo "unknown stage ${STAGE}"; exit 2 ;;
esac
