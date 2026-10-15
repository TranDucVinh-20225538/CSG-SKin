#!/bin/bash
# Paper B R5 P9 launcher (decisions/paperB_r5_p9.md). Prepared under the compute HOLD; not executed.
# Usage: GO=1 [RECIPE=auto] bash scripts/launch_r5_p9.sh v2 [AFTER_JOBID]   (amendment 2026-10-15; replaces ladder)
#   v2: target -> round 1 (10, %2) -> round2 -> round 2 (6, %2) -> best -> 3 seeds (4, %2) -> go5 -> 5 seeds (2) -> report
#   prints the report job id last (P7b depends on it)
# Old:   GO=1 [RECIPE=current|p8|auto] bash scripts/launch_r5_p9.sh ladder [AFTER_JOBID]
#        GO=1 [RECIPE=...] BACKBONE=resnet50|effb3 bash scripts/launch_r5_p9.sh stage2 [AFTER_JOBID]
# ladder: two sequential 5-rung arrays (ResNet-50, EfficientNet-B3; one GPU each, %1) -> report (writes SELECTED_*).
# stage2: needs user approval (outside the 20 GPU-h cap); refuses unless SELECTED_<backbone>.json exists (FORCE=1
#         runs the baseline config instead; log as a deviation).
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs
if [[ "${GO:-0}" != "1" ]]; then echo "compute HOLD: set GO=1 to submit"; exit 1; fi
STAGE="${1:?stage: v2 | ladder | stage2}"
RECIPE="${RECIPE:-current}"
SUF=""
if [[ "${1:-}" == "stage2" ]]; then
  SUF=$("${ROOT}/.venv/bin/python" -c "import sys; sys.path.insert(0, 'scripts'); import r5_recipe; print(r5_recipe.tag('${RECIPE}'))")
fi
DEP=()
if [[ -n "${2:-}" ]]; then DEP=(--dependency=afterany:$2); fi
case "${STAGE}" in
  v2)
    C() { sbatch --parsable --dependency=afterany:$1 --job-name=pB-p9v2-$2 --export=ALL,CMD=p9v2_$2,RECIPE=${RECIPE} slurm/r5_cpu.sbatch; }
    G() { sbatch --parsable --dependency=afterany:$1 --array=$2 --job-name=paperB-r5-p9v2-$3 --export=ALL,STAGE=p9v2_$3,RECIPE=${RECIPE} slurm/r5_single_gpu.sbatch; }
    TG=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --job-name=pB-p9v2-target --export=ALL,CMD=p9v2_target slurm/r5_cpu.sbatch)
    R1=$(sbatch --parsable --dependency=afterok:${TG} --array=0-9%2 --job-name=paperB-r5-p9v2-r1 --export=ALL,STAGE=p9v2_r1,RECIPE=${RECIPE} slurm/r5_single_gpu.sbatch)
    S1=$(C ${R1} round2); R2=$(G ${S1} 0-5%2 r2); S2=$(C ${R2} best); R3=$(G ${S2} 0-3%2 s3)
    S3=$(C ${R3} go5); R5=$(G ${S3} 0-1%2 s5); RP=$(C ${R5} report)
    echo "P9v2 target=${TG} r1=${R1} round2=${S1} r2=${R2} best=${S2} s3=${R3} go5=${S3} s5=${R5} report=${RP}" ;;
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
