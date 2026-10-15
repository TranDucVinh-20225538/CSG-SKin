#!/bin/bash
# Paper B R5 Q1 launcher (decisions/paperB_r5_q1.md). Prepared under the compute HOLD; not executed.
# Usage: GO=1 [RECIPE=current|p8] bash scripts/launch_r5_q1.sh gate  [AFTER_JOBID]
#        GO=1 [RECIPE=...]        bash scripts/launch_r5_q1.sh sweep [AFTER_JOBID]   (refuses unless GATE.json met)
#        GO=1 [RECIPE=...]        bash scripts/launch_r5_q1.sh dose  [AFTER_JOBID]   (refuses unless INVERSION.json inverted)
# gate : index (CPU) -> 3 GPU runs -> report (writes GATE.json)
# sweep: 12 GPU runs -> report (writes INVERSION.json)
# dose : 9 GPU runs (three-stream, λ=2, 3 shifts x 3 seeds) -> report. FORCE=1 overrides the GATE / INVERSION checks (log it as a deviation).
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs
if [[ "${GO:-0}" != "1" ]]; then echo "compute HOLD: set GO=1 to submit"; exit 1; fi
STAGE="${1:?stage: gate | sweep | dose}"
RECIPE="${RECIPE:-current}"
DEP=()
if [[ -n "${2:-}" ]]; then DEP=(--dependency=afterany:$2); fi
OUT="results/paperB/r5/q1"
if [[ "${RECIPE}" != "current" ]]; then OUT="${OUT}/recipe_rec${RECIPE}"; fi
check() {
  "${ROOT}/.venv/bin/python" -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if d.get('$2') is True else 1)"
}
case "${STAGE}" in
  gate)
    if [[ ! -f results/paperB/r5/q1/index/nih_train.csv ]]; then
      IX=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --export=ALL,CMD=q1_index slurm/r5_cpu.sbatch)
      DEP=(--dependency=afterok:${IX})
    fi
    G=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-2%2 --export=ALL,STAGE=gate,RECIPE=${RECIPE} slurm/r5_q1_gpu.sbatch)
    RP=$(sbatch --parsable --dependency=afterany:${G} --export=ALL,CMD=q1_report,RECIPE=${RECIPE} slurm/r5_cpu.sbatch)
    echo "Q1 gate=${G} report=${RP}" ;;
  sweep)
    if [[ "${FORCE:-0}" != "1" ]] && ! check "${OUT}/GATE.json" met; then
      echo "headroom gate not met (or GATE.json missing): see ${OUT}/REPORT.md for replacement OOD candidates"; exit 1
    fi
    S=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-11%2 --export=ALL,STAGE=sweep,RECIPE=${RECIPE} slurm/r5_q1_gpu.sbatch)
    RP=$(sbatch --parsable --dependency=afterany:${S} --export=ALL,CMD=q1_report,RECIPE=${RECIPE} slurm/r5_cpu.sbatch)
    echo "Q1 sweep=${S} report=${RP}" ;;
  dose)
    if [[ "${FORCE:-0}" != "1" ]] && ! check "${OUT}/INVERSION.json" inverted; then
      echo "Q1 λ=2 did not meet rule A on Shenzhen (or INVERSION.json missing): dose test not launched"; exit 1
    fi
    DO=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-8%2 --time=10:00:00 --export=ALL,STAGE=dose,RECIPE=${RECIPE} slurm/r5_q1_gpu.sbatch)
    RP=$(sbatch --parsable --dependency=afterany:${DO} --export=ALL,CMD=q1_report,RECIPE=${RECIPE} slurm/r5_cpu.sbatch)
    echo "Q1 dose=${DO} report=${RP}" ;;
  *) echo "unknown stage ${STAGE}"; exit 2 ;;
esac
