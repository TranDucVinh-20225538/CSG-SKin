#!/bin/bash
# Paper B R5 P11 launcher (decisions/paperB_r5_p11.md). Prepared under the compute HOLD; not executed.
# Usage: GO=1 [RECIPE=current|p8] bash scripts/launch_r5_p11.sh scan  [AFTER_JOBID]
#        GO=1 [RECIPE=...]        bash scripts/launch_r5_p11.sh seeds [AFTER_JOBID]   (after SELECTION.json exists)
# scan : control (no penalty, λ_adv = 0, seed 42) + {mkmmd, coral} × {1, 3, 10, 30} at recipe epochs
#        + {mkmmd, coral} × {10} at 80 epochs (seed 42) -> select (writes SELECTION.json) -> report
# seeds: per penalty, the selected (weight, epochs) at seeds 52, 62, 72, 82 (+ 42 already in the scan) -> report
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs
if [[ "${GO:-0}" != "1" ]]; then echo "compute HOLD: set GO=1 to submit"; exit 1; fi
STAGE="${1:?stage: scan | seeds}"
RECIPE="${RECIPE:-current}"
DEP=()
if [[ -n "${2:-}" ]]; then DEP=(--dependency=afterany:$2); fi
OUT="results/paperB/r5/p11"
if [[ "${RECIPE}" != "current" ]]; then OUT="${OUT}/recipe_rec${RECIPE}"; fi
case "${STAGE}" in
  scan)
    C=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-0 --export=ALL,STAGE=p11_control,RECIPE=${RECIPE} slurm/r5_dual_gpu.sbatch)
    A=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-7%2 \
        --export="ALL,STAGE=p11_scan,PENALTY_LIST=mkmmd coral,WEIGHT_LIST=1 3 10 30,EPOCHS=0,RECIPE=${RECIPE}" slurm/r5_dual_gpu.sbatch)
    L=$(sbatch --parsable --dependency=afterany:${A} --array=0-1%2 \
        --export="ALL,STAGE=p11_scan,PENALTY_LIST=mkmmd coral,WEIGHT_LIST=10,EPOCHS=80,RECIPE=${RECIPE}" slurm/r5_dual_gpu.sbatch)
    SE=$(sbatch --parsable --dependency=afterany:${C}:${A}:${L} --export=ALL,CMD=p11_select,RECIPE=${RECIPE} slurm/r5_cpu.sbatch)
    RP=$(sbatch --parsable --dependency=afterany:${SE} --export=ALL,CMD=p11_report,RECIPE=${RECIPE} slurm/r5_cpu.sbatch)
    echo "P11 control=${C} scan=${A} scan80=${L} select=${SE} report=${RP}" ;;
  seeds)
    SEL="${OUT}/SELECTION.json"
    [[ -f "${SEL}" ]] || { echo "${SEL} missing: run the scan stage first"; exit 1; }
    JOBS=()
    for PEN in mkmmd coral; do
      read -r W EP E <<< "$("${ROOT}/.venv/bin/python" -c "import json; v=json.load(open('${SEL}')).get('${PEN}') or {}; print(v.get('weight', ''), v.get('epochs_flag', ''), v.get('epochs', ''))")"
      if [[ -z "${W}" ]]; then echo "${PEN}: nothing selected; skipped"; continue; fi
      J=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=1-4%2 \
          --export=ALL,STAGE=p11_seeds,PENALTY=${PEN},WEIGHT=${W},EPOCHS=${EP},RECIPE=${RECIPE} slurm/r5_dual_gpu.sbatch)
      echo "P11 seeds ${PEN} w=${W} epochs=${E}: ${J}"
      JOBS+=("${J}")
    done
    if (( ${#JOBS[@]} )); then
      RP=$(sbatch --parsable --dependency=afterany:$(IFS=:; echo "${JOBS[*]}") --export=ALL,CMD=p11_report,RECIPE=${RECIPE} slurm/r5_cpu.sbatch)
      echo "P11 report=${RP}"
    fi ;;
  *) echo "unknown stage ${STAGE}"; exit 2 ;;
esac
