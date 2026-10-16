#!/bin/bash
# Paper B R5 P11 launcher (decisions/paperB_r5_p11.md). Prepared under the compute HOLD; not executed.
# Usage: GO=1 [RECIPE=auto|current|p8] bash scripts/launch_r5_p11.sh scan  [AFTER_JOBID]
#        GO=1 [RECIPE=...]             bash scripts/launch_r5_p11.sh seeds [AFTER_JOBID]
#        GO=1 [RECIPE=auto]            bash scripts/launch_r5_p11.sh full  [AFTER_JOBID]
#          full = scan → select → auto-submit seeds → final report (prints SEEDS_REPORT job id for P7(b) gating)
# Queue (amendment 2026-10-16): after P9, before P7(b).
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs
if [[ "${GO:-0}" != "1" ]]; then echo "compute HOLD: set GO=1 to submit"; exit 1; fi
STAGE="${1:?stage: scan | seeds | full}"
RECIPE="${RECIPE:-auto}"
DEP=()
if [[ -n "${2:-}" ]]; then DEP=(--dependency=afterany:$2); fi
OUT="results/paperB/r5/p11"
if [[ "${RECIPE}" != "current" && "${RECIPE}" != "auto" ]]; then OUT="${OUT}/recipe_rec${RECIPE}"; fi

submit_scan() {
  C=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-0 --job-name=paperB-r5-p11-ctl \
      --export=ALL,STAGE=p11_control,RECIPE=${RECIPE} slurm/r5_dual_gpu.sbatch)
  A=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-7%2 --job-name=paperB-r5-p11-scan \
      --export="ALL,STAGE=p11_scan,PENALTY_LIST=mkmmd coral,WEIGHT_LIST=1 3 10 30,EPOCHS=0,RECIPE=${RECIPE}" slurm/r5_dual_gpu.sbatch)
  L=$(sbatch --parsable --dependency=afterany:${A} --array=0-1%2 --job-name=paperB-r5-p11-scan80 \
      --export="ALL,STAGE=p11_scan,PENALTY_LIST=mkmmd coral,WEIGHT_LIST=10,EPOCHS=80,RECIPE=${RECIPE}" slurm/r5_dual_gpu.sbatch)
  SE=$(sbatch --parsable --dependency=afterany:${C}:${A}:${L} --job-name=pB-p11-select \
      --export=ALL,CMD=p11_select,RECIPE=${RECIPE} slurm/r5_cpu.sbatch)
  RP=$(sbatch --parsable --dependency=afterany:${SE} --job-name=pB-p11-report \
      --export=ALL,CMD=p11_report,RECIPE=${RECIPE} slurm/r5_cpu.sbatch)
  echo "P11 control=${C} scan=${A} scan80=${L} select=${SE} report=${RP}"
}

submit_seeds() {
  local AFTER="${1:-}"
  SEL="${OUT}/SELECTION.json"
  [[ -f "${SEL}" ]] || { echo "${SEL} missing: run the scan stage first"; exit 1; }
  local SDEP=()
  if [[ -n "${AFTER}" ]]; then SDEP=(--dependency=afterany:${AFTER}); fi
  JOBS=()
  for PEN in mkmmd coral; do
    read -r W EP E <<< "$("${ROOT}/.venv/bin/python" -c "import json; v=json.load(open('${SEL}')).get('${PEN}') or {}; print(v.get('weight', ''), v.get('epochs_flag', ''), v.get('epochs', ''))")"
    if [[ -z "${W}" ]]; then echo "${PEN}: nothing selected; skipped"; continue; fi
    J=$(sbatch --parsable ${SDEP[@]+"${SDEP[@]}"} --array=1-4%2 --job-name=paperB-r5-p11-seeds \
        --export=ALL,STAGE=p11_seeds,PENALTY=${PEN},WEIGHT=${W},EPOCHS=${EP},RECIPE=${RECIPE} slurm/r5_dual_gpu.sbatch)
    echo "P11 seeds ${PEN} w=${W} epochs=${E}: ${J}"
    JOBS+=("${J}")
  done
  if (( ${#JOBS[@]} )); then
    RP=$(sbatch --parsable --dependency=afterany:$(IFS=:; echo "${JOBS[*]}") --job-name=pB-p11-seeds-rep \
        --export=ALL,CMD=p11_report,RECIPE=${RECIPE} slurm/r5_cpu.sbatch)
    echo "seeds_report=${RP}"
  else
    echo "seeds_report=none"
  fi
}

case "${STAGE}" in
  scan) submit_scan ;;
  seeds) submit_seeds "${2:-}" ;;
  full)
    submit_scan
    SE=$(squeue -u "$USER" -n pB-p11-select -h -o %A | head -1)
    [[ -n "${SE}" ]] || { echo "could not find pB-p11-select job"; exit 1; }
    WRAP=$(sbatch --parsable --dependency=afterok:${SE} --job-name=pB-p11-after-sel --cpus-per-task=2 --mem=4G --time=00:30:00 \
      --export=ALL,RECIPE=${RECIPE},P7B_JOB=67551 \
      --wrap="bash ${ROOT}/scripts/r5_p11_after_select.sh")
    echo "P11 after_select=${WRAP} (launches seeds; rewires P7b 67551 to seeds_report)" ;;
  *) echo "unknown stage ${STAGE}"; exit 2 ;;
esac
