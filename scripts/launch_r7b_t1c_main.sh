#!/bin/bash
# Paper B R7b T1c main grid after SELECTION.json. Skips if dual chose ×0.1.
# Usage: GO=1 bash scripts/launch_r7b_t1c_main.sh
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
if [[ "${GO:-0}" != "1" ]]; then echo "set GO=1 to submit"; exit 1; fi
SEL=results/paperB/r7b/t1c/SELECTION.json
[[ -f "${SEL}" ]] || { echo "missing ${SEL}; run r7b_t1c_select.py first"; exit 1; }
STATUS=$(.venv/bin/python -c "import json; print(json.load(open('${SEL}'))['status'])")
if [[ "${STATUS}" == "stop_after_selection" ]]; then
  echo "dual chose ×0.1 — no main runs; writing stop report"
  .venv/bin/python scripts/r7b_t1c_report.py
  exit 0
fi
DUAL_M=$(.venv/bin/python -c "import json; print(json.load(open('${SEL}'))['dual_chosen_mult'])")
ERM_M=$(.venv/bin/python -c "import json; print(json.load(open('${SEL}'))['erm_chosen_mult'])")
RECIPE="${RECIPE:-current}"
VIT="${VIT:-dinov2_vits14}"
# ERM 5 seeds at ERM_M
ERM=$(sbatch --parsable --array=0-4%2 --export=ALL,STAGE=t1c_erm,BACKBONE=${VIT},RECIPE=${RECIPE},BB_MULT=${ERM_M} slurm/r7b_single_gpu.sbatch)
# dual λ=0 × 5 seeds and λ=2 × 5 seeds at DUAL_M (10 tasks). Use two arrays.
D0=$(sbatch --parsable --array=0-4%2 --export=ALL,STAGE=t1c_main,BACKBONE=${VIT},RECIPE=${RECIPE},BB_MULT=${DUAL_M},LAMBDA_ADV=0 slurm/r7b_dual_gpu.sbatch)
D2=$(sbatch --parsable --array=0-4%2 --export=ALL,STAGE=t1c_main,BACKBONE=${VIT},RECIPE=${RECIPE},BB_MULT=${DUAL_M},LAMBDA_ADV=2 slurm/r7b_dual_gpu.sbatch)
FIN=$(sbatch --parsable --dependency=afterany:${ERM}:${D0}:${D2} slurm/r7b_finalize.sbatch)
echo "T1c main erm=${ERM} dual0=${D0} dual2=${D2} finalize=${FIN} dual_m=${DUAL_M} erm_m=${ERM_M}"
