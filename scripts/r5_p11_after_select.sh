#!/bin/bash
# Runs after p11_select: launches the seeds stage and rewires P7(b) to wait for the seeds report.
# Env: RECIPE, P7B_JOB (default 67551).
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
RECIPE="${RECIPE:-auto}"
P7B_JOB="${P7B_JOB:-67551}"
OUT=$("${ROOT}/.venv/bin/python" -c "
import r5_recipe, pathlib
r=r5_recipe.concrete('${RECIPE}')
p=pathlib.Path('results/paperB/r5/p11')
print(p if r in ('current','auto') else p/('recipe_rec'+r))
")
SEL="${OUT}/SELECTION.json"
[[ -f "${SEL}" ]] || { echo "missing ${SEL}"; exit 1; }
LOG=$(mktemp)
GO=1 RECIPE="${RECIPE}" bash scripts/launch_r5_p11.sh seeds | tee "${LOG}"
RP=$(awk -F= '/seeds_report=/{print $2}' "${LOG}" | tail -1)
echo "seeds_report=${RP}" | tee "${OUT}/SEEDS_REPORT_JOB.txt"
if [[ -n "${RP}" && "${RP}" != "none" ]]; then
  scontrol update jobid="${P7B_JOB}" dependency=afterany:"${RP}" || true
  scontrol release "${P7B_JOB}" || true
  echo "P7(b) ${P7B_JOB} now afterany:${RP}"
else
  # nothing to seed: release P7(b) after P9 only
  scontrol update jobid="${P7B_JOB}" dependency=afterany:68090 || true
  scontrol release "${P7B_JOB}" || true
  echo "no seeds; P7(b) released on afterany:68090"
fi
