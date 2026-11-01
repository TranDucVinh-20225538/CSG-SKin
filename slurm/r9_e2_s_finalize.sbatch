#!/bin/bash
#SBATCH --job-name=paperB-r9-e2s-fin
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r9e2s_fin_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r9e2s_fin_%j.err
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
# selS exits 2 if gate S not met (expected stop); still write reports.
set +e
"${ROOT}/.venv/bin/python" scripts/train_r9_e2.py --stage selS
SEL_RC=$?
set -e
"${ROOT}/.venv/bin/python" scripts/r9_e2_report.py
"${ROOT}/.venv/bin/python" scripts/r9_finalize.py
if [[ "${SEL_RC}" -eq 2 ]]; then
  echo "Gate S not met — E2 stops (competence gap not closed by 80-epoch schedule)"
  exit 0
fi
if [[ "${SEL_RC}" -ne 0 ]]; then
  exit "${SEL_RC}"
fi
echo "Gate S met — submit stage C next"
exit 0
