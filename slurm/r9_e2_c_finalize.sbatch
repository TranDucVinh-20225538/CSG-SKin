#!/bin/bash
#SBATCH --job-name=paperB-r9-e2c-fin
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r9e2c_fin_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r9e2c_fin_%j.err
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
set +e
"${ROOT}/.venv/bin/python" scripts/train_r9_e2.py --stage gateC
RC=$?
set -e
"${ROOT}/.venv/bin/python" scripts/r9_e2_report.py
"${ROOT}/.venv/bin/python" scripts/r9_finalize.py
if [[ "${RC}" -eq 2 ]]; then
  echo "Gate C not met — E2 stops"
  exit 0
fi
exit "${RC}"
