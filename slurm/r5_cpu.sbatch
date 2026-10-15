#!/bin/bash
#SBATCH --job-name=paperB-r5-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5cpu_%x_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5cpu_%x_%j.err
# CPU step for R5 Q1/P9/P10/Q3/P11. CMD (env) = one of the commands below; RECIPE (env, default current).
#   q1_index        scripts/r5_q1_cxr_data.py --index
#   q1_report       scripts/r5_q1_report.py
#   p9_report       scripts/r5_p9_report.py
#   p10_report      scripts/r5_p10_report.py
#   q3_report       scripts/r5_q3_report.py
#   p11_report      scripts/r5_p11_report.py
#   p11_select      scripts/r5_p11_report.py --select
# Submit through the scripts/launch_r5_*.sh launchers.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
RECIPE="${RECIPE:-current}"
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
echo "=== R5 CPU ${CMD} recipe=${RECIPE} start $(date) host=$(hostname) ==="
case "${CMD}" in
  q1_index)    "${PY}" scripts/r5_q1_cxr_data.py --index ;;
  q1_report)   "${PY}" scripts/r5_q1_report.py --recipe "${RECIPE}" ;;
  p9_report)   "${PY}" scripts/r5_p9_report.py --recipe "${RECIPE}" ;;
  p10_report)  "${PY}" scripts/r5_p10_report.py ;;
  q3_report)   "${PY}" scripts/r5_q3_report.py --recipe "${RECIPE}" ;;
  p11_report)  "${PY}" scripts/r5_p11_report.py --recipe "${RECIPE}" ;;
  p11_select)  "${PY}" scripts/r5_p11_report.py --recipe "${RECIPE}" --select ;;
  *) echo "unknown CMD ${CMD}"; exit 2 ;;
esac
echo "=== done $(date) ==="
