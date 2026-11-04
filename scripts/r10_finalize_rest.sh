#!/bin/bash
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1

echo "=== G1: R1 report (--no_boot; means + SCIN; CI later optional) ==="
$PY scripts/r6_report.py --item r1 --no_boot

echo "=== G1: leakage-matched ==="
$PY scripts/r6_r1_leakage_matched.py

echo "=== G1: clinical post hoc ==="
$PY scripts/r5_posthoc_scoring.py clinical

echo "=== G1: R9 E1 report (--no_boot) ==="
$PY scripts/r9_e1_report.py --no_boot

echo "=== G2: T2 report ==="
$PY scripts/r7_t2_report.py

echo "=== C4: R2 SCIN ==="
$PY scripts/r6_r2_scin_report.py || true

echo "=== r9 number store ==="
$PY scripts/build_r9_number_store.py || true

echo "DONE r10 finalize rest"
