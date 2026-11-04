#!/bin/bash
# R10 CPU follow-ups after G1–G3. Usage: bash scripts/r10_finalize.sh
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
mkdir -p logs results/paperB/r10

echo "=== C5 (fixed z alias) ==="
$PY scripts/r10_c2_c3_c5.py --items c5

echo "=== C1 + G1 class-aware probes (new cells) ==="
$PY scripts/r6_class_aware_heldout_probes.py

echo "=== G3 n50 λ=0.5 report + probes ==="
$PY scripts/r10_g3_report.py

echo "=== G1: R1 report ==="
$PY scripts/r6_report.py --item r1

echo "=== G1: leakage-matched ==="
$PY scripts/r6_r1_leakage_matched.py

echo "=== G1: clinical post hoc (r1 cells) ==="
$PY scripts/r5_posthoc_scoring.py clinical

echo "=== G1: R9 E1 report (shift refs) ==="
$PY scripts/r9_e1_report.py

echo "=== G2: T2 report ==="
$PY scripts/r7_t2_report.py

echo "=== C4: R2 SCIN report (if script present) ==="
$PY scripts/r6_r2_scin_report.py || true

echo "=== rebuild r9 number store ==="
$PY scripts/build_r9_number_store.py || true

echo "DONE r10 finalize"
