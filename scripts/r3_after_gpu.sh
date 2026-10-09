#!/bin/bash
# R3: after the lesion-level sweep (65152) and B1 (65189) leave the queue, regenerate every report that depends on
# them, commit and push. Usage: nohup scripts/r3_after_gpu.sh 65152 65189 65155 > /tmp/r3_after_gpu.log 2>&1 &
set -uo pipefail
ROOT=/data2/cmdir/home/toandq/CSG-Skin-paperB
cd "$ROOT"
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:$ROOT/scripts
PY=$ROOT/.venv/bin/python
JOBS=$(IFS=,; echo "$*")
while squeue -h -j "$JOBS" 2>/dev/null | grep -q .; do sleep 60; done
while pgrep -f r3_c_lesion_per_run.py > /dev/null; do sleep 30; done
echo "=== GPU jobs done $(date) ==="
sacct -j "$JOBS" --format=JobID%14,State,Elapsed -X
$PY scripts/r2_item4_sweep_report.py
$PY scripts/r3_c_lesion_per_run.py
$PY scripts/r3_b1_report.py
$PY scripts/r3_a1_identity.py | tail -1
$PY scripts/r3_a2_bootstrap_lesion.py | tail -1
$PY scripts/r3_c_lesion_figures.py
/usr/bin/git add results/paperB/r2/item4/SWEEP_REPORT.md results/paperB/r2/item4/sweep_results.json \
  results/paperB/r2/item4/runs results/paperB/r3/a1 results/paperB/r3/a2 results/paperB/r3/b1 results/paperB/r3/c \
  results/paperB/figures/fig1_intervention* results/paperB/figures/fig2_confidence* \
  results/paperB/figures/fig3_detectors* results/paperB/figures/fig5_leakage_no_mediation*
printf 'R3: lesion sweep complete + B1 (n=5): sweep report, A1, A2, B1, C figures regenerated\n' > /tmp/csg_msg_r3
/usr/bin/git commit -q -F /tmp/csg_msg_r3 && /usr/bin/git pull -q --rebase origin paperb/reviewer-r1 && /usr/bin/git push -q origin paperb/reviewer-r1
/usr/bin/git log --oneline -1
echo "=== done $(date) ==="
