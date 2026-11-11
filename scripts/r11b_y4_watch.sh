#!/bin/bash
# Wait for Y4 screen; write selection; STOP. Do NOT submit main here.
set -euo pipefail
WT=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b
MAIN=/data2/cmdir/home/toandq/CSG-Skin-paperB
Y4JOB=${1:?}
LOG=$WT/results/paperB/r11b/logs/y4_watch.log
RUNS=$WT/results/paperB/r11b/y4/screen/runs
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:$WT/scripts:$MAIN/scripts
PY=$MAIN/.venv/bin/python
echo "watch y4 select-only $(date) job=$Y4JOB" | tee -a "$LOG"
while true; do
  n=$(ls -d "$RUNS"/*/summary.json 2>/dev/null | wc -l)
  st=$(sacct -j "$Y4JOB" --format=State -n -P 2>/dev/null | sort -u | tr '\n' ',' || true)
  echo "$(date) n=$n/4 states=$st" | tee -a "$LOG"
  if [ "$n" -ge 4 ]; then break; fi
  if echo "$st" | grep -Eq 'COMPLETED|FAILED|CANCELLED|TIMEOUT' && ! echo "$st" | grep -Eq 'RUNNING|PENDING'; then
    sleep 20; break
  fi
  sleep 120
done
$PY -u $WT/scripts/r11b_y4_select.py | tee -a "$LOG"
echo "Y4 SELECTION written. Commit+push before any main submit." | tee -a "$LOG"
