#!/bin/bash
# Wait for Y2 screen; write selection; STOP. Do NOT submit main here.
# Main may only be submitted after SELECTION.json is committed+pushed.
set -euo pipefail
WT=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b
MAIN=/data2/cmdir/home/toandq/CSG-Skin-paperB
Y2JOB=${1:?y2 screen jobid}
LOG=$WT/results/paperB/r11b/logs/y2_watch.log
RUNS=$WT/results/paperB/r11b/y2/screen/runs
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:$WT/scripts:$MAIN/scripts
PY=$MAIN/.venv/bin/python
echo "watch y2 select-only $(date) job=$Y2JOB" | tee -a "$LOG"
while true; do
  n=$(ls -d "$RUNS"/*/summary.json 2>/dev/null | wc -l)
  st=$(sacct -j "$Y2JOB" --format=State -n -P 2>/dev/null | sort -u | tr '\n' ',' || echo unknown)
  echo "$(date) screen_summaries=$n/40 states=$st" | tee -a "$LOG"
  if [ "$n" -ge 40 ]; then break; fi
  if echo "$st" | grep -Eq 'COMPLETED|FAILED|CANCELLED|TIMEOUT' && ! echo "$st" | grep -Eq 'RUNNING|PENDING'; then
    sleep 20; break
  fi
  sleep 60
done
$PY -u $WT/scripts/r11b_y2_select.py | tee -a "$LOG"
echo "SELECTION written. Do NOT submit main until SELECTION.json is committed+pushed." | tee -a "$LOG"
if [ ! -f $WT/results/paperB/r11b/y2/SELECTION.json ]; then
  echo "missing SELECTION.json" | tee -a "$LOG"; exit 1
fi
sel=$($PY -c "import json;print(json.load(open('$WT/results/paperB/r11b/y2/SELECTION.json'))['selected'])")
if [ "$sel" = "None" ]; then
  echo "no admissible → stop Y2 and Y3; no submit" | tee -a "$LOG"
  exit 0
fi
echo "admissible selected=$sel — commit/push SELECTION then submit main manually" | tee -a "$LOG"
