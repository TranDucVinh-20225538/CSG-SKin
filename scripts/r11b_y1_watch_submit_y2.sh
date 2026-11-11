#!/bin/bash
set -euo pipefail
WT=/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b
MAIN=/data2/cmdir/home/toandq/CSG-Skin-paperB
OUT=$WT/results/paperB/r11b/y1
LOG=$WT/results/paperB/r11b/logs/y1_watch.log
Y1JOB=${1:?y1 jobid}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:$WT/scripts:$MAIN/scripts
PY=$MAIN/.venv/bin/python
echo "watch start $(date) y1job=$Y1JOB" | tee -a "$LOG"
while true; do
  n=$(ls "$OUT"/x*.json 2>/dev/null | wc -l)
  st=$(sacct -j "$Y1JOB" --format=State -n -P 2>/dev/null | sort -u | tr '\n' ',' || echo unknown)
  echo "$(date) n=$n/55 states=$st" | tee -a "$LOG"
  if [ "$n" -ge 55 ]; then
    break
  fi
  if echo "$st" | grep -Eq 'COMPLETED|FAILED|CANCELLED|TIMEOUT' && ! echo "$st" | grep -Eq 'RUNNING|PENDING'; then
    sleep 30
    n=$(ls "$OUT"/x*.json 2>/dev/null | wc -l)
    echo "job ended n=$n" | tee -a "$LOG"
    break
  fi
  sleep 60
done
$PY -u $WT/scripts/r11b_y1_diagnose.py --finalize | tee -a "$LOG"
echo "Y1 done $(date) n=$(ls "$OUT"/x*.json | wc -l)" | tee -a "$LOG"
JOB=$(sbatch --parsable $WT/slurm/r11b_y2_screen.sbatch)
echo "Y2_SCREEN_JOB=$JOB $(date)" | tee -a "$LOG"
echo "$JOB" > $WT/results/paperB/r11b/logs/y2_screen_jobid.txt
