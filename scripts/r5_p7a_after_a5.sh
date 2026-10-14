#!/bin/bash
# Login-node watcher (light work only). Waits until the R7 A5 jobs are all finished (watcher submitted scoring +
# report and none of them is queued) and the SCIN download is complete, then submits P7(a) extraction (GPU array)
# and scoring (CPU, afterok).
ROOT=/data2/cmdir/home/toandq/CSG-Skin-paperB
ST=/data2/cmdir/home/toandq/r7work/a5/state
SCIN=$ROOT/data/unseen_clinical/scin
cd $ROOT
for n in $(seq 1 1440); do
  a5=0
  if [ -f $ST/scoring.jobid ]; then
    ids=$(tr ' ' ',' < $ST/scoring.jobid)
    squeue -h -j "$ids" 2>/dev/null | grep -q . || a5=1
  fi
  squeue -u $USER -h -o %j | grep -q "^r7-a5" && a5=0
  scin=0; grep -q "^done" $SCIN/fetch.log 2>/dev/null && scin=1
  echo "$(date '+%F %T') a5_done=$a5 scin_done=$scin"
  if [ $a5 = 1 ] && [ $scin = 1 ]; then
    .venv/bin/python scripts/r5_p7a_sets.py || { echo "STOP: sets failed"; exit 1; }
    e=$(sbatch --parsable slurm/r5_p7a_extract.sbatch)
    s=$(sbatch --parsable --dependency=afterok:$e slurm/r5_p7a_score.sbatch)
    echo "$(date) submitted extract $e score $s"; exit 0
  fi
  sleep 300
done
echo "timeout $(date)"
