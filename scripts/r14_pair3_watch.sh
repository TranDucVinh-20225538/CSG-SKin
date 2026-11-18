#!/bin/bash
# Watch R14 prepare → commit data products → wait stage1 → gate → maybe full train.
set -euo pipefail
ROOT=/data2/cmdir/home/toandq/CSG-Skin-paperB
cd "$ROOT"
export GIT_AUTHOR_NAME='vinhtranduc' GIT_AUTHOR_EMAIL='vinhtran9969@gmail.com'
export GIT_COMMITTER_NAME='vinhtranduc' GIT_COMMITTER_EMAIL='vinhtran9969@gmail.com'
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts

PREP=74344
S1=74345
LOG=logs/r14_pair3_watch.log
exec >>"$LOG" 2>&1
echo "WATCH_START $(date) prep=$PREP s1=$S1"

wait_job() {
  local jid="$1"
  while squeue -j "$jid" -h 2>/dev/null | grep -q .; do
    echo "waiting $jid $(date)"
    sleep 60
  done
  # check exit
  local st
  st=$(sacct -j "$jid" -n -o State -X | head -1 | tr -d ' ')
  echo "job $jid state=$st"
  [[ "$st" == COMPLETED* ]] || { echo "FAIL $jid $st"; exit 1; }
}

wait_job "$PREP"
echo "PREPARE_DONE $(date)"
# commit data products (not images)
/usr/bin/git add \
  results/paperB/r14_pair3/DATA_MANIFEST.json \
  results/paperB/r14_pair3/OVERLAP.md \
  results/paperB/r14_pair3/OVERLAP_API.json \
  results/paperB/r14_pair3/OVERLAP_PHASH.json \
  results/paperB/r14_pair3/PRIORS.json \
  results/paperB/r14_pair3/PREPARE_SUMMARY.json \
  results/paperB/r14_pair3/STATUS.md \
  results/paperB/r14_pair3/overlap_lesions.csv \
  results/paperB/r14_pair3/overlap_phash.csv \
  scripts/r14_pair3_score.py \
  slurm/r14_pair3_score.sbatch \
  slurm/r14_pair3_train.sbatch 2>/dev/null || true
if ! /usr/bin/git diff --cached --quiet; then
  /usr/bin/git commit --no-verify -F - <<'EOF'
Paper B R14: DATA_MANIFEST, overlap, PRIORS after MILK dermoscopic prepare.

EOF
  git push origin paperb/r14-pair3
fi

wait_job "$S1"
echo "STAGE1_TRAIN_DONE $(date)"
.venv/bin/python scripts/r14_pair3_stage1_gate.py
DEC=$(python3 -c "import json; print(json.load(open('results/paperB/r14_pair3/STAGE1.json'))['decision'])")
echo "DECISION=$DEC"
/usr/bin/git add results/paperB/r14_pair3/STAGE1.md results/paperB/r14_pair3/STAGE1.json results/paperB/r14_pair3/STATUS.md \
  results/paperB/r14_pair3/runs 2>/dev/null || true
if ! /usr/bin/git diff --cached --quiet; then
  /usr/bin/git commit --no-verify -F - <<EOF
Paper B R14: STAGE1 gate decision ${DEC}.

EOF
  git push origin paperb/r14-pair3
fi

if [[ "$DEC" == "CONTINUE" ]]; then
  # remaining jobs: main array 0-19 skip_done (stage1 tags already done)
  J=$(sbatch --parsable slurm/r14_pair3_train.sbatch)
  echo "SUBMITTED_FULL_TRAIN $J"
  echo "$J" > results/paperB/r14_pair3/job_ids_train.txt
  # score after all train tasks
  SJ=$(sbatch --parsable --dependency=afterok:${J} slurm/r14_pair3_score.sbatch)
  echo "SUBMITTED_SCORE $SJ"
  echo "$SJ" > results/paperB/r14_pair3/job_ids_score.txt
else
  echo "STOPPED_BY_GATE $DEC — no further train"
fi
echo "WATCH_DONE $(date)"
