#!/bin/bash
# Wait for the R4 SLURM arrays, then post-process, run D, commit and push.
# Usage: nohup scripts/r4_after_jobs.sh JOBID... > /tmp/r4_after_jobs.log 2>&1 &
set -uo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
cd "${ROOT}"
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
JOBS=$(IFS=,; echo "$*")
while [[ -n "$(squeue -h -j "${JOBS}" 2>/dev/null)" ]]; do sleep 120; done
echo "=== jobs done $(date) ==="
sacct -X -n -j "${JOBS}" -o JobID%14,State%12,Elapsed | sort -u

$PY scripts/r4_phase13_per_run.py
$PY scripts/r3_d_aggregates.py
$PY scripts/r2_item4_sweep_report.py | tail -1
$PY scripts/r3_c_lesion_per_run.py
$PY scripts/r3_c_lesion_figures.py | tail -1
$PY scripts/r3_a2_bootstrap_lesion.py | tail -1
$PY scripts/r4_report.py
$PY scripts/r4_manifest.py
$PY scripts/verify_manuscript_numbers.py --manifest paper/number_manifest.json --allowlist paper/number_allowlist.json \
  --out results/paperB/r4/verify.md --fail-on manifest
echo "D exit=$?"

/usr/bin/git add scripts/r4_*.py scripts/r4_after_jobs.sh slurm/r4_*.sbatch scripts/r3_c_lesion_figures.py \
  scripts/r3_a2_bootstrap_lesion.py results/paperB/r4 results/paperB/r2/item4/runs results/paperB/r2/item4/SWEEP_REPORT.md \
  results/paperB/r2/item4/sweep_results.json results/paperB/r3/a2 results/paperB/r3/c results/paperB/figures \
  results/paperB/phase13/per_run results/paperB/phase13/per_run_aggregate.json results/paperB/phase6_xfer \
  results/paperB/phase3_sweep/sweep_aggregate.json paper/number_manifest.json paper/number_manifest_lesion_table1.json \
  paper/number_manifest_r4_mmd.json
/usr/bin/git add results/paperB/phase35_mech 2>/dev/null
printf 'R4: lesion Table 1 at n=5 (all lambda), image-level lambda 0.25/1 rows at n=5, ResNet-50 0.994 source, MMD arm on pad_heldout; manifest extended, D rerun\n' > /tmp/csg_msg_r4
/usr/bin/git commit -q -F /tmp/csg_msg_r4 && /usr/bin/git pull -q --rebase origin paperb/reviewer-r1 && /usr/bin/git push -q origin paperb/reviewer-r1
/usr/bin/git log --oneline -1
echo "=== done $(date) ==="
