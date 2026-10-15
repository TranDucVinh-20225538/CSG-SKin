#!/bin/bash
#SBATCH --job-name=paperB-r5r6-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5r6cpu_%x_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5r6cpu_%x_%j.err
# Paper B R5 P1–P6, P12, P13, Q4, Q5 and R6 R3–R5 CPU steps. Select with STEP (sbatch --export=ALL,STEP=p1 ...).
# Extra arguments for the step go in ARGS (e.g. ARGS="--extra results/paperB/r5/q4/extra_r5.csv").

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
STEP="${STEP:?set STEP}"
ARGS="${ARGS:-}"
echo "=== paperB r5/r6 CPU step ${STEP} start $(date) host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
case "${STEP}" in
  frames)        ${PY} scripts/r5_frozen_frames.py ${ARGS} ;;
  p1)            ${PY} scripts/r5_p1_mechanism.py ${ARGS} ;;
  p2)            ${PY} scripts/r5_p2_confidence.py ${ARGS} ;;
  p3)            ${PY} scripts/r5_p3_frozen_monitor.py ${ARGS:---source both} ;;
  p4)            ${PY} scripts/r5_p4_robustness.py ${ARGS} ;;
  p4_advlr_report) ${PY} scripts/r5_p4_advlr_report.py ${ARGS} ;;
  p5_score)      ${PY} scripts/r5_p5_single_encoder.py --score ${ARGS} ;;
  p6)            ${PY} scripts/r5_p6_leakage_probe.py ${ARGS} ;;
  p12)           ${PY} scripts/r5_p12_frozen_encoders.py ${ARGS} ;;
  p13_hash)      ${PY} scripts/r5_p13_fitz_clusters.py --hash ${ARGS} ;;
  p13_boot)      ${PY} scripts/r5_p13_fitz_clusters.py --bootstrap ${ARGS} ;;
  q4_manifest)   ${PY} scripts/r5_q4_manifest.py ${ARGS} ;;
  q4)            ${PY} scripts/r5_q4_collapse.py ${ARGS} ;;
  q5)            ${PY} scripts/r5_q5_frozen_runs.py ${ARGS} ;;
  r3_toy)        ${PY} scripts/r6_r3_toy_labelshift.py --workers "${SLURM_CPUS_PER_TASK:-8}" ${ARGS} ;;
  r4)            ${PY} scripts/r6_r4_warning_signals.py ${ARGS} ;;
  r5)            ${PY} scripts/r6_r5_label_shift.py ${ARGS} ;;
  d1)            ${PY} scripts/r5_d1_domain_direction.py ${ARGS} ;;
  rescore)       ${PY} scripts/r5_rescore_heldout.py run ${SLURM_ARRAY_TASK_ID} ;;
  rescore_report) ${PY} scripts/r5_rescore_heldout.py report ;;
  ckaudit)       ${PY} scripts/r5_ckpt_audit.py compete ${SLURM_ARRAY_TASK_ID} ;;
  ckaudit_erm)   ${PY} scripts/r5_ckpt_audit.py erm ${SLURM_ARRAY_TASK_ID} ;;
  ckaudit_report) ${PY} scripts/r5_ckpt_audit.py report ;;
  p8traj_report) ${PY} scripts/r5_p8_traj.py report ;;
  p8rule)        ${PY} scripts/r5_main_recipe_rule.py ;;
  p8heldout)     ${PY} scripts/r5_p8_heldout.py ;;
  t1last)        ${PY} scripts/r5_t1_last.py run ${SLURM_ARRAY_TASK_ID} ;;
  t1last_report) ${PY} scripts/r5_t1_last.py report ;;
  *) echo "unknown STEP ${STEP}"; exit 2 ;;
esac
echo "=== done ${STEP} $(date) ==="
