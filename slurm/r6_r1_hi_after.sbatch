#!/bin/bash
#SBATCH --job-name=pB-r1hi-after
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6r1hi_after_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6r1hi_after_%j.err
# After R1-hi main wave: score probes/gates; if λ=8 gate empty, submit adv-LR×3 λ=4 (5 seeds).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
echo "=== R1-hi after start $(date) commit=$(git rev-parse --short HEAD) ==="
${PY} scripts/r6_r1_hi_report.py
GATE=$(${PY} scripts/r6_r1_hi_report.py --gate_check)
echo "gate_check ${GATE}"
NEED=$(${PY} -c "import json,sys; print(json.loads(sys.argv[1])['need_advlr3'])" "${GATE}")
if [ "${NEED}" = "True" ]; then
  echo "λ=8 gate empty → submitting advlr3 wave (5 seeds)"
  ADV=$(sbatch --parsable --export=ALL,WAVE=advlr3 --array=0-4%2 \
    "${ROOT}/slurm/r6_r1_hi.sbatch")
  # re-report after advlr3
  sbatch --dependency=afterany:${ADV}_* --export=ALL \
    --job-name=pB-r1hi-rep2 --cpus-per-task=8 --mem=48G --time=04:00:00 \
    --output="${ROOT}/logs/r6r1hi_rep2_%j.out" \
    --error="${ROOT}/logs/r6r1hi_rep2_%j.err" \
    --wrap="cd ${ROOT} && export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts CUDA_VISIBLE_DEVICES='' && ${PY} scripts/r6_r1_hi_report.py"
  echo "advlr3 job ${ADV}"
else
  echo "no advlr3 needed"
fi
echo "=== R1-hi after done $(date) ==="
