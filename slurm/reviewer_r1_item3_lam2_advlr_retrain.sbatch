#!/bin/bash
#SBATCH --job-name=paperB-i3-l2
#SBATCH --array=0-5%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/reviewer_r1_item3_lam2_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/reviewer_r1_item3_lam2_%A_%a.err
#
# Item 3 (corrected): genuine retrain @ λ=2, adv_lr ×10 and ×100, 3 seeds (6 runs).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
MULTS=(10 10 10 100 100 100)
SEEDS=(42 52 62 42 52 62)
TID=${SLURM_ARRAY_TASK_ID}
LAM=2
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"

RUN="runB_orth1_ladv2_s${SEEDS[$TID]}_advlr${MULTS[$TID]}"
echo "=== lam2 advlr retrain task=${TID} ${RUN} mult=${MULTS[$TID]} seed=${SEEDS[$TID]} start $(date) ==="

# Drop prior re-score-only artifacts (Phase-2 ckpt reuse bug, fixed 2026-10-06).
rm -f "${ROOT}/results/paperB/phase3_sweep/${RUN}/summary.json"
rm -rf "${ROOT}/checkpoints/phase3/${RUN}"

${PY} "${ROOT}/scripts/train_phase3_sweep.py" \
  --lambda_adv "${LAM}" --seed "${SEEDS[$TID]}" \
  --adv_lr_multiplier "${MULTS[$TID]}" \
  --num_workers 4 --max_epochs 40

python3 - <<PY
import json, sys
from pathlib import Path
p = Path("${ROOT}/results/paperB/phase3_sweep/${RUN}/summary.json")
d = json.loads(p.read_text())
if d.get("reused_phase2_ckpt"):
    sys.exit("FAIL: still reused Phase-2 ckpt — not a fresh train")
print("OK reused_phase2_ckpt=", d.get("reused_phase2_ckpt"), "ckpt=", d.get("best_checkpoint", "")[:120])
PY

echo "=== task=${TID} done $(date) ==="
