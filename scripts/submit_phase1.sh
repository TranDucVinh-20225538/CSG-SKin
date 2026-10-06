#!/usr/bin/env bash
# Paper B Phase 1 — dual-branch OOD, inference only. One GPU, resume-safe.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
LOG="${ROOT}/logs/phase1_ood.log"
mkdir -p "${ROOT}/logs"
echo "=== Paper B Phase 1 submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun --gres=gpu:1 -c 8 --mem=48G -t 06:00:00 \
  --job-name=paperB-p1 \
  bash -lc "
    set -euo pipefail
    cd ${ROOT}
    export PYTHONUNBUFFERED=1
    export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin
    echo '=== Phase 1 start' \$(date)
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} ${ROOT}/scripts/eval_ood_dual_branch.py --skip_done
    echo '=== Phase 1 done' \$(date)
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
