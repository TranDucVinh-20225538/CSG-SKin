#!/usr/bin/env bash
# Phase 1.5b — z_context / z_lesion score distributions + PCA AUROC. One GPU, short.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
LOG="${ROOT}/logs/phase15b_margin.log"
mkdir -p "${ROOT}/logs"
echo "=== Paper B Phase 1.5b submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun --gres=gpu:1 -c 8 --mem=48G -t 02:00:00 \
  --job-name=paperB-p15b \
  bash -lc "
    set -euo pipefail
    cd ${ROOT}
    export PYTHONUNBUFFERED=1
    export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
    echo '=== Phase 1.5b start' \$(date)
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} ${ROOT}/scripts/eval_separation_margin.py --seed 42 --method runB_orth1
    echo '=== Phase 1.5b done' \$(date)
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
