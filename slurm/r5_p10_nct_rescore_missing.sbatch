#!/bin/bash
#SBATCH --job-name=pB-p10-nct2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_nct_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_nct_%j.err
# Retry NCT rescore for missing induced s72/s82 (gate failed, descriptive only).
# NFS-safe: num_workers=0 + local scratch staging + retries.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== P10 NCT rescore missing induced 72/82 $(date) host=$(hostname) SLURM_TMPDIR=${SLURM_TMPDIR:-unset} ==="
# Same env as successful partial run 68724; fall back to torch-env if needed.
PY="${ROOT}/.venv/bin/python"
if ! "$PY" -c "import wilds,torch" 2>/dev/null; then
  PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
fi
echo "python=$PY"
"$PY" scripts/r5_p10_nct_rescore.py --device cpu --num_workers 0 \
  --local_scratch --retries 4 --retry_sleep 10 \
  --only induced_l2:72 --only induced_l2:82
echo "=== done $(date) ==="
