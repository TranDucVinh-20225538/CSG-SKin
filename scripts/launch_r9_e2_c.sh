#!/bin/bash
# Paper B R9 E2 stage C (after Gate S). Usage: GO=1 bash scripts/launch_r9_e2_c.sh
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
if [[ "${GO:-0}" != "1" ]]; then echo "set GO=1 to submit"; exit 1; fi
test -f results/paperB/r9/e2/GATE_S.json
python3 -c "import json,sys; g=json.load(open('results/paperB/r9/e2/GATE_S.json')); sys.exit(0 if g.get('met') else 1)"
J=$(sbatch --parsable slurm/r9_e2_c_gpu.sbatch)
F=$(sbatch --parsable --dependency=afterok:${J} slurm/r9_e2_c_finalize.sbatch)
echo "R9 E2C=${J} FIN=${F}"
