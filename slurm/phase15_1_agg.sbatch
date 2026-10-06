#!/bin/bash
#SBATCH --job-name=paperB-p151a
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase15_1_agg_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase15_1_agg_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/aggregate_phase15_1.py"
