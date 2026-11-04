#!/bin/bash
#SBATCH --job-name=paperB-r10-fin
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/r10fin_%j.out
#SBATCH --error=logs/r10fin_%j.err
set -euo pipefail
cd /data2/cmdir/home/toandq/CSG-Skin-paperB
bash scripts/r10_finalize.sh
