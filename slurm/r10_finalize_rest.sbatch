#!/bin/bash
#SBATCH --job-name=paperB-r10-fin2
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/r10fin2_%j.out
#SBATCH --error=logs/r10fin2_%j.err
set -euo pipefail
cd /data2/cmdir/home/toandq/CSG-Skin-paperB
bash scripts/r10_finalize_rest.sh
