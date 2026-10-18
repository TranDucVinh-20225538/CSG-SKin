#!/bin/bash
#SBATCH --job-name=pB-rel-p10last
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_rel_last_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_rel_last_%j.err
# Release held posthoc-last array after remaining R2 finishes.

set -euo pipefail
scontrol release 68809 || true
squeue -j 68809 -o '%.18i %.8T %R' | head
