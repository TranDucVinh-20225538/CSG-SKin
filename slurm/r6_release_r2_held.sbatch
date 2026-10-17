#!/bin/bash
#SBATCH --job-name=pB-r2-release
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6_r2_release_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6_r2_release_%j.err
# After P10 shift arms + R2-prior: release held R2 tasks 16-39 (before R1-hi).

set -euo pipefail
echo "=== release R2 held 68562_[16-39] $(date) ==="
scontrol release 68562_[16-39] || scontrol release 68562 || true
squeue -j 68562 -o '%.18i %.2t %R' | head -20
echo "=== done $(date) ==="
