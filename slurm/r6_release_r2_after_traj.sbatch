#!/bin/bash
#SBATCH --job-name=pB-r2-after-traj
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6_r2_after_traj_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6_r2_after_traj_%j.err
# Re-release held R2 after the 2-run P10 trajectory smoke finishes.

set -euo pipefail
echo "=== release R2 after traj smoke $(date) ==="
scontrol release 68562_[20-39] 2>/dev/null || scontrol release 68562 || true
squeue -j 68562 -o '%.18i %.2t %R' | head -20
echo "=== done $(date) ==="
