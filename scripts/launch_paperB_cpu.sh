#!/bin/bash
# Paper B R5 P1–P6, P12, P13, Q4, Q5 and R6 R3–R5: CPU submissions with dependencies. NOT executed by the prep agent.
# HOLD: run only after the user says go and the round-7 A5 jobs have finished:  GO=1 bash scripts/launch_paperB_cpu.sh
# GPU stages (P4 adversary LR, frozen-encoder extraction) are separate commands at the bottom, also behind GO=1.
set -euo pipefail
[ "${GO:-0}" = 1 ] || { echo "HOLD: set GO=1 after the user's go-ahead"; exit 1; }
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs
CPU=slurm/paperB_r5r6_cpu.sbatch
sub() { sbatch --parsable "$@"; }
step() { local s="$1"; shift; sub --job-name="pB-${s}" --export=ALL,STEP="${s}" "$@" "${CPU}"; }

# Stage 0: metadata (frame lists) and independent cached-feature items.
J_FRAMES=$(step frames --cpus-per-task=2 --mem=8G --time=00:30:00)
J_P1=$(step p1)
J_P2=$(step p2 --cpus-per-task=4 --time=01:00:00)
J_P4=$(step p4)
J_P6=$(step p6)
J_R5=$(step r5 --cpus-per-task=2 --mem=8G --time=00:30:00)
J_TOY=$(step r3_toy --cpus-per-task=24 --mem=32G --time=06:00:00)

# P13: hash the 3,887 Fitzpatrick images, then re-bootstrap; P3 (mapped R50 features) waits for the clusters.
J_HASH=$(step p13_hash --cpus-per-task=4 --mem=16G --time=01:00:00)
J_P13=$(step p13_boot --dependency=afterok:${J_HASH})
J_P3=$(step p3 --dependency=afterok:${J_HASH} --export=ALL,STEP=p3,ARGS="--source mapped")

# P5: single-encoder λ = 4 features (CPU), then scoring.
J_P5X=$(sub slurm/r5_p5_extract.sbatch)
J_P5=$(step p5_score --dependency=afterok:${J_P5X} --cpus-per-task=4)

# Q4 / R4: manifest after the P5 features exist (adds the single-encoder family), then both analyses.
J_MAN=$(step q4_manifest --dependency=afterok:${J_P5X} --cpus-per-task=1 --mem=4G --time=00:15:00)
J_Q4=$(step q4 --dependency=afterok:${J_MAN})
J_R4=$(step r4 --dependency=afterok:${J_MAN})

echo "submitted: frames=${J_FRAMES} p1=${J_P1} p2=${J_P2} p4=${J_P4} p6=${J_P6} r5=${J_R5} toy=${J_TOY} p13hash=${J_HASH}"
echo "           p13=${J_P13} p3=${J_P3} p5x=${J_P5X} p5=${J_P5} manifest=${J_MAN} q4=${J_Q4} r4=${J_R4}"

# ---- GPU stages (run separately, behind the GPU queue order; GPU arrays use %2) ----
if [ "${GPU:-0}" = 1 ]; then
  # P4 GPU part: λ = 2, adversary LR ×10/×100, 3 seeds (≈9.6 GPU-h), then its report.
  J_ADV=$(sub slurm/r5_p4_advlr.sbatch)
  step p4_advlr_report --dependency=afterok:${J_ADV} --cpus-per-task=4
  # Frozen encoders (≈1 GPU-h total for R50 + DINOv2-B/L), then P3 full frames and P12 (MONET included).
  J_FRZ=$(sub --dependency=afterok:${J_FRAMES} slurm/r5_frozen_extract.sbatch)
  step p3 --dependency=afterok:${J_FRZ} --export=ALL,STEP=p3,ARGS="--source both"
  step p12 --dependency=afterok:${J_FRZ}
fi
# Q5 (after Q1–Q3 runs exist and their run lists are written as manifest extras):
#   GO=1 sbatch --export=ALL,STEP=q4_manifest,ARGS="--extra results/paperB/r5/q4/extra_q1q3.csv" slurm/paperB_r5r6_cpu.sbatch
#   GO=1 sbatch --export=ALL,STEP=q5,ARGS="--frozen cxr_pneumonia=results/paperB/r5/frozen/features_cxr/imagenet_r50" slurm/paperB_r5r6_cpu.sbatch
#   then re-run STEP=q4 and STEP=r4 for the all-runs correlation and leave-one-task-out.
