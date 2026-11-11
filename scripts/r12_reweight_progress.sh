#!/bin/bash
ROOT=/data2/cmdir/home/toandq/CSG-Skin-paperB
OUT=$ROOT/results/paperB/r12_reweight
n=$(ls -1 "$OUT"/runs/*/summary.json 2>/dev/null | wc -l)
echo "summaries=$n/45"
squeue -u toandq | rg 'r12|paperB-r' || true
ls "$OUT"/runs 2>/dev/null | wc -l
