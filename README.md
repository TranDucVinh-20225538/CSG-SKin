# CSG-Skin Paper B working tree

The repo at `/data2/hpcshared/Vinh/CSG-Skin` is owned by `anhnv` and is **not writable** by `toandq`. This directory holds all Paper B scripts and outputs. Existing `results/`, `results/cbm_revision/`, and `results/effb3_control/` are never modified.

| Role | Path |
|---|---|
| Read-only repo | `/data2/hpcshared/Vinh/CSG-Skin` |
| Phase 0+ outputs | `results/paperB/` |
| New scripts | `scripts/` |
| Python env | `.venv` (torch-env + lightning 2.2.5 / torchmetrics 1.4.1) |

Phase 0: **all gates PASS**. See `results/paperB/PHASE0_CHECKS.md`.
Phase 1: submitted via `scripts/submit_phase1.sh`. Log: `logs/phase1_ood.log`.
