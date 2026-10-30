# Paper B R7 / C2 — which run set each figure and clinical number uses (CPU)

Committed before the C2 job. Deviations appended at the end.

## Design

- For Figure 1, Figure 2 and every clinical percentage in §4.4 and App. H, list the run set (Table 1 last.ckpt runs, sweep runs, or R1 runs) and checkpoint.
- **`results/paperB/figures/FIGURES_REPORT.md` is stale** (still points at phase3/phase13). Audit against the current [`scripts/plot_manuscript_figures.py`](../scripts/plot_manuscript_figures.py): Fig. 1 on writing-r6 is already drawn from `t1last` (`load_t1last_by_lambda`).
- If any panel or number uses a different set from the rest, regenerate it from the Table 1 last.ckpt runs and report old vs new value.
- No `.tex` edits; the author updates the text.

## Outputs

`results/paperB/r7/c2/REPORT.md` (inventory table; old vs new if regenerated).

## Deviations
