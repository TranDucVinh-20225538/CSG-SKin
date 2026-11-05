# Paper B R11 — GPU-hour estimate (before first GPU job)

Basis (R9): **1.51 GPU-h** per three-stream run; **1.37 GPU-h** per two-stream 40-epoch run; **≈ 0.1 GPU-h** per forward pass of one run over all sets.

| item | plan | GPU-h |
|---|---|---:|
| X2 | M2 ERM checkpoints **on disk** (`checkpoints/r5_p8/erm_ref/`, 5× `last.ckpt`) — **no retrain**. Forwards: M2 missing mednode/sd198/scin × 5 seeds ≈ 5×0.1; M3 ImageNet frozen Eff-B3 one pass over all sets ≈ 0.1; clinical flagging uses cached λ=2 dual features. | **≈ 0.6–1.5** |
| X3 | Feature cache once (ERM frozen) ≈ 0.1; head+adversary 40ep: λ∈{0,0.5,2,8}×5 + shift0 λ=2×5 = 25 runs. Head-only ≈ 0.12–0.20 GPU-h/run → **≈ 3–5**. | **≈ 3–5** |
| X4 | 15 two-stream 40ep runs (λ∈{0,2}×5 shift100 + λ=2×5 shift0): 15×1.37 = **20.55**; post-train forwards 15×0.1 ≈ 1.5. | **≈ 21–22** |
| X5 | Uses X3 cache; λ∈{0,2}×3 seeds×5 t-levels = 30 light head runs + one synthetic forward. | **≈ 1–2** |
| **Total GPU** | | **≈ 26–31** |

CPU-only (no GPU): levels a, b, X1.

**M2 retrain contingency:** if checkpoints were missing, +5×1.37 ≈ **+6.9 GPU-h** (not needed: present).

X2 note for later decisions: Table 1 λ=0.5 uses seeds {42,43,44,45,46}; erm_ref seeds are {42,52,62,72,82} — only seed 42 overlaps for M2 pairing at λ=0.5.
