# Phase 14.B — GRL sign check (30 minutes, then stop)

GRL graph is the standard one: head cosine vs no-GRL = 1.000 (not flipped), encoder cosine = -1.000 (flipped). The head's failure to leave ln(3) is not this sign error. Phase 12 stays inconclusive — implementation never produced a working adversary — and is not reported as a Camelyon result.

| seed | head cos (GRL vs no-GRL) | ∂z cos | encoder cos | ||g_head|| GRL | ||g_head|| no-GRL |
|---:|---|---|---|---|---|
| 0 | 1.0000 | -1.0000 | -1.0007 | 1.097 | 1.097 |
| 1 | 1.0000 | -1.0000 | -1.0006 | 1.658 | 1.658 |
| 2 | 1.0000 | -1.0000 | -1.0006 | 1.125 | 1.125 |

Expected if GRL is correct: head cosine ≈ +1, encoder cosine ≈ −1. Expected if the minus sign hits the head: head cosine ≈ −1.

Reopen WILDS: **no**. Drop Phase 12 from Results: **yes**.

**Phase 12 does not support a conclusion.** The adversary never trained (CE = ln 3 at every epoch). No Camelyon17 result is reported. Limitations: tried, implementation unresolved, left for later work. Not outcome (c) in the sense of 'Camelyon resists invariance'.
