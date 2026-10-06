# B1 dense — interim (seed 42 only; 52/62 in flight)

Mahalanobis omitted (BCN→HAM baseline below chance). Metrics @ λ ∈ {0, 0.25, 1}.

| λ | Leakage bal | kNN-50 | Cosine |
|---|------------:|-------:|-------:|
| 0 | 0.969 | 0.704 | 0.606 |
| 0.25 | 0.799 | 0.494 | 0.474 |
| 1 | 0.799 | 0.467 | 0.459 |

Replace with mean±std when `dense3seed_aggregate.json` exists (seeds 42, 52, 62).
