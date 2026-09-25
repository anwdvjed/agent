# Encoder comparison

Synthetic: True. One training seed; not evidence of real-world superiority.

| Variant | Parameters | Train seconds | Warm median ms | Warm p95 ms | Bin 1 Brier |
|---|---:|---:|---:|---:|---:|
| gru | 23260 | 2.00 | 0.45 | 0.61 | 0.0755 |
| loop1 | 29759 | 1.78 | 1.10 | 1.18 | 0.0755 |
| loop2 | 29759 | 1.89 | 1.51 | 1.61 | 0.0755 |
| loop4 | 29759 | 2.12 | 2.40 | 2.49 | 0.0755 |
| no_time | 29759 | 1.89 | 1.40 | 1.49 | 0.0755 |
| temporal_only | 29759 | 1.72 | 0.96 | 1.09 | 0.0755 |
| graph_only | 29759 | 1.76 | 1.15 | 1.25 | 0.0755 |
