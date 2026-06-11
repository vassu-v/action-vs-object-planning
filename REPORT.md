# Scaling Experiment Report

## Overview
This experiment validates the theoretical prediction that the maximum number of objects $N^*$ that can be grounded in a mean-pooled representation scales with the embedding dimension $d$.

## Results

| d | N | Success Rate |
|---|---|--------------|
| 64 | 5 | 100.0% |
| 64 | 8 | 100.0% |
| 64 | 10 | 80.0% |
| 64 | 12 | 60.0% |
| 64 | 15 | 20.0% |
| 64 | 20 | 0.0% |
| 128 | 9 | 100.0% |
| 128 | 15 | 100.0% |
| 128 | 20 | 80.0% |
| 128 | 25 | 40.0% |
| 128 | 30 | 20.0% |
| 128 | 40 | 0.0% |
| 256 | 9 | 100.0% |
| 256 | 20 | 100.0% |
| 256 | 30 | 100.0% |
| 256 | 40 | 80.0% |
| 256 | 50 | 40.0% |

## Summary
### Predicted vs Observed $N^*$
| d | Predicted $N^*$ | Observed Collapse Point (approx) |
|---|----------------|---------------------------------|
| 64 | 12 | 12-15 |
| 128 | 22-23 | 25-30 |
| 256 | 40-41 | 40-50 |

The empirical data demonstrates a clear rightward shift of the grounding collapse point as $d$ increases. For $d=64$, performance begins to degrade after $N=10$, while for $d=256$, the model maintains high success rates up to $N=40$. This aligns with the theoretical prediction derived from the Gaussian embedding approximation.
