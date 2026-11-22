# Stress tests A4 (noise) and A5b (tune only K or only V)

## A4. Run-to-run noise of CT-KV (split 87, 200 epochs)

| run | average |
|---|---|
| a3 (seed 0) | 44.90 |
| rep1 | 44.82 |
| rep2 | 44.95 |
| seed1 | 44.51 |
| seed2 | 44.37 |
| reproduction log (seed 0) | 44.86 |

- same seed 0 (GPU nondeterminism only), 4 runs: average 44.88 ± 0.06 (range 44.82–44.95); per-task std mean 0.22, max 0.53 (tweet_eval-stance_feminist)

- all runs incl. seeds 1, 2, 6 runs: average 44.73 ± 0.24 (range 44.37–44.95); per-task std mean 0.48, max 1.98 (glue-wnli)

## A5b. Tune only K or only V (no compression, 200 epochs)

| split | K+V (paper) | only V | only K | ICL |
|---|---|---|---|---|
| 13 | 44.70 | 44.34 | 40.29 | 35.25 |
| 21 | 44.62 | 44.41 | 39.64 | 36.18 |
| 87 | 44.90 | 44.92 | 39.91 | 36.14 |
| mean | 44.74 | 44.56 | 39.95 | 35.86 |

Training peak memory (last split): K+V 14299 MB, only V 14219 MB, only K 14219 MB; time per task 9.5 / 9.1 / 9.1 s.
