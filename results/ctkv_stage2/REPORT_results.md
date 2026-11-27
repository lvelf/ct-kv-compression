# CT-KV Stage 2: per-query sparse reading of demonstration KV — results

All numbers: accuracy in points on the frozen Stage 1 common test IDs, seed-mean first, tasks equal weight. Brackets: 95% paired bootstrap (5,000 draws; seeds × questions; seed-only in parentheses). Cells are `adapt/read`; dense = diagnostic float32 path at r = 1 (same implementation as the sparse reads); prod = production SDPA kernel; S1 = Stage 1 production runs.

## Main comparison (arm D refined demonstrations, top-k 75%)

| evaluation | tasks | dense/dense | top-k/top-k | sparse − dense | confirmation cells only | SD diff | fixed / broken Q |
|---|---|---|---|---|---|---|---|
| BBH Boolean (Llama-3-8B-Instruct, 10-shot, 16 steps) | 1 | 79.88 | 79.97 | +0.10 [-0.53, +0.78] | +0.24 [-0.47, +1.00] | +0.29 [-0.24, +0.80] | 30 / 26 |
| NLP-LR classification (GPT-2 Large, 16-shot, 200 steps) | 12 | 63.39 | 63.29 | -0.10 [-0.50, +0.30] | -0.05 [-0.45, +0.33] | +0.27 [-0.21, +0.64] | 367 / 504 |
| NLP-LR multiple choice (GPT-2 Large, 16-shot, 200 steps) | 9 | 44.94 | 45.01 | +0.08 [-0.18, +0.34] | +0.09 [-0.17, +0.36] | -0.14 [-0.34, +0.15] | 469 / 413 |

Confirmation cells exclude the exploration cells (BBH Boolean, financial_phrasebank, ai2_arc × seeds 1000–1002); see PLAN.md (no independent dev set exists).

## Levels (mean accuracy per cell)

Each level is averaged over the (task, seed) units on which the main cell (D top-k 75%/top-k 75%) exists.

| cell | bbh | clf | mc |
|---|---|---|---|
| A ICL (Stage 1, prod) | 80.78 | 57.17 | 40.54 |
| A CT-KV (Stage 1, prod) | 78.12 | 61.86 | 44.78 |
| D ICL (Stage 1, prod) | 81.00 | 57.93 | 40.60 |
| D CT-KV (Stage 1, prod) | 79.75 | 63.33 | 44.95 |
| D step 0, dense read | 81.05 | 57.93 | 40.57 |
| D step 0, top-k 75% read | 81.42 | 58.19 | 40.59 |
| D step 0, top-k 50% read | 81.40 | 58.37 | 40.59 |
| D step 0, fixed 75% read | 82.05 | 58.76 | 40.42 |
| D step 0, random 75% read | 80.62 | 57.90 | 39.99 |
| D dense/dense | 79.88 | 63.39 | 44.94 |
| D dense/prod | 79.78 | 63.45 | 44.98 |
| D dense/top-k 75% | 79.97 | 63.58 | 44.91 |
| D top-k 75%/dense | 79.78 | 63.14 | 44.98 |
| **D top-k 75%/top-k 75% (main)** | 79.97 | 63.29 | 45.01 |
| D dense/top-k 50% | 79.90 | 63.23 | 44.97 |
| D top-k 50%/dense | 80.05 | 62.57 | 45.05 |
| D top-k 50%/top-k 50% | 80.38 | 63.24 | 44.88 |
| D fixed 75%/fixed 75% | 80.85 | 63.75 | 44.93 |
| D random 75%/random 75% | 79.60 | 62.99 | 44.69 |
| D physical prune 75% (Ryan) | 80.58 | — | — |
| D physical prune 50% (Ryan) | 80.83 | — | — |
| A dense/dense | 78.12 | 61.88 | 44.73 |
| A top-k 75%/top-k 75% | 78.10 | 61.91 | 44.80 |

## Run matrix (75%)

| contrast | bbh | clf | mc |
|---|---|---|---|
| dense adapt, top-k read − dense/dense | +0.10 [-0.35, +0.53] | +0.19 [-0.12, +0.50] | -0.03 [-0.23, +0.17] |
| top-k adapt, dense read − dense/dense | -0.10 [-0.78, +0.55] | -0.25 [-0.78, +0.26] | +0.04 [-0.19, +0.28] |
| top-k adapt, top-k read − top-k adapt, dense read | +0.20 [-0.20, +0.62] | +0.14 [-0.24, +0.52] | +0.04 [-0.17, +0.25] |

## Budget sensitivity (50%)

| contrast | bbh | clf | mc |
|---|---|---|---|
| 50%: sparse/sparse − dense/dense | +0.50 [-0.40, +1.48] | -0.15 [-0.73, +0.42] | -0.06 [-0.38, +0.27] |
| 50%: dense adapt, top-k read − dense/dense | +0.03 [-0.62, +0.72] | -0.16 [-0.70, +0.39] | +0.03 [-0.28, +0.33] |
| 50%: top-k adapt, dense read − dense/dense | +0.17 [-0.60, +1.02] | -0.82 [-1.58, -0.06] | +0.11 [-0.22, +0.43] |

## Controls: random, fixed answer_attention, physical pruning

| contrast | bbh | clf | mc |
|---|---|---|---|
| dynamic top-k − random (75%, sparse/sparse) | +0.38 [-1.08, +1.82] | +0.30 [-0.94, +1.52] | +0.32 [-0.53, +1.20] |
| dynamic top-k − fixed answer_attention (75%, sparse/sparse) | -0.88 [-2.45, +0.70] | -0.47 [-1.25, +0.32] | +0.08 [-0.33, +0.52] |
| fixed answer_attention 75% − dense/dense | +0.97 [-0.53, +2.50] | +0.36 [-0.39, +1.14] | -0.01 [-0.46, +0.42] |
| random 75% − dense/dense | -0.28 [-1.65, +1.23] | -0.40 [-1.57, +0.83] | -0.25 [-1.11, +0.65] |
| physical prune 75% (Ryan) − Stage 1 D CT-KV (prod) | +0.83 [-1.02, +2.78] | — | — |
| physical prune 50% (Ryan) − Stage 1 D CT-KV (prod) | +1.08 [-1.07, +3.30] | — | — |

## Against zero-update and original CT-KV references

| contrast | bbh | clf | mc |
|---|---|---|---|
| main sparse/sparse − A ICL (Stage 1, zero-update, random demos) | -0.80 [-3.43, +1.90] | +6.11 [+3.92, +8.25] | +4.48 [+3.13, +5.81] |
| main sparse/sparse − D ICL (zero-update, refined demos, dense read) | -1.08 [-3.90, +1.80] | +5.36 [+3.44, +7.19] | +4.44 [+3.10, +5.73] |
| main sparse/sparse − A CT-KV (Stage 1 original CT-KV) | +1.85 [-1.42, +5.47] | +1.42 [-0.30, +3.09] | +0.23 [-0.71, +1.18] |
| main sparse/sparse − D CT-KV (Stage 1, prod kernel) | +0.22 [-0.28, +0.75] | -0.05 [-0.36, +0.26] | +0.07 [-0.15, +0.29] |

## Step 0 (no update): read only

| contrast | bbh | clf | mc |
|---|---|---|---|
| step 0: top-k 75% read − dense read (ICL, refined demos) | +0.38 [-0.10, +0.95] | +0.26 [+0.06, +0.52] | +0.02 [-0.15, +0.18] |
| step 0: top-k 50% read − dense read | +0.35 [-0.42, +1.20] | +0.44 [+0.06, +0.87] | +0.02 [-0.20, +0.24] |
| step 0: random 75% read − dense read | -0.42 [-2.00, +1.03] | -0.03 [-0.79, +0.72] | -0.58 [-1.27, +0.10] |
| step 0: fixed 75% read − dense read | +1.00 [-0.72, +2.80] | +0.83 [+0.30, +1.41] | -0.16 [-0.47, +0.15] |

## Implementation checks

| contrast | bbh | clf | mc |
|---|---|---|---|
| implementation: D dense/dense (diagnostic) − Stage 1 D CT-KV (prod) | +0.12 [-0.35, +0.68] | +0.06 [-0.32, +0.44] | -0.01 [-0.24, +0.22] |
| implementation: dense checkpoint, prod read − dense read | -0.10 [-0.50, +0.25] | +0.06 [-0.15, +0.29] | +0.04 [-0.12, +0.19] |
| implementation: step 0, dense read − Stage 1 D ICL (prod) | +0.05 [-0.45, +0.55] | -0.00 [-0.15, +0.15] | -0.03 [-0.16, +0.10] |

## Arm A (random demonstrations)

| contrast | bbh | clf | mc |
|---|---|---|---|
| A arm: sparse/sparse − dense/dense | -0.03 [-0.57, +0.50] | +0.04 [-0.29, +0.36] | +0.07 [-0.19, +0.33] |
| **synergy**: (D sparse − D dense) − (A sparse − A dense) | +0.12 [-0.75, +0.98] | -0.14 [-0.66, +0.37] | +0.01 [-0.32, +0.36] |

## Per task: main contrast (D top-k 75% sparse/sparse − dense/dense)

| family | task | seeds | Q | dense/dense | sparse/sparse | diff | fixed / broken | SD diff |
|---|---|---|---|---|---|---|---|---|
| bbh | boolean_expressions | 20 | 200 | 79.88 | 79.97 | +0.10 [-0.53, +0.78] | 30 / 26 | +0.29 |
| clf | climate_fever | 10 | 307 | 39.71 | 39.84 | +0.13 [-0.88, +1.14] | 42 / 38 | -0.08 |
| clf | ethos-national_origin | 10 | 87 | 77.59 | 77.70 | +0.11 [-0.80, +1.15] | 3 / 2 | +0.10 |
| clf | ethos-race | 10 | 87 | 81.61 | 81.61 | +0.00 [+0.00, +0.00] | 0 / 0 | +0.00 |
| clf | ethos-religion | 10 | 87 | 76.32 | 76.32 | +0.00 [-0.69, +0.69] | 2 / 2 | +0.44 |
| clf | financial_phrasebank | 10 | 970 | 68.93 | 67.94 | -0.99 [-2.24, +0.01] | 138 / 234 | +1.36 |
| clf | glue-wnli | 10 | 51 | 53.53 | 52.94 | -0.59 [-3.14, +1.76] | 6 / 9 | +0.54 |
| clf | hate_speech18 | 10 | 2141 | 88.54 | 88.51 | -0.03 [-0.15, +0.05] | 9 / 15 | +0.11 |
| clf | poem_sentiment | 10 | 105 | 59.90 | 60.10 | +0.19 [-1.05, +1.43] | 9 / 7 | -0.26 |
| clf | sick | 10 | 495 | 62.83 | 62.65 | -0.18 [-1.01, +0.61] | 55 / 64 | +0.29 |
| clf | tweet_eval-hate | 10 | 999 | 57.66 | 57.34 | -0.32 [-0.77, +0.12] | 82 / 114 | +0.05 |
| clf | tweet_eval-stance_atheism | 10 | 52 | 51.54 | 52.12 | +0.58 [-1.93, +3.27] | 12 / 9 | -0.47 |
| clf | tweet_eval-stance_feminist | 10 | 67 | 42.54 | 42.39 | -0.15 [-2.54, +1.94] | 9 / 10 | +1.22 |
| mc | ai2_arc | 10 | 299 | 27.63 | 27.42 | -0.20 [-1.07, +0.57] | 26 / 32 | -0.05 |
| mc | codah | 10 | 556 | 45.09 | 45.20 | +0.11 [-0.63, +0.85] | 59 / 53 | -0.03 |
| mc | commonsense_qa | 10 | 1221 | 44.70 | 44.83 | +0.13 [-0.27, +0.55] | 129 / 113 | +0.03 |
| mc | openbookqa | 10 | 500 | 31.12 | 31.28 | +0.16 [-0.40, +0.74] | 45 / 37 | -0.17 |
| mc | qasc | 10 | 926 | 27.76 | 28.01 | +0.25 [-0.25, +0.81] | 120 / 97 | -0.11 |
| mc | quarel | 10 | 278 | 54.86 | 55.11 | +0.25 [-0.36, +0.90] | 18 / 11 | +0.23 |
| mc | quartz-no_knowledge | 10 | 384 | 54.69 | 54.56 | -0.13 [-0.78, +0.52] | 23 / 28 | +0.10 |
| mc | quartz-with_knowledge | 10 | 384 | 53.20 | 53.41 | +0.21 [-0.47, +0.89] | 41 / 33 | +0.04 |
| mc | superglue-copa | 10 | 100 | 65.40 | 65.30 | -0.10 [-1.50, +1.30] | 8 / 9 | -1.34 |

## Random-mask seed spread (exploration cells, random 75% sparse/sparse)

| family | task | seed | mask 0 | mask 1 | mask 2 | range |
|---|---|---|---|---|---|---|
| bbh | boolean_expressions | 1000 | 71.50 | 71.50 | 69.00 | 2.50 |
| bbh | boolean_expressions | 1001 | 77.00 | 78.50 | 74.00 | 4.50 |
| bbh | boolean_expressions | 1002 | 77.00 | 76.00 | 78.00 | 2.00 |
| clf | financial_phrasebank | 1000 | 67.11 | 64.95 | 64.12 | 2.99 |
| clf | financial_phrasebank | 1001 | 69.48 | 69.28 | 69.28 | 0.21 |
| clf | financial_phrasebank | 1002 | 64.85 | 65.26 | 65.26 | 0.41 |
| mc | ai2_arc | 1000 | 25.75 | 28.43 | 24.41 | 4.01 |
| mc | ai2_arc | 1001 | 28.76 | 27.76 | 28.09 | 1.00 |
| mc | ai2_arc | 1002 | 28.09 | 26.76 | 25.08 | 3.01 |

## BBH label-shift diagnostics (mean over seeds; arm D unless noted)

| adapt / read | step | acc | pred True | recall T | recall F | kept share True | legal share True | kept frac True demos | kept frac False demos |
|---|---|---|---|---|---|---|---|---|---|
| A icl/dense | 0 | 80.67 | 0.633 | 0.915 | 0.685 | 0.590 | 0.590 | 1.000 | 1.000 |
| A dense/dense | 16 | 78.12 | 0.642 | 0.900 | 0.648 | 0.590 | 0.590 | 1.000 | 1.000 |
| A icl/prod | 0 | 80.78 | 0.634 | 0.917 | 0.685 | nan | nan | nan | nan |
| A dense/prod | 16 | 78.30 | 0.640 | 0.900 | 0.652 | nan | nan | nan | nan |
| A icl/topk75 | 0 | 80.90 | 0.633 | 0.917 | 0.687 | 0.589 | 0.590 | 0.748 | 0.749 |
| A dense/topk75 | 16 | 78.38 | 0.640 | 0.900 | 0.653 | 0.589 | 0.590 | 0.748 | 0.749 |
| A topk75/dense | 16 | 78.05 | 0.646 | 0.902 | 0.644 | 0.590 | 0.590 | 1.000 | 1.000 |
| A topk75/topk75 | 16 | 78.10 | 0.645 | 0.902 | 0.645 | 0.589 | 0.590 | 0.748 | 0.749 |
| D icl/dense | 0 | 81.05 | 0.620 | 0.906 | 0.703 | 0.600 | 0.600 | 1.000 | 1.000 |
| D dense/dense | 16 | 79.88 | 0.602 | 0.878 | 0.710 | 0.600 | 0.600 | 1.000 | 1.000 |
| D icl/fixed75 | 0 | 82.05 | 0.593 | 0.890 | 0.743 | 0.627 | 0.600 | 0.784 | 0.699 |
| D dense/fixed75 | 16 | 80.12 | 0.586 | 0.866 | 0.729 | 0.627 | 0.600 | 0.784 | 0.699 |
| D icl/prod | 0 | 81.00 | 0.621 | 0.906 | 0.702 | nan | nan | nan | nan |
| D dense/prod | 16 | 79.78 | 0.602 | 0.877 | 0.708 | nan | nan | nan | nan |
| D icl/rand75 | 0 | 80.62 | 0.636 | 0.917 | 0.681 | 0.600 | 0.600 | 0.750 | 0.750 |
| D dense/rand75 | 16 | 79.22 | 0.618 | 0.887 | 0.685 | 0.600 | 0.600 | 0.750 | 0.750 |
| D icl/topk50 | 0 | 81.40 | 0.621 | 0.911 | 0.705 | 0.617 | 0.600 | 0.514 | 0.479 |
| D dense/topk50 | 16 | 79.90 | 0.605 | 0.882 | 0.706 | 0.617 | 0.600 | 0.514 | 0.479 |
| D icl/topk75 | 0 | 81.42 | 0.619 | 0.909 | 0.707 | 0.607 | 0.600 | 0.759 | 0.737 |
| D dense/topk75 | 16 | 79.97 | 0.601 | 0.878 | 0.711 | 0.607 | 0.600 | 0.759 | 0.737 |
| D fixed75/dense | 16 | 80.28 | 0.617 | 0.896 | 0.697 | 0.600 | 0.600 | 1.000 | 1.000 |
| D fixed75/fixed75 | 16 | 80.85 | 0.597 | 0.883 | 0.724 | 0.627 | 0.600 | 0.784 | 0.699 |
| D prune50/prod | 16 | 80.83 | 0.582 | 0.868 | 0.741 | nan | nan | nan | nan |
| D prune75/prod | 16 | 80.58 | 0.610 | 0.892 | 0.708 | nan | nan | nan | nan |
| D rand75/dense | 16 | 80.12 | 0.597 | 0.875 | 0.718 | 0.600 | 0.600 | 1.000 | 1.000 |
| D rand75/rand75 | 16 | 79.60 | 0.610 | 0.883 | 0.698 | 0.600 | 0.600 | 0.750 | 0.750 |
| D topk50/dense | 16 | 80.05 | 0.603 | 0.881 | 0.710 | 0.600 | 0.600 | 1.000 | 1.000 |
| D topk50/topk50 | 16 | 80.38 | 0.605 | 0.886 | 0.711 | 0.617 | 0.600 | 0.514 | 0.479 |
| D topk75/dense | 16 | 79.78 | 0.604 | 0.879 | 0.706 | 0.600 | 0.600 | 1.000 | 1.000 |
| D topk75/topk75 | 16 | 79.97 | 0.605 | 0.882 | 0.707 | 0.607 | 0.600 | 0.759 | 0.737 |

Across 20 seeds (D): demo True-share − 0.5 has range +0.10…+0.10; Δ(pred True) sparse/sparse − dense/dense mean +0.003; Δacc mean +0.10; kept − legal True-token share (top-k read) mean +0.0071. corr(bias, Δpred True) = +0.00, corr(bias, Δacc) = +0.00.
Kept share is the share of selected demo positions (summed over layers, heads and query tokens) that belong to True-labelled demonstrations; legal share is the same over all legal candidates, so a longer True demo does not count as routing preference. Demo-equal columns average the kept fraction per demonstration.

## Cost (medians over jobs)

Timings below come from the accuracy runs, in which the BBH and NLP-LR queues shared the GPU; the serial benchmark (`bench/`) is the reference for speed.

| family | adapt | jobs | adaptation s | forward s | backward s | select init s | step s | peak MiB | resident KV MiB | train access |
|---|---|---|---|---|---|---|---|---|---|---|
| bbh | A dense | 20 | 6.45 | 4.48 | 1.80 | nan | 0.396 | 16018 | 22.5 | 1.000 |
| bbh | A topk75 | 20 | 3.27 | 1.67 | 1.50 | nan | 0.202 | 16028 | 22.5 | 0.747 |
| bbh | D dense | 20 | 6.34 | 4.20 | 1.78 | 0.18 | 0.392 | 16018 | 22.5 | 1.000 |
| bbh | D fixed75 | 20 | 3.41 | 1.66 | 1.50 | 0.15 | 0.202 | 16028 | 22.5 | 0.747 |
| bbh | D prune50 | 20 | 2.75 | 1.17 | 1.34 | 0.14 | 0.161 | 15657 | 12.5 | nan |
| bbh | D prune75 | 20 | 2.82 | 1.20 | 1.37 | 0.15 | 0.165 | 15715 | 17.5 | nan |
| bbh | D rand75 | 20 | 3.27 | 1.67 | 1.50 | nan | 0.202 | 16028 | 22.5 | 0.747 |
| bbh | D topk50 | 20 | 3.27 | 1.67 | 1.50 | nan | 0.202 | 16028 | 22.5 | 0.498 |
| bbh | D topk75 | 20 | 3.29 | 1.68 | 1.50 | nan | 0.203 | 16028 | 22.5 | 0.747 |
| clf | A dense | 120 | 12.69 | 5.86 | 6.24 | nan | 0.063 | 8942 | 80.8 | 1.000 |
| clf | A topk75 | 120 | 16.46 | 9.16 | 6.65 | nan | 0.082 | 9145 | 80.8 | 0.749 |
| clf | D dense | 120 | 12.90 | 5.94 | 6.37 | 0.03 | 0.064 | 8979 | 80.4 | 1.000 |
| clf | D fixed75 | 120 | 17.47 | 9.70 | 7.12 | 0.03 | 0.087 | 9186 | 80.4 | 0.749 |
| clf | D rand75 | 120 | 17.61 | 9.86 | 7.15 | nan | 0.088 | 9172 | 80.4 | 0.749 |
| clf | D topk50 | 120 | 17.47 | 9.74 | 7.12 | nan | 0.087 | 9171 | 80.4 | 0.499 |
| clf | D topk75 | 120 | 17.48 | 9.75 | 7.13 | nan | 0.087 | 9169 | 80.4 | 0.749 |
| mc | A dense | 90 | 10.24 | 4.87 | 4.87 | nan | 0.051 | 7306 | 61.3 | 1.000 |
| mc | A topk75 | 90 | 13.18 | 7.41 | 5.24 | nan | 0.066 | 7404 | 61.3 | 0.749 |
| mc | D dense | 90 | 11.79 | 5.49 | 5.69 | 0.03 | 0.059 | 7963 | 60.7 | 1.000 |
| mc | D fixed75 | 90 | 14.82 | 8.07 | 5.99 | 0.03 | 0.073 | 8167 | 60.7 | 0.749 |
| mc | D rand75 | 90 | 14.87 | 8.15 | 5.98 | nan | 0.073 | 8151 | 60.7 | 0.749 |
| mc | D topk50 | 90 | 14.79 | 8.08 | 5.99 | nan | 0.073 | 8158 | 60.7 | 0.499 |
| mc | D topk75 | 90 | 14.77 | 8.06 | 5.97 | nan | 0.073 | 8158 | 60.7 | 0.749 |

## Cost: serial benchmark (GPU alone; arm D exploration cells; medians over jobs)

Each job runs one adaptation and only its own matching read at the endpoint. Ratios are against the production SDPA job (prod) of the same family. The sparse paths are diagnostic exact top-k implementations that materialise the full score matrix, so they cannot be faster than dense; the access ratio is the fraction of legal demonstration KV positions read, i.e. the reduction a fused kernel could at most exploit.

| family | adapt | jobs | adaptation s | × prod | step s | forward s | backward s | select init s | peak MiB | resident KV MiB | eval ms/item | × prod | access (train / read) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bbh | prod | 3 | 2.75 | 1.00 | 0.169 | 1.21 | 1.43 | nan | 15815 | 22.5 | 14.69 | 1.00 | nan / nan |
| bbh | dense | 3 | 3.04 | 1.11 | 0.188 | 1.47 | 1.47 | nan | 16018 | 22.5 | 19.54 | 1.33 | 1.000 / 1.000 |
| bbh | topk75 | 3 | 3.27 | 1.19 | 0.202 | 1.67 | 1.50 | nan | 16028 | 22.5 | 21.65 | 1.47 | 0.747 / 0.750 |
| bbh | topk50 | 3 | 3.27 | 1.19 | 0.202 | 1.67 | 1.50 | nan | 16028 | 22.5 | 21.69 | 1.48 | 0.498 / 0.500 |
| bbh | fixed75 | 3 | 3.42 | 1.24 | 0.202 | 1.67 | 1.50 | 0.15 | 16028 | 22.5 | 21.64 | 1.47 | 0.747 / 0.750 |
| bbh | rand75 | 3 | 3.27 | 1.19 | 0.202 | 1.67 | 1.50 | nan | 16028 | 22.5 | 21.74 | 1.48 | 0.747 / 0.750 |
| bbh | prune75 | 3 | 2.82 | 1.02 | 0.165 | 1.20 | 1.37 | 0.15 | 15715 | 17.5 | 14.59 | 0.99 | nan / nan |
| bbh | prune50 | 3 | 2.76 | 1.00 | 0.161 | 1.17 | 1.34 | 0.15 | 15657 | 12.5 | 14.36 | 0.98 | nan / nan |
| clf | prod | 3 | 9.03 | 1.00 | 0.045 | 3.77 | 4.70 | nan | 6148 | 79.1 | 1.28 | 1.00 | nan / nan |
| clf | dense | 3 | 12.64 | 1.40 | 0.063 | 5.98 | 6.18 | nan | 8811 | 79.1 | 1.99 | 1.55 | 1.000 / 1.000 |
| clf | topk75 | 3 | 16.12 | 1.79 | 0.080 | 9.12 | 6.44 | nan | 9050 | 79.1 | 3.02 | 2.36 | 0.749 / 0.749 |
| clf | topk50 | 3 | 16.14 | 1.79 | 0.080 | 9.13 | 6.45 | nan | 9050 | 79.1 | 3.02 | 2.36 | 0.499 / 0.500 |
| clf | fixed75 | 3 | 16.15 | 1.79 | 0.080 | 9.10 | 6.44 | 0.03 | 9055 | 79.1 | 3.01 | 2.35 | 0.749 / 0.749 |
| clf | rand75 | 3 | 16.22 | 1.80 | 0.081 | 9.20 | 6.48 | nan | 9048 | 79.1 | 3.05 | 2.38 | 0.749 / 0.749 |
| mc | prod | 3 | 10.61 | 1.00 | 0.052 | 3.91 | 6.12 | nan | 6962 | 91.2 | 1.30 | 1.00 | nan / nan |
| mc | dense | 3 | 15.95 | 1.50 | 0.079 | 6.89 | 8.48 | nan | 10531 | 91.2 | 2.24 | 1.72 | 1.000 / 1.000 |
| mc | topk75 | 3 | 22.26 | 2.10 | 0.111 | 11.93 | 9.73 | nan | 10841 | 91.2 | 3.78 | 2.91 | 0.749 / 0.750 |
| mc | topk50 | 3 | 22.24 | 2.10 | 0.111 | 11.93 | 9.71 | nan | 10842 | 91.2 | 3.78 | 2.91 | 0.499 / 0.500 |
| mc | fixed75 | 3 | 22.21 | 2.09 | 0.111 | 11.83 | 9.74 | 0.03 | 10828 | 91.2 | 3.74 | 2.88 | 0.749 / 0.750 |
| mc | rand75 | 3 | 22.48 | 2.12 | 0.112 | 12.10 | 9.79 | nan | 10840 | 91.2 | 3.83 | 2.95 | 0.749 / 0.750 |

