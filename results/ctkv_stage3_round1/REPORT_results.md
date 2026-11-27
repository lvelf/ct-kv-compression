# CT-KV Stage 3 round 1: half-delta KV update and same-trajectory early stopping — results

Trajectories: 230 / 230 expected. Runner technical checks failing: none.

Accuracy in points on the per-task common questions (present in all five conditions of every seed); seed mean per task, tasks equal weight. Brackets: 95% paired bootstrap (5,000 draws over seeds × questions, shared across conditions); seed-only interval in parentheses.

## Five-condition accuracy

| family | tasks | icl | early_quarter | early_half | full | half_delta |
|---|---|---|---|---|---|---|
| BBH Boolean (Llama-3-8B-Instruct, 10-shot, T = 16) | 1 | 81.00 [76.45, 85.47] | 80.15 [75.82, 84.25] | 79.75 [75.38, 83.92] | 79.70 [75.38, 83.85] | 81.25 [76.72, 85.55] |
| NLP-LR classification (GPT-2 Large, 16-shot, T = 200) | 12 | 57.93 [55.56, 60.27] | 63.56 [61.56, 65.57] | 63.44 [61.44, 65.41] | 63.39 [61.38, 65.32] | 61.48 [59.17, 63.85] |
| NLP-LR multiple choice (GPT-2 Large, 16-shot, T = 200) | 9 | 40.60 [39.04, 42.21] | 45.43 [43.90, 46.90] | 45.06 [43.60, 46.47] | 44.97 [43.54, 46.39] | 44.03 [42.51, 45.53] |
| clf runner score (macro-F1) | 12 | 32.42 | 42.49 | 43.95 | 44.32 | 38.62 |
| mc runner score (accuracy) | 9 | 40.60 | 45.43 | 45.06 | 44.97 | 44.03 |

Gradient steps: BBH icl 0 / early_quarter 4 / early_half 8 / full 16 / half_delta 16; NLP-LR 0 / 50 / 100 / 200 / 200.

## Contrasts

| contrast | family | a | b | diff | 95% CI | seed-only CI | fixed / broken Q | seeds better / worse / tie |
|---|---|---|---|---|---|---|---|---|
| main: half_delta − full | bbh | 81.25 | 79.70 | +1.55 | [-0.05, +3.12] | (+0.62, +2.50) | 156 / 94 | 14 / 3 / 3 |
| main: half_delta − full | clf | 61.48 | 63.39 | -1.91 | [-3.44, -0.30] | (-2.86, -0.95) | 3020 / 5094 | 46 / 47 / 27 |
| main: half_delta − full | mc | 44.03 | 44.97 | -0.94 | [-1.91, +0.05] | (-1.38, -0.52) | 3257 / 3878 | 34 / 54 / 2 |
| half_delta − icl | bbh | 81.25 | 81.00 | +0.25 | [-1.45, +1.95] | (-0.92, +1.40) | 139 / 129 | 12 / 8 / 0 |
| half_delta − icl | clf | 61.48 | 57.93 | +3.55 | [+2.38, +4.78] | (+2.65, +4.49) | 5305 / 1892 | 52 / 24 / 44 |
| half_delta − icl | mc | 44.03 | 40.60 | +3.43 | [+2.64, +4.20] | (+3.13, +3.73) | 3882 / 1523 | 72 / 13 / 5 |
| full − icl | bbh | 79.70 | 81.00 | -1.30 | [-3.97, +1.42] | (-3.23, +0.60) | 227 / 279 | 8 / 12 / 0 |
| full − icl | clf | 63.39 | 57.93 | +5.46 | [+3.60, +7.24] | (+4.63, +6.30) | 10071 / 4584 | 54 / 44 / 22 |
| full − icl | mc | 44.97 | 40.60 | +4.38 | [+3.00, +5.65] | (+3.89, +4.89) | 6660 / 3680 | 75 / 15 / 0 |
| early_half − full | bbh | 79.75 | 79.70 | +0.05 | [-0.43, +0.58] | (-0.22, +0.35) | 19 / 17 | 5 / 7 / 8 |
| early_half − full | clf | 63.44 | 63.39 | +0.05 | [-0.38, +0.50] | (-0.21, +0.32) | 546 / 451 | 44 / 35 / 41 |
| early_half − full | mc | 45.06 | 44.97 | +0.09 | [-0.27, +0.43] | (-0.12, +0.27) | 411 / 390 | 42 / 31 / 17 |
| early_quarter − full | bbh | 80.15 | 79.70 | +0.45 | [-0.65, +1.48] | (-0.05, +0.88) | 93 / 75 | 12 / 4 / 4 |
| early_quarter − full | clf | 63.56 | 63.39 | +0.17 | [-0.89, +1.21] | (-0.39, +0.73) | 2195 / 1953 | 48 / 40 / 32 |
| early_quarter − full | mc | 45.43 | 44.97 | +0.45 | [-0.41, +1.34] | (+0.12, +0.79) | 2858 / 2534 | 53 / 29 / 8 |
| early_half − half_delta | bbh | 79.75 | 81.25 | -1.50 | [-2.97, -0.08] | (-2.42, -0.67) | 80 / 140 | 2 / 17 / 1 |
| early_half − half_delta | clf | 63.44 | 61.48 | +1.96 | [+0.32, +3.53] | (+0.97, +2.92) | 5043 / 2874 | 48 / 47 / 25 |
| early_half − half_delta | mc | 45.06 | 44.03 | +1.03 | [+0.07, +1.95] | (+0.63, +1.43) | 3807 / 3165 | 57 / 29 / 4 |
| early_quarter − icl | bbh | 80.15 | 81.00 | -0.85 | [-3.17, +1.50] | (-2.60, +0.85) | 190 / 224 | 8 / 11 / 1 |
| early_quarter − icl | clf | 63.56 | 57.93 | +5.63 | [+3.53, +7.81] | (+4.87, +6.40) | 9567 / 3838 | 55 / 35 / 30 |
| early_quarter − icl | mc | 45.43 | 40.60 | +4.83 | [+3.68, +5.96] | (+4.38, +5.26) | 5902 / 2598 | 75 / 14 / 1 |
| early_half − icl | bbh | 79.75 | 81.00 | -1.25 | [-3.83, +1.35] | (-3.12, +0.53) | 214 / 264 | 8 / 11 / 1 |
| early_half − icl | clf | 63.44 | 57.93 | +5.51 | [+3.64, +7.35] | (+4.65, +6.38) | 9995 / 4413 | 52 / 43 / 25 |
| early_half − icl | mc | 45.06 | 40.60 | +4.46 | [+3.13, +5.72] | (+3.96, +4.99) | 6614 / 3613 | 76 / 13 / 1 |

## Pre-registered decisions

- BBH half_delta − full = +1.55 [-0.05, +3.12]: not shown to improve (lower bound ≤ 0).
- clf half_delta − full = -1.91 [-3.44, -0.30]: uncertain (interval crosses −1.0).
- mc half_delta − full = -0.94 [-1.91, +0.05]: uncertain (interval crosses −1.0).

## Per task: main contrast (half_delta − full) and full − icl

| family | task | seeds | Q | icl | full | half_delta | half_delta − full | full − icl | fixed / broken |
|---|---|---|---|---|---|---|---|---|---|
| bbh | boolean_expressions | 20 | 200 | 81.00 | 79.70 | 81.25 | +1.55 [-0.05, +3.12] | -1.30 | 156 / 94 |
| clf | climate_fever | 10 | 307 | 37.10 | 39.87 | 39.80 | -0.07 [-3.32, +3.68] | +2.77 | 277 / 279 |
| clf | ethos-national_origin | 10 | 87 | 79.31 | 77.70 | 79.20 | +1.49 [-1.15, +4.37] | -1.61 | 25 / 12 |
| clf | ethos-race | 10 | 87 | 81.61 | 81.61 | 81.61 | +0.00 [+0.00, +0.00] | +0.00 | 0 / 0 |
| clf | ethos-religion | 10 | 87 | 77.70 | 76.55 | 77.59 | +1.03 [-0.80, +3.10] | -1.15 | 14 / 5 |
| clf | financial_phrasebank | 10 | 970 | 29.02 | 68.54 | 53.67 | -14.87 [-22.54, -7.35] | +39.52 | 883 / 2325 |
| clf | glue-wnli | 10 | 51 | 47.06 | 53.73 | 46.67 | -7.06 [-17.65, +3.33] | +6.67 | 53 / 89 |
| clf | hate_speech18 | 10 | 2141 | 88.74 | 88.51 | 88.74 | +0.23 [+0.03, +0.57] | -0.23 | 66 / 17 |
| clf | poem_sentiment | 10 | 105 | 65.24 | 59.90 | 63.05 | +3.14 [-0.10, +7.14] | -5.33 | 70 / 37 |
| clf | sick | 10 | 495 | 28.89 | 62.57 | 46.38 | -16.18 [-22.59, -9.82] | +33.68 | 626 / 1427 |
| clf | tweet_eval-hate | 10 | 999 | 57.12 | 57.66 | 58.17 | +0.51 [-1.23, +2.30] | +0.54 | 851 / 800 |
| clf | tweet_eval-stance_atheism | 10 | 52 | 55.19 | 51.35 | 56.35 | +5.00 [-1.92, +12.31] | -3.85 | 64 / 38 |
| clf | tweet_eval-stance_feminist | 10 | 67 | 48.21 | 42.69 | 46.57 | +3.88 [-3.73, +12.99] | -5.52 | 91 / 65 |
| mc | ai2_arc | 10 | 299 | 25.42 | 27.49 | 26.25 | -1.24 [-4.08, +1.61] | +2.07 | 185 / 222 |
| mc | codah | 10 | 556 | 41.29 | 45.11 | 45.11 | +0.00 [-2.16, +2.21] | +3.81 | 450 / 450 |
| mc | commonsense_qa | 10 | 1221 | 34.28 | 44.49 | 43.03 | -1.46 [-3.03, +0.15] | +10.20 | 826 / 1004 |
| mc | openbookqa | 10 | 500 | 26.64 | 31.12 | 29.80 | -1.32 [-3.80, +1.04] | +4.48 | 301 / 367 |
| mc | qasc | 10 | 926 | 16.34 | 28.08 | 24.68 | -3.40 [-5.35, -1.57] | +11.74 | 594 / 909 |
| mc | quarel | 10 | 278 | 57.41 | 55.29 | 56.73 | +1.44 [-0.90, +3.78] | -2.12 | 180 / 140 |
| mc | quartz-no_knowledge | 10 | 384 | 52.60 | 54.53 | 54.45 | -0.08 [-2.45, +2.58] | +1.93 | 296 / 299 |
| mc | quartz-with_knowledge | 10 | 384 | 51.09 | 53.26 | 51.93 | -1.33 [-4.35, +1.59] | +2.16 | 341 / 392 |
| mc | superglue-copa | 10 | 100 | 60.30 | 65.40 | 64.30 | -1.10 [-7.10, +5.20] | +5.10 | 84 / 95 |

## BBH label diagnostics (mean over seeds)

| condition | acc | pred True | recall True | recall False | invalid |
|---|---|---|---|---|---|
| icl | 81.00 | 0.621 | 0.906 | 0.702 | 0.000 |
| early_quarter | 80.15 | 0.613 | 0.891 | 0.701 | 0.000 |
| early_half | 79.75 | 0.608 | 0.882 | 0.702 | 0.000 |
| full | 79.70 | 0.607 | 0.882 | 0.702 | 0.000 |
| half_delta | 81.25 | 0.615 | 0.904 | 0.710 | 0.000 |

Gold True share: 0.530.

## Cost (medians over trajectories; queue ran serially, GPU not shared)

| family | jobs | prefill s | adapt s → early_quarter | → early_half | → full | interp s | eval s (5 conds) | peak MiB | early_quarter share | early_half share |
|---|---|---|---|---|---|---|---|---|---|---|
| bbh | 20 | 0.028 | 0.68 | 1.34 | 2.64 | 0.0010 | 14.56 | 15882 | 0.259 | 0.505 |
| clf | 120 | 0.010 | 2.34 | 4.67 | 9.30 | 0.0011 | 2.00 | 6465 | 0.253 | 0.502 |
| mc | 90 | 0.009 | 2.21 | 4.41 | 8.79 | 0.0011 | 8.06 | 6044 | 0.252 | 0.502 |

## Displacement geometry (medians; C_t − C_0 against C_T − C_0)

| family | condition | tensor | norm ratio | cosine | ‖C_T − C_0‖ / ‖C_0‖ |
|---|---|---|---|---|---|
| bbh | early_quarter | K | 0.785 | 0.954 | 0.0006 |
| bbh | early_quarter | V | 0.736 | 0.949 | 0.0103 |
| bbh | early_half | K | 0.963 | 0.996 | 0.0006 |
| bbh | early_half | V | 0.952 | 0.995 | 0.0103 |
| bbh | half_delta | K | 0.525 | 0.922 | 0.0006 |
| bbh | half_delta | V | 0.518 | 0.962 | 0.0103 |
| clf | early_quarter | K | 0.752 | 0.945 | 0.0051 |
| clf | early_quarter | V | 0.752 | 0.946 | 0.0242 |
| clf | early_half | K | 0.978 | 0.998 | 0.0051 |
| clf | early_half | V | 0.977 | 0.998 | 0.0242 |
| clf | half_delta | K | 0.500 | 0.999 | 0.0051 |
| clf | half_delta | V | 0.500 | 0.999 | 0.0242 |
| mc | early_quarter | K | 0.741 | 0.933 | 0.0075 |
| mc | early_quarter | V | 0.739 | 0.937 | 0.0372 |
| mc | early_half | K | 0.980 | 0.998 | 0.0075 |
| mc | early_half | V | 0.979 | 0.998 | 0.0372 |
| mc | half_delta | K | 0.500 | 0.999 | 0.0075 |
| mc | half_delta | V | 0.500 | 1.000 | 0.0372 |

## Coverage

| family | task | seeds | Q used / target |
|---|---|---|---|
| bbh | boolean_expressions | 20/20 | 200 / 200 |
| clf | climate_fever | 10/10 | 307 / 307 |
| clf | ethos-national_origin | 10/10 | 87 / 87 |
| clf | ethos-race | 10/10 | 87 / 87 |
| clf | ethos-religion | 10/10 | 87 / 87 |
| clf | financial_phrasebank | 10/10 | 970 / 970 |
| clf | glue-wnli | 10/10 | 51 / 51 |
| clf | hate_speech18 | 10/10 | 2141 / 2141 |
| clf | poem_sentiment | 10/10 | 105 / 105 |
| clf | sick | 10/10 | 495 / 495 |
| clf | tweet_eval-hate | 10/10 | 999 / 999 |
| clf | tweet_eval-stance_atheism | 10/10 | 52 / 52 |
| clf | tweet_eval-stance_feminist | 10/10 | 67 / 67 |
| mc | ai2_arc | 10/10 | 299 / 299 |
| mc | codah | 10/10 | 556 / 556 |
| mc | commonsense_qa | 10/10 | 1221 / 1221 |
| mc | openbookqa | 10/10 | 500 / 500 |
| mc | qasc | 10/10 | 926 / 926 |
| mc | quarel | 10/10 | 278 / 278 |
| mc | quartz-no_knowledge | 10/10 | 384 / 384 |
| mc | quartz-with_knowledge | 10/10 | 384 / 384 |
| mc | superglue-copa | 10/10 | 100 / 100 |
