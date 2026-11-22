# Stress test B3: compressing the CT-KV initialization (NLP-LR, GPT-2 Large)

Each demonstration keeps ceil(ratio × its length) tokens; the same tokens in every layer and head. `e0` = compress, then ICL (no training). `e200` = compress, then CT-KV for 200 epochs.

Scores are averaged over the splits listed in the last column (e0/e200). "Gain kept" is (e200 − ICL) / (uncompressed CT-KV − ICL), with both baselines averaged over the same splits as that row's e200 runs.

Memory: `KV MB` is the size of the demonstration cache (bf16, all layers). `train peak` and `eval peak` are torch peak allocated memory per task, including the 1.5 GB model. Times are per task, averaged over tasks.

| method | ratio | kv len | KV MB | e0 | e200 | gain kept | sink kept | train s | train peak MB | eval s | eval peak MB | splits e0/e200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 1.0 | 452 | 79.4 | 35.86 | 44.36 | 100% | 100% | 10.0 | 14551 | 17.7 | 10139 | 3/5 |
| attn | 0.75 | 345 | 60.7 | 35.34 | 44.16 | 93% | 100% | 9.5 | 12614 | 16.7 | 8580 | 3/3 |
| attn | 0.5 | 230 | 40.4 | 35.01 | 42.55 | 75% | 100% | 8.8 | 10370 | 16.5 | 6896 | 3/3 |
| attn | 0.25 | 119 | 20.9 | 32.37 | 39.87 | 45% | 100% | 8.3 | 8197 | 16.3 | 5264 | 3/3 |
| attn | 0.125 | 63 | 11.2 | 32.50 | 39.13 | 37% | 100% | 8.0 | 7115 | 15.6 | 4455 | 3/3 |
| first | 0.25 | 120 | 21.1 | 30.93 | 40.14 | 49% | 100% | 8.1 | 8171 | 16.0 | 5289 | 2/2 |
| l2 | 0.75 | 345 | 60.7 | 35.93 | 43.81 | 89% | 100% | 9.5 | 12614 | 16.8 | 8580 | 3/3 |
| l2 | 0.5 | 230 | 40.4 | 35.40 | 43.01 | 80% | 100% | 8.7 | 10370 | 16.2 | 6896 | 3/3 |
| l2 | 0.25 | 119 | 20.9 | 31.18 | 40.22 | 49% | 100% | 8.3 | 8197 | 16.1 | 5264 | 3/3 |
| l2 | 0.125 | 63 | 11.2 | 31.29 | 38.35 | 28% | 100% | 8.0 | 7115 | 15.7 | 4455 | 3/3 |
| l2_high | 0.25 | 120 | 21.1 | 34.57 | 35.51 | -2% | 0% | 8.1 | 8171 | 15.7 | 5289 | 2/2 |
| last | 0.25 | 120 | 21.1 | 34.81 | 36.33 | 7% | 0% | 8.1 | 8171 | 15.9 | 5289 | 2/2 |
| pool | 0.5 | 232 | 40.9 | 33.68 | 39.77 | 45% | – | 8.6 | 10371 | 16.2 | 6940 | 2/2 |
| pool | 0.25 | 120 | 21.1 | 34.03 | 36.29 | 7% | – | 8.2 | 8171 | 16.1 | 5289 | 2/2 |
| pool | 0.125 | 64 | 11.3 | 33.91 | 35.63 | -1% | – | 7.9 | 7080 | 15.6 | 4472 | 2/2 |
| random | 0.75 | 345 | 60.7 | 36.44 | 43.63 | 87% | 100% | 9.5 | 12614 | 16.8 | 8580 | 3/3 |
| random | 0.5 | 230 | 40.4 | 36.27 | 43.16 | 82% | 100% | 8.7 | 10370 | 16.3 | 6896 | 3/3 |
| random | 0.25 | 119 | 20.9 | 32.63 | 36.75 | 10% | 0% | 8.3 | 8197 | 16.2 | 5264 | 3/3 |
| random | 0.125 | 63 | 11.2 | 33.38 | 34.64 | -14% | 0% | 8.1 | 7115 | 15.8 | 4455 | 3/3 |
| random_segstart | 0.25 | 118 | 20.8 | 32.93 | 40.83 | 54% | 100% | 7.6 | 7783 | 16.2 | 5266 | 1/1 |
| random_sink | 0.75 | 345 | 60.7 | 36.44 | 43.67 | 88% | 100% | 9.5 | 12614 | 16.9 | 8580 | 3/3 |
| random_sink | 0.5 | 230 | 40.4 | 36.27 | 43.11 | 81% | 100% | 8.7 | 10370 | 16.3 | 6896 | 3/3 |
| random_sink | 0.25 | 119 | 20.9 | 32.33 | 40.51 | 52% | 100% | 8.3 | 8197 | 15.9 | 5264 | 3/3 |
| random_sink | 0.125 | 63 | 11.2 | 33.21 | 38.47 | 29% | 100% | 8.1 | 7115 | 15.6 | 4455 | 3/3 |

## Budget grid, epochs 200 (splits [13, 21, 87])

| keep ratio | random | random_sink | attn | l2 |
|---|---|---|---|---|
| 1.0 (uncompressed) | 44.76 | 44.76 | 44.76 | 44.76 |
| 0.75 | 43.63 | 43.67 | 44.16 | 43.81 |
| 0.5 | 43.16 | 43.11 | 42.55 | 43.01 |
| 0.25 | 36.75 | 40.51 | 39.87 | 40.22 |
| 0.125 | 34.64 | 38.47 | 39.13 | 38.35 |

ICL (uncompressed, no training) on these splits: 35.86

## Budget grid, epochs 0 (splits [13, 21, 87])

| keep ratio | random | random_sink | attn | l2 |
|---|---|---|---|---|
| 1.0 (uncompressed) | 35.86 | 35.86 | 35.86 | 35.86 |
| 0.75 | 36.44 | 36.44 | 35.34 | 35.93 |
| 0.5 | 36.27 | 36.27 | 35.01 | 35.40 |
| 0.25 | 32.63 | 32.33 | 32.37 | 31.18 |
| 0.125 | 33.38 | 33.21 | 32.50 | 31.29 |

ICL (uncompressed, no training) on these splits: 35.86
