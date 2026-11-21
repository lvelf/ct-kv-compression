# Stress test B3: compressing the CT-KV initialization (NLP-LR, GPT-2 Large)

Splits: [87]. Uncompressed baselines on these splits: ICL **36.14**, CT-KV (200 epochs) **44.86**.

Each demonstration keeps ceil(ratio × its length) tokens. `e0` means compression then ICL (no training), and `e200` means compression then CT-KV.
"Gain kept" is (score − ICL) / (uncompressed CT-KV − ICL).

| method | keep ratio | kv len | e0 (compressed ICL) | e200 (compressed CT-KV) | gain kept | train s/task | splits |
|---|---|---|---|---|---|---|---|
| attn | 0.5 | 229 | nan | 43.14 | 80% | 8.0 | 0/1 |
| attn | 0.25 | 118 | 32.99 | 40.06 | 45% | 7.6 | 1/1 |
| attn | 0.125 | 63 | nan | 38.87 | 31% | 7.4 | 0/1 |
| first | 0.25 | 118 | nan | 39.81 | 42% | 7.6 | 0/1 |
| l2 | 0.5 | 229 | nan | 41.97 | 67% | 8.0 | 0/1 |
| l2 | 0.25 | 118 | 29.97 | 39.65 | 40% | 7.7 | 1/1 |
| l2 | 0.125 | 63 | nan | 38.55 | 28% | 7.4 | 0/1 |
| l2_high | 0.25 | 118 | nan | 36.20 | 1% | 7.6 | 0/1 |
| last | 0.25 | 118 | nan | 35.45 | -8% | 7.6 | 0/1 |
| pool | 0.25 | 118 | 34.33 | 35.62 | -6% | 7.7 | 1/1 |
| random | 0.25 | 118 | 31.88 | 36.17 | 0% | 7.7 | 1/1 |
