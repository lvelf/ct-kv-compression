# NLP-LR reproduction (GPT-2 Large, hr_to_lr, 21/26 tasks fit in 1024 tokens)

Env: RTX 5090, torch 2.7.1+cu128, transformers 4.47.1, accelerate 1.1.1, bf16. CT-KV: lr 1e-3, 200 epochs, token dropout 0.05.

| Method | split 13 | split 21 | split 42 | split 87 | split 100 | Mean ± std (ddof=1) | Paper Table 1 |
|---|---|---|---|---|---|---|---|
| zeroshot | 35.55 | 35.58 | 35.55 | 35.55 | 35.46 | 35.54 ± 0.05 | 34.9 ± 0.62 |
| icl | 35.25 | 36.18 | 34.98 | 36.14 | 36.12 | 35.74 ± 0.57 | 35.6 ± 0.65 |
| ctkv | 44.78 | 44.64 | 43.06 | 44.86 | 44.42 | 44.35 ± 0.74 | 44.2 ± 0.55 |

## Per-task mean over 5 splits

| Task | zeroshot | icl | ctkv |
|---|---|---|---|
| ai2_arc | 23.4 | 25.8 | 27.2 |
| climate_fever | 19.6 | 19.3 | 27.3 |
| codah | 38.5 | 42.2 | 45.2 |
| commonsense_qa | 29.3 | 33.0 | 44.0 |
| ethos-national_origin | 45.9 | 44.2 | 49.0 |
| ethos-race | 56.9 | 44.9 | 44.9 |
| ethos-religion | 45.2 | 45.0 | 46.9 |
| financial_phrasebank | 14.5 | 24.2 | 58.9 |
| glue-wnli | 47.3 | 30.6 | 47.0 |
| hate_speech18 | 47.0 | 47.0 | 48.9 |
| openbookqa | 32.2 | 27.2 | 29.9 |
| poem_sentiment | 26.4 | 27.0 | 29.5 |
| qasc | 18.3 | 16.5 | 28.6 |
| quarel | 55.0 | 57.1 | 53.7 |
| quartz-no_knowledge | 53.4 | 53.4 | 55.0 |
| quartz-with_knowledge | 50.8 | 50.9 | 51.9 |
| sick | 17.3 | 14.9 | 56.3 |
| superglue-copa | 63.0 | 61.6 | 64.4 |
| tweet_eval-hate | 36.4 | 36.9 | 54.4 |
| tweet_eval-stance_atheism | 9.8 | 23.1 | 35.3 |
| tweet_eval-stance_feminist | 15.9 | 25.6 | 33.1 |
