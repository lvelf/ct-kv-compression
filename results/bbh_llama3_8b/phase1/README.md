# Llama 3 8B on BIG-Bench Hard

This independent reconstruction compares standard in-context learning with full CT-KV on 27 BBH task files, five local demonstration splits, and ten ordered demonstrations per task and split. The two methods share one Meta-Llama-3-8B-Instruct checkpoint and an A100. The CT-KV condition uses 16 updates. The model weights remain frozen. The authors' exact BBH launcher and original demonstration IDs were not verified.

[Benchmark results](benchmark_summary.csv) give each task file equal weight within each seed, followed by a mean and sample standard deviation over the five seed scores. The table includes strict and normalized scoring. [Seed coverage](benchmark_seeds.csv) records one row per method and seed. The four-task compression study uses separate matched comparisons.

The model revision is 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2. The source base is 80b951d9662beac2e8ef2091a0b1b8024bbbd807. The experiment overlay SHA-256 is 98c8686aa97ddb0508f76afc5b1f95885d34392c7e5c1338143d510acca0ecc3. The core environment uses PyTorch 2.4.0 with CUDA 12.1, Transformers 4.47.1, BF16 and one A100.

From the repository root, with the model available and the experiment dependencies installed, use a fresh output directory.

```bash
OUT="$(mktemp -d)"
python -m bbh_eval.prepare_benchmark_splits --project-demo-pool --output "$OUT/benchmark27.json"
python -m bbh_eval.experiment paper-all --project-model 8b --root "$OUT" --execute
python -m bbh_eval.summarize_llama_runs --root "$OUT/paper-all" --out "$OUT/paper-summary" --benchmark-manifest "$OUT/benchmark27.json"
```
