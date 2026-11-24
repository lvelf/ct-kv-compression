# BBH reconstruction of Standard ICL and Full CT-KV

Phase 1 independently reconstructs the **Standard ICL and Full CT-KV results on BIG-Bench Hard (BBH)** from Table 1 of *Context Tuning for In-Context Optimization*. With the same checkpoint and demonstrations for both methods, strict accuracy rose from 49.13% with ICL to 56.38% with Full CT-KV. The **7.25 percentage-point gain** closely matches the **7.5-point gain** between the paper's reported means. Both absolute scores were lower in this reconstruction. [Lu et al., Table 1](https://arxiv.org/html/2507.04221v3#S5.T1)

| Source | Standard ICL accuracy | Full CT-KV accuracy | Full CT-KV gain over ICL |
|---|---:|---:|---:|
| Published BBH results | 50.4 ± 0.78% | 57.9 ± 0.78% | +7.5 percentage points |
| This reconstruction (strict scoring) | 49.13 ± 0.49% | 56.38 ± 0.89% | +7.25 percentage points |

The table shows mean accuracy and standard deviation across five demonstration splits. Relative to the paper's displayed means, our ICL score was 1.27 points lower and our Full CT-KV score was 1.52 points lower. The difference between the two gains was about 0.25 points. [Reconstruction results](benchmark_summary.csv)

The absolute gaps are about three combined standard errors. Treating the two studies’ five-seed means as independent gives a standard error of 0.41 points for ICL and 0.53 for Full CT-KV, using `sqrt((SD_paper² + SD_ours²) / 5)`. The respective gaps are 3.07 and 2.86 times those values. Seed variation alone is a weak explanation under this assumption. The paper reports rounded summaries and the cross-study seed pairing is unknown, so this comparison cannot establish the cause of either gap. [Our five seed scores](benchmark_seeds.csv) · [Published Table 1](https://arxiv.org/pdf/2507.04221v3#page=9)

### What could explain the absolute score differences?

- **Model checkpoint.** We used a pinned revision of `meta-llama/Meta-Llama-3-8B-Instruct`. The paper identifies the BBH model as Llama 3 8B. We could not verify the exact checkpoint revision used to produce Table 1.
- **Evaluation code.** We ran our own BBH evaluation code. We could not verify the exact launcher and answer-scoring implementation used for the paper's Table 1. Differences in how generated text is interpreted could affect measured accuracy.

These are possible explanations. Our experiments did not isolate the effect of any one of them.

### Experiment settings

We evaluated 27 BBH task files with ten ordered demonstrations per task and split on one A100. Full CT-KV updated the cache with optimizer seed 0 and frozen model weights. The five splits vary the demonstrations and their associated evaluation questions.

| BBH Full CT-KV setting | Reconstruction |
|---|---|
| Optimizer and learning rate | Adam, initial LR `1e-3`, cosine decay without warm-up |
| KV updates | 16 per demonstration set |
| Token dropout | 0.10 during optimization |
| Microbatch and gradient accumulation | 2 demonstrations per microbatch, 5 microbatches per update across all 10 demonstrations |

These settings match the paper’s BBH CT-KV configuration. [Appendix C and Table 6](https://arxiv.org/pdf/2507.04221v3#page=19) · [Optimization code](../../../bbh_eval/ctkv.py)

[Benchmark results](benchmark_summary.csv) give each task file equal weight within each seed, followed by a mean and sample standard deviation over the five seed scores. The table includes strict and normalized scoring. [Seed coverage](benchmark_seeds.csv) records one row per method and seed. The four-task compression study uses separate matched comparisons.

The model revision is 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2. The source base is 80b951d9662beac2e8ef2091a0b1b8024bbbd807. The experiment overlay SHA-256 is 98c8686aa97ddb0508f76afc5b1f95885d34392c7e5c1338143d510acca0ecc3. The core environment uses PyTorch 2.4.0 with CUDA 12.1, Transformers 4.47.1, BF16 and one A100.

<details>
<summary>Full 10-shot prompt used for Boolean Expressions</summary>

Both phases used the direct Q/A format below with a single beginning-of-sequence token and no chat template. This is the exact prompt text for demonstration seed 0 and held-out row 151. The final `A:` is where generation starts. [Prompt builder](../../../bbh_eval/run_llama_icl.py#L108-L118) · [Split IDs](../../../bbh_eval/splits.json)

```text
Evaluate the result of a random Boolean expression. Answer with only 'True' or 'False'.

Q: False or ( False ) and not False is
A: False

Q: False or not not True or not False is
A: True

Q: False and False or not ( True ) is
A: False

Q: False or not ( True ) and True is
A: False

Q: False and not ( False and True ) is
A: False

Q: True and not True or True and True is
A: True

Q: False or not False or True or True is
A: True

Q: not not not True and False and False is
A: False

Q: False and False and not not not True is
A: False

Q: not ( False ) or not not False is
A: True

Q: not True or True and False or False is
A:
```

</details>

From the repository root, with the model available and the experiment dependencies installed, use a fresh output directory.

```bash
OUT="$(mktemp -d)"
python -m bbh_eval.prepare_benchmark_splits --project-demo-pool --output "$OUT/benchmark27.json"
python -m bbh_eval.experiment paper-all --project-model 8b --root "$OUT" --execute
python -m bbh_eval.summarize_llama_runs --root "$OUT/paper-all" --out "$OUT/paper-summary" --benchmark-manifest "$OUT/benchmark27.json"
```
