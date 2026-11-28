# BBH follow-up · Why did Boolean Expressions lose accuracy?

This study uses the same model and fixed test questions to measure how demonstration label balance, demonstration count, update count, and the choice of trainable cache tensors affect accuracy and the direction of predictions. The primary comparison is Full CT-KV minus Standard ICL with the same demonstrations. The package uses a separate working directory from the November 24 experiment.

## Prespecified design

| Experiment | Formal settings | New runs |
|---|---|---:|
| 1. Demonstration label balance | Boolean Expressions at 10 and 20 shots, with 20%, 50%, or 80% True labels across three demonstration seeds. For each set of demonstrations, run Standard ICL once. Run both Full CT-KV and 25% `answer_attention` CT-KV for 16 updates each with optimizer seeds 0 and 1. | 90 |
| 2. Update count | Boolean Expressions and Date Understanding, original 10-shot prompts and three demonstration seeds. Run Full CT-KV for 4, 8, or 32 updates with optimizer seed 0. | 18 |
| 3. Trainable tensors | Both tasks, original 10- and 20-shot prompts and three demonstration seeds. Retain the full cache and run Key-only and Value-only for 16 updates with optimizer seed 0. | 24 |
| 4. Profiling | Record time and memory during the runs above. For direct ICL, record test inference time and peak GPU allocation. | 0 |
| 5. Random KV selection | Both tasks, original 20-shot prompts and three demonstration seeds. Retain 25%, 50%, or 75% of demonstration KV positions. Run 16 updates with optimizer seed 0 and one random mask per task, seed, and budget. | 18 |
| Fresh matched controls | Twelve full-cache, 16-update controls (two tasks × two shot counts × three seeds), plus 18 original 20-shot `answer_attention` controls (two tasks × three seeds × three budgets). | 30 |
| **Total** | **200 test questions per run, yielding 36,000 formal predictions** | **180** |

Boolean uses the same 200 test questions throughout, with 106 True answers and 94 False answers. Date uses its own original 200 questions. The separate Boolean demonstration pool contains 50 questions, with 29 True and 21 False answers. The True-to-False counts are 2:8, 5:5, and 8:2 at 10 shots, and 4:16, 10:10, and 16:4 at 20 shots. Each demonstration seed fixes the random draws within labels, label positions, and example order. All methods see the same prompt within a matched comparison. At each label ratio, the first ten demonstrations of a 20-shot set form the paired 10-shot set. The other experiments use the November 24 demonstration IDs and order for comparison. Different seeds can select some of the same examples.

The model is Meta-Llama-3-8B-Instruct at revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`. We use BF16, frozen model weights, direct Q/A prompts, greedy decoding, and strict exact-match scoring. Joint KV training uses Adam with a learning rate of 0.001, token dropout of 0.1, leave-one-out masking, and microbatches of two. Each optimizer update accumulates gradients across all demonstrations. Each run sets its cosine learning-rate schedule to its own total number of updates. Retention percentages apply to demonstration token positions only. All instruction positions are retained.

## Running on Vast.ai from a Mac

1. Download `BBH_Mechanisms_v1.zip` on your Mac. Upload it to `/workspace/` through the Jupyter file browser on Vast.ai. The original configuration used one A100-SXM4-80GB, 100 GB of disk space, Python 3.10, and CUDA 12.1.
2. For the first extraction, run the following block in the **Vast.ai Jupyter terminal**. To resume or continue in a newly opened terminal, go directly to the existing experiment directory. If the ZIP file is missing, check where it was uploaded.

```bash
cd /workspace &&
test ! -e bbh-mechanisms-v1 &&
python3 -m zipfile -t BBH_Mechanisms_v1.zip &&
python3 -m zipfile -e BBH_Mechanisms_v1.zip . &&
cd bbh-mechanisms-v1 &&
bash study.sh plan &&
bash study.sh setup &&
bash study.sh access &&
bash study.sh preflight &&
bash study.sh smoke
```

3. To estimate the runtime before launching every configuration, optionally run the five formal pilot configurations with 10 shots, 50% True labels, and demonstration seed 0. The smoke test uses a few questions in a separate directory and is excluded from formal statistics. The `access` command first tries the existing model cache. If Hugging Face authorization is required, enter a read token at the hidden prompt.

```bash
cd /workspace/bbh-mechanisms-v1
bash study.sh start pilot
bash study.sh status pilot
```

4. If you launched the pilot, wait for `5/5 complete` and `driver idle`, then start the full experiment. Use `main` in place of `all` to answer the primary question first. Run one driver at a time. It checks each configuration and skips completed runs.

```bash
cd /workspace/bbh-mechanisms-v1
bash study.sh start all
bash study.sh status all
```

To follow the live log, run these two lines in a second Jupyter terminal. Pressing Ctrl+C in that terminal closes the log viewer.

```bash
cd /workspace/bbh-mechanisms-v1
tail -f "$(ls -t results/driver-logs/*.log | head -n 1)"
```

| Launch command | Configurations in this group | Contents |
|---|---:|---|
| `bash study.sh start main` | 90 | Includes the five pilot runs, training checkpoints, and profiling. |
| `bash study.sh start steps` | 24 | Eighteen runs with alternative update counts and six matched 10-shot, full-cache, 16-update controls. |
| `bash study.sh start params` | 36 | Twenty-four Key-only or Value-only runs and twelve joint, full-cache, 16-update controls. Six controls are shared with the steps group. |
| `bash study.sh start random` | 36 | Eighteen random-selection runs and eighteen attention-selection controls at matched budgets. |
| `bash study.sh start all` | 180 | All unique configurations. The driver skips completed runs. |

To pause, run `bash study.sh stop`, wait for the current configuration to finish writing, and confirm that the driver is idle with `bash study.sh status`. To resume, run `bash study.sh start all` from the same directory. If a worker fails, the driver stops and lists the failure files at the end of the log. Inspect the attempt in `results/attempts/`. After resolving the problem, repeat the same start command. The failed configuration restarts from the beginning. The analysis includes formal results that pass the 200-prediction and validation checks and have the DONE marker.

5. On Vast.ai, analyze and archive the results after completion or an early pause.

```bash
cd /workspace/bbh-mechanisms-v1 &&
bash study.sh status &&
bash study.sh analyze &&
bash study.sh archive
```

The archive command prints the actual path matching `/workspace/BBH_Mechanisms_XXXXXXXX.tar.gz` and its matching `.sha256` file. Download both through Jupyter and run the following in **Mac Terminal**. If several archives are present, this uses the most recently modified matching checksum in Downloads.

```bash
cd "$HOME/Downloads" &&
sumfile=$(ls -t BBH_Mechanisms_*.tar.gz.sha256 | head -n 1) &&
test -n "$sumfile" &&
shasum -a 256 -c "$sumfile"
```

Wait for `OK` after the archive name before ending the rental. Background jobs launched with `start` run while the instance is active. Stopping or destroying the Vast.ai instance interrupts the job. The archive command can also back up partial results.

## Interpreting the results

| Hypothesis | Prespecified comparison and expected evidence |
|---|---|
| Skewed demonstration labels contribute to the accuracy drop. | At fixed shot count and seed, calculate the change from ICL to 16-update Full CT-KV in accuracy and share of predictions that are True. Compare each skew direction separately with 50% True. A prediction shift toward the majority demonstration label, lower accuracy on the underrepresented label, and a larger overall drop under skew support this explanation. |
| More demonstrations reduce the drop. | Within each label ratio and seed, compare the tuned-minus-untuned accuracy change at 20 shots with the corresponding change at 10 shots. Check whether this difference varies by label ratio. |
| The number of trainable cache parameters may matter. | With the same full cache, compare Key-only, Value-only, and joint updates. Key-only and Value-only train the same number of parameters with different computational roles. Differences between them implicate the choice of tensors. Retaining 25% by attention changes both visible information and trainable parameter count. Its final difference from full-cache ICL combines selection and tuning. |
| Some tasks may need fewer updates. | Inspect validation checkpoints at steps 0, 2, 4, 8, and 16 along one 16-step schedule. Compare test scores at the endpoints of separate 4-, 8-, 16-, and 32-step schedules. Each has its own cosine schedule, so step 4 of a 16-step run and the endpoint of a 4-step run are distinct measurements. |
| Attention selection outperforms random selection. | Pair the 20-shot `random` and `answer_attention` runs by question and demonstration seed. Compare accuracy, time, and memory at the same number of retained bytes. |

Each tuned run holds out ten questions from the demonstration pool for validation. Boolean validation always uses five True and five False questions. Validation questions are disjoint from the demonstrations for that run and from the formal test set. At steps 0, 2, 4, 8, and 16 when reached, and at the scheduled endpoint (including 32), record greedy accuracy and mean negative log-likelihood (NLL) per gold answer token. Lower NLL indicates higher probability assigned to the reference tokens. The retrospective stopping analysis chooses the prespecified training length with the lowest final validation NLL, breaking ties in favor of fewer updates. It compares the chosen schedule with fixed 16 on the test set. Running all schedules before making the choice yields no realized GPU time savings. Intermediate checks and final test scoring use separate questions. Because earlier analyses examined aggregate results on these 200 test questions, mechanism hypotheses should be tested again on an independent question set.

Each run saves `results/runs/<id>/predictions.jsonl`, `validation.json`, `metrics.json`, and `DONE.json`. Analysis outputs reside in `results/analysis/`. Start with `paired_summary.csv` and `paired_by_seed.csv`, then inspect `validation_checkpoints.csv` and `resource_profiles.csv`. On Boolean, the True prediction count includes every question answered True. Separate class-specific accuracies cover questions with gold True and gold False labels. Date uses the original exact-match rule for multiple-choice answers.

For paired comparisons with two optimizer seeds, first average the seeds within a demonstration set. Then compute the mean and its standard error across the three demonstration seeds (sample SD divided by √3). Report each seed's result and direction. The fixed test questions and potentially overlapping demonstration sets limit that standard error to variation across these sampled sets. A difference near zero alone cannot establish equivalence. Historical settings had at most one random-selection seed per condition, which limits generalization across random masks.

Timing covers cache construction, selection, training forward and backward passes, the optimizer stage (including clipping and scheduling), mean time per update, and total adaptation time. Validation and inference are timed separately. Synchronized GPU timing introduces overhead that matters when comparing with earlier uninstrumented runs. Memory measurements include tensor bytes for the cache, gradients, and Adam state, as well as live and peak PyTorch allocations. For the first microbatch only, the study measures floating-point storage saved by autograd for backward, excluding model and prefix storage. Values measured at different times and phase-specific peaks cannot be added to reconstruct total GPU memory. See `ANALYSIS.md` for detailed table definitions.
