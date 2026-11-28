# BBH follow-up | Why does KV tuning hurt Boolean accuracy, and where does the time go?

**November 28, 2026 · 180 completed runs · Boolean Expressions and Date Understanding**

We tested whether the Boolean accuracy drop depends on the labels in the demonstrations, which cache tensors receive updates, and how long we tune. We also measured how much computation and GPU memory each setting uses. All runs used Llama 3 8B Instruct on one A100-SXM4-80GB, with 200 fixed test questions per task and three demonstration seeds. Each run produced 200 predictions, for 36,000 prediction records across the study.

## Main insights

1. **Demonstration label balance changed the direction of the Boolean result.** With 10 demonstrations, Full CT-KV lowered accuracy by 6.42 percentage points when only two demonstrations were True. With five True and five False, it raised accuracy by 3.67 points. Doubling the demonstrations while keeping the same imbalance did not remove the drop.
2. **The choice of K or V mattered even when the number of trainable parameters was identical.** On the original 10-shot Boolean prompts, Key-only scored 86.00% and Value-only scored 71.50%. Date showed the opposite preference, at 49.17% and 55.50%, respectively.
3. **The training forward and backward passes consumed 93.4% of adaptation time.** Updating only K or V halved gradient and Adam-state memory, but mean adaptation time stayed near 17.4 s. Physically retaining 25% of demonstration positions shortened adaptation from 17.29 to 14.97 s. Accuracy varied by task and budget.
4. **The two tasks favored different training lengths.** Across the original 10-shot test endpoints, Boolean accuracy fell as the training schedule lengthened. Date improved through 16 steps and gained only another 0.17 points at 32. These results motivate testing a stopping rule that counts the cost of validation.

## 1. What changed when we controlled the labels?

Each entry is the change in Boolean test accuracy after 16 joint KV updates relative to ICL on the **same ordered demonstrations**. Each cell averages three demonstration seeds, with the two optimizer seeds averaged within each demonstration seed.

| True labels among the demonstrations | 10-shot change (pp) | 20-shot change (pp) |
| --- | ---: | ---: |
| 20% True | −6.42 | −6.92 |
| 50% True | +3.67 | +1.50 |
| 80% True | −0.83 | −1.08 |

The test questions and answers stayed fixed at 106 True and 94 False. In the 10-shot, 20%-True condition, tuning reduced the share of test questions answered True from **64.00% to 37.75%**. Accuracy on questions whose correct answer was True fell from 94.03% to 63.21%. Accuracy on False questions rose from 69.86% to 90.96%. The shift toward False explains how tuning could help one class while hurting overall accuracy.

## 2. Training time breakdown

Reference setting for both profiling tables: original 20-shot prompts, full cache, 16 joint KV updates. Values are means across the two tasks and three demonstration seeds, with each of the six runs weighted equally. All times are seconds.

| Requested metric | Measured value | What the timer covers |
| --- | ---: | --- |
| KV cache construction | 0.374 | Build the full prefix cache from the instruction and demonstrations. |
| Training forward passes | 6.300 | Cumulative time computing the training loss across 16 updates. |
| Backward passes | 9.838 | Cumulative time computing gradients across 16 updates. |
| Optimizer stage | 0.161 | Gradient clipping, Adam update and learning-rate scheduler, summed over 16 updates. |
| Mean time per optimization step | 1.041 | Direct measurement of the complete update loop per step, including loop overhead. |
| Total adaptation time | 17.288 | Cache construction, selection or index handling, training and associated overhead. |

The full-cache path spent 0.025 s on selection/index handling. The per-step and total rows already include the training components above. Model loading and test decoding fall outside the adaptation timer. Diagnostic validation used another **3.48 s** on average and was excluded from the reported adaptation time.

## 3. GPU memory breakdown

| Requested metric | Measured value | Scope |
| --- | ---: | --- |
| Retained KV cache | 147.04 MiB | K and V tensor storage for the retained prefix, including the instruction. |
| Activations saved for backward | 1,719.44 MiB | Unique floating-point storage saved by autograd in the first training microbatch, excluding model and prefix storage. Coverage of activations is partial. |
| Gradient memory | 147.04 MiB | Gradients of the trainable cache tensors, measured at the first update. |
| GPU optimizer state | 294.08 MiB | GPU-resident Adam state tensors after the first update. |
| Peak allocated GPU memory during adaptation | 17.63 GiB | Peak allocation recorded by PyTorch during adaptation, with diagnostic validation tracked separately. |

The largest observed increase in live allocations across a training forward pass was **1,801.19 MiB** on average. This includes other allocations and provides a second proxy for activation cost. The global peak attributable specifically to activations remains unknown. These quantities are measured at different points and should not be added to reconstruct the GPU peak.

## 4. Scalability Analysis

### Demonstrations and tokens

Original prompts, full cache, 16 joint updates. Token counts exclude the instruction and are averaged over the three seeds.

| Task | Shots | Mean demo tokens | Mean adaptation |
| --- | ---: | ---: | ---: |
| Boolean Expressions | 10 | 160 | 7.20 s |
| Boolean Expressions | 20 | 320 | 13.71 s |
| Date Understanding | 10 | 1,000 | 8.54 s |
| Date Understanding | 20 | 1,990.67 | 20.87 s |

Date seed 0 used 992 and 1,977 demonstration tokens at 10 and 20 shots. The table reports means across all three seeds. The 5-shot results belong to the earlier November 24 study. Here, doubling the shot count also increased the input length, so their individual contributions to runtime remain unknown.

### KV retention budget

Original 20-shot prompts, 16 joint updates, means across both tasks and three seeds. Retention applies to demonstration KV positions. All instruction positions are kept. These are fresh results from this 180-run study.

| Demo KV retained | Selection | Mean accuracy | KV cache | Peak allocated during adaptation | Adaptation |
| --- | --- | ---: | ---: | ---: | ---: |
| 25% | Answer attention | 69.25% | 38.71 MiB | 16.14 GiB | 14.97 s |
| 50% | Answer attention | 71.50% | 74.81 MiB | 16.64 GiB | 16.54 s |
| 75% | Answer attention | 71.58% | 110.90 MiB | 17.14 GiB | 16.91 s |
| 100% | Full cache | 71.00% | 147.04 MiB | 17.63 GiB | 17.29 s |

**Date Understanding alone.** The same 20-shot comparison below isolates the longer prompts and adds the saved-activation measurement. Each row averages three seeds. The 100% row uses the full cache.

| Demo KV retained | KV cache (MiB) | Peak GPU allocation (GiB) | Adaptation (s) | Saved activations, first microbatch (MiB) | Test accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25% | 64.92 | 16.99 | 15.62 | 1,453 | 57.17% |
| 50% | 127.13 | 17.84 | 17.63 | 1,955 | 60.83% |
| 75% | 189.29 | 18.71 | 19.26 | 2,463 | 60.83% |
| 100% | 251.58 | 19.56 | 20.87 | 2,968 | 59.67% |

On these Date runs, 50% and 75% retention gave the same mean accuracy. The 50% condition used 62.17 MiB less cache and took 1.63 s less to adapt, using unrounded means.

At equal retained bytes, the accuracy differences for **answer attention minus random selection** were −0.50, −0.17 and −1.50 points on Boolean at 25%, 50% and 75% retention. On Date, the corresponding differences were −1.00, +3.50 and +3.17 points. Each seed and budget had one random mask, so the small differences should be interpreted with that source of uncertainty in mind.

### Full KV, Key-only and Value-only

The three tuned methods used the same original prompts, full cache, 16 updates and optimizer seed. The ICL column is the aligned historical zero-update reference.

| Task | Shots | ICL | Full KV | Key-only | Value-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boolean Expressions | 10 | 85.83% | 71.00% | 86.00% | 71.50% |
| Boolean Expressions | 20 | 82.17% | 82.33% | 82.67% | 82.67% |
| Date Understanding | 10 | 48.50% | 55.50% | 49.17% | 55.50% |
| Date Understanding | 20 | 49.83% | 59.67% | 49.67% | 60.17% |

On **10-shot Date**, gradients plus GPU Adam state averaged **383.25 MiB** for joint updates and **191.63 MiB** for either single-tensor method. Adaptation peaks were 17.85 GiB for joint updates and 17.65 GiB for both alternatives. Times were 8.54 s for joint updates, 8.45 s for Key-only and 8.43 s for Value-only. Value-only matched joint accuracy on these runs. Key-only scored 49.17%, close to the historical ICL reference of 48.50%.

At **20 shots across both tasks**, both single-tensor methods still stored the full 147.04 MiB cache. Each used 73.52 MiB of gradients and 147.04 MiB of GPU Adam state. Peak allocation fell from 17.63 to 17.41 GiB. Adaptation took 17.43 s for Key-only and 17.46 s for Value-only, compared with 17.29 s for joint updates.

## 5. Convergence Analysis

**Research question.** As CT-KV updates the cache, does it keep getting better at answering new questions? We followed the same run on validation questions and compared the final test accuracies of separate runs with different training lengths.

### What happened during training?

At each update we logged loss on the demonstrations used for training. At steps 0, 2, 4, 8 and 16 when reached, and at the endpoint of the 32-step runs, we evaluated ten separate validation questions. The following means come from the three original Boolean 10-shot runs with a 16-step schedule.

| Step | Training NLL used for the update | Validation NLL after the update | Validation accuracy |
| ---: | ---: | ---: | ---: |
| 0 | — | 0.212 | 86.67% |
| 2 | 0.02851 | 0.841 | 70.00% |
| 4 | 0.14112 | 1.318 | 70.00% |
| 8 | 0.00053 | 1.799 | 66.67% |
| 16 | 0.14762 | 1.984 | 66.67% |

**What NLL measures.** Negative log-likelihood measures how much probability the model gives the correct answer. For one correct answer token with probability `p`, its loss is `−ln(p)`. If the correct token is “True,” assigning it 90% probability gives a loss of 0.105. Assigning it 60% gives 0.511. The model can answer correctly both times, while NLL reveals the difference in confidence. For a multi-token answer, the evaluation scores each correct next token given the question and preceding correct tokens, then averages over all answer tokens in the validation set. Lower is better.

**Why use it here?** At steps 8 and 16, Boolean validation accuracy was identical at 66.67%, while NLL rose from 1.799 to 1.984. Across the validation answers, the probability scores for the correct tokens worsened even though the number of correct answers stayed the same. Training NLL was already near zero at step 8. Fitting the demonstrations well did not translate into better answers on unseen questions.

### How many updates helped on the test questions?

Each column below reports mean accuracy on the same 200 test questions per task. The 0-step column comes from the earlier matched ICL run. The 4-, 8-, 16- and 32-step columns are fresh, separate runs. Each uses a cosine learning-rate schedule scaled to its own training length. A run designed to end at step 4 can therefore give a different result from checkpoint 4 of a longer run.

| Task, original 10-shot prompts | 0 updates (historical ICL) | 4 updates | 8 updates | 16 updates | 32 updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boolean Expressions | 85.83% | 76.67% | 73.17% | 71.00% | 70.33% |
| Date Understanding | 48.50% | 52.33% | 53.83% | 55.50% | 55.67% |

For **Boolean**, the historical 0-step reference had the highest accuracy on the original 10-shot prompts. The separate label-balanced experiment in Section 1 did benefit from tuning. For **Date**, 16 steps gave 55.50% and 32 gave 55.67%. Doubling the updates added only 0.17 percentage points on average. The study established neither a minimum required step count nor a threshold for satisfactory accuracy.

**Could validation loss help choose a stopping point?** After all four schedules had finished, we chose the training length with the lowest final validation NLL separately for each seed. Among the candidates 4, 8, 16 and 32, the rule chose **4/4/4** for Boolean and **16/8/16** for Date. Compared with always choosing 16, test accuracy was 5.67 points higher on Boolean and 0.50 points lower on Date. Zero steps was outside the candidate set. A real-time rule that stops a running experiment still needs to be evaluated for accuracy and actual time saved.

## Limitations

- **Scope and uncertainty.** Two tasks and three demonstration seeds were evaluated. The controlled label-ratio experiment used two optimizer seeds per tuned condition. Original-prompt comparisons used one optimizer seed, and random compression used one mask per seed and budget. The same 200 test questions per task had also been examined in earlier experiments.
- **Shots and length.** This study profiled 10 and 20 shots. The input grew longer as examples were added. The earlier 5-shot study cannot supply a matched 5-shot breakdown for these profilers.
- **Activation coverage.** The saved-tensor probe covers the first training microbatch. Peak PyTorch allocation covers the entire adaptation, but its activation-only component remains unknown.
- **Stopping decisions.** Each run had ten validation questions, so one changed answer moves its validation accuracy by ten percentage points. The tested training lengths were discrete and each had its own schedule. The data do not establish a universal stopping point.

## Next steps

1. **Test a stopping rule prospectively.** Set an acceptable accuracy loss before the experiment. Compare fixed 16 steps with a rule that can stop at 0, 2, 4, 8 or 16 on a common learning-rate schedule. Choose the rule on development data. During the evaluation, use separate validation questions for stopping and fresh test questions for scoring. Report actual steps, test accuracy and total time including validation.
2. **Test demonstration selection beyond label balance.** On Boolean, compare uniform random sampling, balanced random sampling and balanced sampling that covers different operators and nesting depths. Use the same shot count, order policy and comparable token lengths. Measure accuracy before tuning and after 16 updates.
3. **Isolate input length and check the memory trend.** At a fixed shot count and label mix, compare short and long prompts within the same task. Repeat the activation measurement across microbatches and report validation overhead separately. Include a matched 5-shot arm if the goal is to compare 5, 10 and 20 shots under the same instrumentation.

## Evidence

- [Original GPU results archive](BBH_Mechanisms_wQBC23Zr.tar.gz) and [SHA-256 checksum](BBH_Mechanisms_wQBC23Zr.tar.gz.sha256). The archive includes per-question predictions, validation checkpoints, metrics and worker logs for all 180 formal runs. The separate smoke run is excluded from every table above.
- [Original experiment code and run instructions](source/README.md), [prespecified analysis plan](source/ANALYSIS.md), [formal run plan](results/plan.json) and [recorded environment](results/provenance/).
- [Accuracy by run](results/analysis/absolute_by_run.csv), [paired contrasts](results/analysis/paired_summary.csv) and [contrasts by demonstration seed](results/analysis/paired_by_seed.csv).
- [Time, memory and token counts](results/analysis/resource_profiles.csv), [training loss](results/analysis/training_loss.csv), [validation checkpoints](results/analysis/validation_checkpoints.csv), [validation-based horizon selection](results/analysis/validation_horizon_selection.csv) and [matched historical runs](results/analysis/historical_checks.csv).
