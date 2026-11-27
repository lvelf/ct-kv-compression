# Accuracy round 2 plan: answer-shift calibration and candidate ranking loss (target: original CT-KV + 7 points)

Guideline: `docs/experiment-guideline/CTKV_Accuracy_Plus7_Experiment_Guideline.md` (学习时间戳 2026-11-27 16:52:58 EST, pre-stamp sha256 da2ced8f…). Base commit b1cee2a (the version the guideline checked); this plan and the runner are committed on top of it.

**Written before any test score of a new method existed.** The only runs so far are scratch debug runs with `timing_only` jobs (no test question is loaded) on financial_phrasebank and qasc, used to exercise the code and to measure step times.

## Target and pre-registered main comparison

- Panel: NLP-LR, GPT-2 Large, 16-shot, 21 tasks × 10 demonstration seeds, frozen common question IDs, dense production SDPA.
- **Main contrast: `D50_rank_cal − A200`** on the 21-task total `(12 × classification mean + 9 × multiple-choice mean) / 21`, with λ = 0.5 and the `N/A` null template. Classification and multiple choice are also reported separately.
- Historical reference: A200 54.54, target 61.54; D50 (Stage 3) 55.79.
- Wording: point difference ≥ 7 is "point estimate reaches the target"; interval lower bound > 0 is "evidence of a positive gain". This is the historical panel, which has been evaluated many times. There is no independent confirmation set in this round, so no result here is called an independent confirmation.
- Independent data: none has been reserved yet. The MetaICL tasks outside these 21 have never been run in this project and are left untouched as a possible confirmation set; they are not used for any choice in this round.

## Conditions

One original-loss trajectory and one ranking-loss trajectory per arm × task × seed (4 × 210 = 840 trajectories). All use the Stage 1 job verbatim: demos and order, optimizer seed 0, AdamW 1e-3, full batch, leave-one-out, 5% KV token dropout, grad clip 1.0, cosine schedule over 200 steps. Step 50 is step 50 of that schedule.

| condition | demos | trajectory | step | scoring |
|---|---|---|---|---|
| A0, A50, A200 | A | original loss | 0 / 50 / 200 | original |
| A50_cal, A200_cal | A | same caches | 50 / 200 | calibrated |
| D0, D50, D200 | D | original loss | 0 / 50 / 200 | original |
| D50_cal (D200_cal, extra) | D | same caches | 50 / 200 | calibrated |
| D50_rank, A50_rank | D / A | original + ranking loss | 50 | original |
| **D50_rank_cal**, A50_rank_cal | D / A | same caches | 50 | calibrated |
| D_rank@N, D_rank_cal@N (and A) | D / A | same ranking trajectory | N_task (equal GPU time) | original / calibrated |
| D_mix (extra) | D | D0 and D50 caches | — | argmax of 0.5 s_D0 + 0.5 s_D50 (two cache reads per question) |

The original-loss trajectories are rerun (not taken from the archive), because calibration needs null scores from the same cache. Their agreement with Stage 1 / Stage 3 is reported.

## Calibration (fixed)

- s_C(x, y) = −(mean answer-token loss), from the same forward as the original evaluator, reduced in FP32.
- b_t(y) = s_Ct(x∅, y) − s_C0(x∅, y); s*(x, y) = s_Ct(x, y) − λ b_t(y); prediction = first argmax. For the rank conditions, C0 and Ct are those of the ranking trajectory itself.
- **λ = 0.5 (main)**; λ = 0 and 1 are sensitivity results. Nothing is tuned on the panel.
- **Null question (main template `na`):** the question content is replaced by `N/A`, keeping the task format:
  - glue-wnli, sick: `sentence 1: N/A [SEP] sentence 2: N/A`
  - superglue-copa: the question's own `Cause:` / `Effect:` prefix followed by `N/A`
  - all other tasks: `N/A` (their inputs are plain text and do not display the candidates)
- Second template `dots` (`…` in place of `N/A`): sensitivity only.
- Null items are de-duplicated by (rendered null input, candidate text), so classification needs one null forward per label and multiple choice one per distinct candidate text. Rendered samples and hashes: `checks/null_samples.json`.
- **Original scoring** keeps the historical evaluator exactly (fp16-logit loss, first minimum). Calibrated scoring uses the FP32 scores. The number of questions where the FP32 argmax at λ = 0 differs from the original prediction is reported (near-ties).
- No gold label, gold index or test-label statistic enters scoring.

## Ranking loss (fixed)

- L = L_original + 1.0 × L_rank, L_rank = (1/n) Σ_i [logsumexp_{y ∈ Y_i} s_i(y) − s_i(y_i⁺)], in FP32, over the n held-out demonstrations with at least two valid candidates.
- Y_i: the task's label set (classification) or the demonstration's own options (multiple choice), de-duplicated by answer token sequence. Demonstrations with one valid candidate keep the original loss only and are counted.
- All candidates of demonstration i share its KV mask row (leave-one-out span and the same dropout draw as the original-loss forward of that step). The gold score comes from the original-loss forward.
- The gradient is exact and computed in two passes (softmax weights over all candidates without grad, then per-chunk backward), so the softmax denominator always covers every candidate. Chunk size 16.
- Optimizer, clipping and the 200-step schedule are unchanged.

## Equal-step and equal-GPU-time comparisons

- **Equal steps:** 50 updates, original vs ranking loss.
- **Equal GPU time:** the budget is the adaptation time of original A200 on the same task (D50 original is reported as the light reference).
  - Timing units: per task, seed 1000, arms A and D, 20 steps of each loss, no test evaluation, GPU alone.
  - t = median step time after 3 warm-up steps, averaged over the two arms.
  - N_task = clip(5 × floor(200 × t_orig / t_rank / 5), 5, 200), written to `rank_budget.json` before the panel starts.
  - The ranking trajectory is evaluated at step 50 and step N_task and stops at max(50, N_task). Actual times and overruns are reported.

## Statistics

- Paired by task, seed number and question ID on the frozen common IDs; tasks equal-weighted within a family; total = (12 clf + 9 mc) / 21.
- 5,000 paired bootstrap draws (seeds and questions resampled within each task, the same draw for every condition), 95% interval, fixed RNG seed.
- Secondary contrasts: each D condition − D50; A vs D under the same loss and scoring; the interaction (D50_rank_cal − D50_rank) − (D50_cal − D50); cal − original for A50, A200, D50; each condition vs A0, D0, D200, A200_cal.
- Mechanism table: questions split by (D0 right/wrong) × (D50 right/wrong), with count, task-weighted share, accuracy of each new condition and net change vs D50. Also by label, task and number of options.
- Label-shift table (classification): per seed and label, demo label share − pool label share, centred shift b_t(y) − mean_y b_t(y), and the change in predicted share and recall from C0 to Ct, for A and D.
- Coverage is reported as expected / actual units and questions. A missing unit is shown as `pending` or `failed`, with a sensitivity result that counts missing questions as wrong.

## Smoke gate

Smoke jobs (part of the panel): financial_phrasebank and ai2_arc, seed 1000, arm D, both trajectories. `context-tuning/round2_check.py` must pass before the panel starts; accuracy never decides. It checks:

- calibration unit tests (λ = 0 unchanged, C_t = C_0 gives zero shift, hand-made sign test);
- L_rank finite, rank-only gradient non-zero on the KV tensors, model weights without gradient;
- the training-time tensor scorer against the original evaluator (same cache, eval mode), and candidate-order invariance;
- candidates of a demonstration use its mask row, with its own span zero;
- cache checksum, RNG state and train flag unchanged by scoring;
- the original-loss smoke trajectory against Stage 3 (step 0 identical; steps 50 and 200 within 5% flips, the BF16 rerun spread);
- step counts, common-ID coverage, parseable outputs, resume.

Known evaluator property (measured on demonstrations, no test data): with BF16, the same item scored in a different batch composition changes its loss by up to about 0.09 (mean 0.01). This is the original evaluator's behaviour and is left unchanged.

## Queue

Appended to `results/ctkv_stage2/queue/bench.txt` (serial, GPU alone), in this order:

1. `r2_timing`, `r2_budget` (builds the job lists)
2. `r2_smoke`, `r2_check`
3. `r2_D_orig`, `r2_D_rank`, `r2_A_orig` (the four D groups and A200 first)
4. `r2_A_rank`

Runner: `context-tuning/round2_nlplr.py`; jobs: `round2_jobs.py`; calibration: `round2_cal.py`. Each job writes a `.tmp` file, validates it and renames atomically; a restart skips finished jobs. Resume command: rerun the same queue line.

## Follow-up trigger (guideline §6)

- After the main table and the equal-budget comparison: if `D50_rank_cal − A200` is below +7 points on this panel, run the dual-cache routing experiment with the guideline's default rule (TF-IDF fitted on the allowed training pool, H built greedily for the worst-covered half, routing by mean cosine to the 3 nearest demonstrations, ties to G), using the frozen main method for both caches. Its results are exploratory, because the trigger is read from the historical panel.
- If the point estimate reaches +7, the frozen method goes to an independent confirmation first.
