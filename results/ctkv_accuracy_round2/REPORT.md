# CT-KV accuracy round 2: answer-shift calibration and candidate ranking loss — report

Guideline: `docs/experiment-guideline/CTKV_Accuracy_Plus7_Experiment_Guideline.md` (学习时间戳 2026-11-27 16:52:58 EST). Plan: `PLAN.md` (pre-registered before any new-method score; the trigger evaluation is recorded there).

## The three answers

Panel: NLP-LR, 21 tasks, GPT-2 Large, 16-shot, dense attention, 10 demonstration seeds, 10,096 common questions. Accuracy in points, tasks equal weight, 95% paired bootstrap.

1. **Gain over original CT-KV (A200): +0.20 [−0.97, +1.36].** The pre-registered main method (selected demonstrations, ranking loss, 50 steps, calibration λ = 0.5) scores 54.71 against 54.52 for A200. The target was +7 (61.54). **The target is missed by about 6.8 points, and the gain is not distinguishable from 0.**
2. **Where the change comes from** (main method against D50, net −1.07 points; questions grouped by whether D0 and D50 answer them correctly):

   | group | share of questions | net change vs D50 |
   |---|---|---|
   | D0 right, D50 wrong (old answers recovered) | 8.6% | +3.52 |
   | both wrong (newly solved) | 35.6% | +1.06 |
   | D0 wrong, D50 right (CT-KV gains lost) | 13.9% | −4.61 |
   | both right (lost) | 41.9% | −1.03 |

   The method recovers 41% of the answers that CT-KV broke, but it gives back a third of the answers that CT-KV fixed. It solves only 3% of the questions that both D0 and D50 get wrong.
3. **Extra cost** (D arm, medians per task × seed, GPU alone):

   | | original loss | ranking loss |
   |---|---|---|
   | adaptation to step 50, classification / multiple choice | 2.18 s / 2.02 s | 7.64 s / 8.71 s |
   | time per step (ratio by task) | 1× | 2.3–10.6× |
   | peak memory, classification / multiple choice | 6,224 / 5,813 MiB | 10,369 / 9,612 MiB |

   Calibration adds one null scoring pass at step 0 and one at the read step: about 0.01 s for classification (2 null items), and about 2.0 s for multiple choice (a median of 1,178 null items, more than the 1.6 s spent scoring the questions).

## Status

- **Complete.** Main panel 840 / 840 trajectories (arms A and D × original and ranking loss × 21 tasks × 10 seeds). Dual-cache follow-up 630 / 630. No failure, no retry, no out-of-memory event.
- **Coverage is complete**: every task has 10 seeds and all of its common questions in every condition (`coverage.csv`, `dual_coverage.csv`).
- **Smoke gate passed** before the main run (`checks/smoke_check.json`): step 0 equals Stage 3 exactly; steps 50 and 200 differ from Stage 3 on at most 3.1% of questions, which is the known BF16 nondeterminism.
- **The panel reproduces history**: A200 54.52 (historical 54.54), D50 55.78 (55.79).
- **The run had the GPU to itself** (serial queue): 16:58–21:35 EST for the main panel, 21:35–00:07 for the dual-cache follow-up.
- **No independent confirmation set exists.** All results are on the historical panel, which has been evaluated many times. The dual-cache results are exploratory.

## Main table

| condition | total [95% CI] | classification (12) | multiple choice (9) | diff vs A200 |
|---|---|---|---|---|
| A0 (ICL, random demos) | 50.04 [48.46, 51.65] | 57.17 | 40.54 | −4.47 [−5.78, −3.17] |
| A200 (original CT-KV, baseline) | 54.52 [53.19, 55.81] | 61.80 | 44.80 | — |
| D0 | 50.50 [49.02, 52.04] | 57.93 | 40.60 | −4.01 [−5.22, −2.81] |
| D50 | 55.78 [54.45, 57.12] | 63.55 | 45.42 | +1.26 [+0.20, +2.37] |
| D200 | 55.41 [54.09, 56.71] | 63.29 | 44.90 | +0.89 [−0.13, +1.93] |
| D50 + calibration | 54.15 [52.66, 55.64] | 60.57 | 45.59 | −0.37 [−1.52, +0.80] |
| D50 + ranking loss | 56.16 [54.77, 57.48] | 64.14 | 45.52 | +1.64 [+0.57, +2.72] |
| **D50 + ranking + calibration (main)** | 54.71 [53.24, 56.17] | 61.68 | 45.43 | **+0.20 [−0.97, +1.36]** |
| equal GPU time: ranking at step N | 55.45 [54.11, 56.78] | 63.70 | 44.46 | +0.94 [−0.11, +1.95] |
| equal GPU time: ranking at step N + calibration | 54.74 [53.31, 56.16] | 62.27 | 44.69 | +0.22 [−0.89, +1.33] |
| D_mix (0.5 s_D0 + 0.5 s_D50, two cache reads) | 53.00 [51.55, 54.52] | 60.12 | 43.50 | −1.52 [−2.69, −0.36] |

The best condition is D50 with the ranking loss and no calibration, at +1.64 over A200. Of this, +1.26 is already there in D50 (demonstration selection plus stopping at 50 steps).

## What each component does

### Calibration lowers accuracy at step 50

- **D50_cal − D50: −1.63 [−2.66, −0.63].** The loss is all in classification (−2.98 [−4.73, −1.28]); multiple choice is unchanged (+0.17 [−0.31, +0.64]). The A arm shows the same (A50_cal − A50: −1.31 [−2.14, −0.54]).
- **The loss is concentrated in tasks where CT-KV works by moving the label prior.** On sick, D50 is 64.4 and D50_cal is 42.4. On financial_phrasebank it is 68.0 and 63.7. Stage 3 found the same two tasks to be the ones where ICL sits near 29% and the full update is needed. Removing the answer shift there removes most of what CT-KV learned.
- **Calibration helps on a few tasks**: poem_sentiment 62.3 → 65.7, superglue-copa 67.7 → 68.8.
- **A stronger correction is worse.** λ = 1 gives −3.17 [−4.36, −2.01] against D50. The "…" null template gives −2.59 [−3.61, −1.57].
- **At 200 steps calibration is neutral**: A200_cal − A200 +0.05 [−0.60, +0.69], D200_cal − D200 +0.12 [−0.57, +0.80]. For multiple choice it is slightly positive at 200 steps (A arm +0.66 [+0.03, +1.26]).
- **Label statistics** (`label_shift.csv`): calibration moves the predicted label shares away from the gold shares (mean absolute gap 0.164 at D50, 0.238 after calibration; 0.289 at D0). The size of the shift is not explained by how far the demonstration label shares are from the pool shares (correlation −0.15 to +0.06).
- **Scoring in FP32 without a correction (λ = 0) changes nothing**: −0.03 [−0.11, +0.02].

### The ranking loss gives a small gain at equal steps and none at equal GPU time

- **Equal steps:** D50_rank − D50 +0.37 [−0.13, +0.87]; A50_rank − A50 +0.21 [−0.24, +0.64]. Neither interval excludes 0.
- **Equal GPU time:** the ranking loss costs 2.3–10.6 times as much per step, so the budget is 15–85 steps by task (`rank_budget.json`). At that budget it is 55.45, which is below D50 (55.78) and 0.70 below ranking at 50 steps (−0.70 [−1.24, −0.20]). D50 costs a quarter of the A200 adaptation time and is higher than every equal-time ranking condition.
- **No interaction with calibration:** +0.19 [−0.41, +0.77].
- **Per task** (D50_rank − A200): the largest gain is climate_fever, +10.9 [+6.8, +15.2]; D50 alone already has +10.1 there. Most other tasks are within ±2.

### Demonstration selection is the only component with a clear effect

- D50 − A50: +1.00 [+0.05, +2.00]. With the ranking loss: +1.16 [+0.14, +2.19]. The effect is in classification (+1.70 and +2.02); multiple choice is 0.

## Dual-cache routing (guideline §6; triggered, exploratory)

The trigger fired because the main contrast is below +7. The specification was frozen in `PLAN.md` at 17:15, before any main result. G is the D demonstrations; H is 16 further pool examples chosen for TF-IDF coverage; each question goes to the cache whose 3 nearest demonstrations are more similar. Every cache uses the main method. This uses 32 labelled examples and two adaptations.

| condition | total | classification | multiple choice | diff vs always G | diff vs A200 |
|---|---|---|---|---|---|
| always G (= main method) | 54.71 | 61.68 | 45.43 | — | +0.20 [−0.97, +1.36] |
| always H | 54.16 | 60.33 | 45.93 | −0.56 [−1.90, +0.71] | −0.36 [−1.78, +0.99] |
| G/H random routing | 54.37 | 61.04 | 45.48 | −0.34 [−1.14, +0.41] | −0.15 [−1.35, +1.04] |
| **G/H text routing** | 54.59 | 61.41 | 45.49 | **−0.13 [−0.87, +0.61]** | **+0.07 [−1.14, +1.23]** |
| R1/R2 (random caches) text routing | 54.17 | 60.55 | 45.65 | −0.55 [−1.55, +0.43] | −0.35 [−1.45, +0.72] |
| per-question oracle over G and H (not deployable) | 64.51 | 71.47 | 55.23 | +9.80 [+8.88, +10.79] | +9.99 [+8.73, +11.20] |

- **Text routing does not improve on a single cache.** It is equal to always G, equal to random routing (+0.22 [−0.40, +0.83]) and equal to routing between two random caches (+0.42 [−0.48, +1.32]). It is 1.19 below plain D50 (−1.19 [−2.24, −0.24]), because both caches carry the calibration loss.
- **The routing signal carries no information about which cache is right.** On questions routed to G, G scores 54.72 and H 53.76. On questions routed to H, G scores 54.65 and H 54.69. H was built to cover different text (target-subset coverage 0.106 by G, 0.524 by H; |G ∩ H| = 0), and 43% of questions go to H, so the rule does route. Text similarity to the demonstrations does not predict correctness.
- **The two caches do disagree usefully.** They agree in correctness on 79.9% of questions; only G is right on 10.4% and only H on 9.8%. The oracle is therefore 9.8 points above either cache. The oracle uses the gold answer and both reads, so it is an upper bound for a selector, not an accuracy. Two random 16-shot caches would give a similar bound; this was not computed.
- **Cost:** a second adaptation (median 7.7 s to step 50) and its null scoring; question scoring stays at one cache read per question.

## Decision

1. **The +7 target is not met, and nothing tested here comes close.** The best condition is +1.64 over A200. No method claim.
2. **Drop the answer-shift calibration at 50 steps.** It removes the label-prior shift that is CT-KV's gain on the tasks where ICL is badly miscalibrated. At 200 steps it is harmless but gives nothing on the total.
3. **The ranking loss is not worth its cost.** +0.37 at equal steps is inside the noise, and at equal GPU time it is below the original loss at 50 steps.
4. **Keep D50 as the working configuration**: selected demonstrations, original loss, 50 steps. It is +1.26 [+0.20, +2.37] over A200 at a quarter of the adaptation time.
5. **TF-IDF routing between two caches does not work.** The 9.8-point oracle gap shows headroom in choosing between caches per question, but any selector would have to use a signal other than text similarity (for example, the caches' own scores), and it must be designed and validated on tasks outside this panel.

## Not done

- **LoRA-TTT reference and BBH transfer check** (guideline §7, optional): not run.
- **Independent confirmation:** not applicable, because no method reached the target.

## Files

- `REPORT_results.md`: all generated tables, including per-task rows, the full contrast list, flip groups, label shift, cost and the per-task step budget.
- `conditions.csv`, `contrasts.csv`, `flip_groups.csv`, `label_shift.csv`, `cost.csv`, `coverage.csv`.
- `predictions.jsonl.gz`: per-question scores and null scores at every checkpoint.
- Dual cache: `REPORT_dual_results.md`, `dual_conditions.csv`, `dual_contrasts.csv`, `dual_routing.csv`, `dual_cost.csv`, `dual_coverage.csv`, `dual_selection.jsonl`, `dual_routes.json.gz`, `dual_predictions.jsonl.gz`.
- Manifests: `jobs_*.jsonl`, `rank_budget.json`. Smoke gate: `checks/`.
- Raw per-trajectory outputs: `raw_archive.tar.xz`.
- Code: `context-tuning/round2_{nlplr,jobs,cal,check,analyze,dual_jobs,dual_analyze}.py`.
