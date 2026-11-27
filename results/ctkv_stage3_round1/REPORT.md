# CT-KV Stage 3 round 1: half-delta KV update and same-trajectory early stopping — report

Guideline: `docs/experiment-guideline/CTKV_Stage3_Round1_Guideline.md` (学习时间戳 2026-11-27 11:46:22 EST). Plan: `PLAN.md` (pre-registered).

All numbers are in these files:
- `REPORT_results.md`
- `conditions.csv`, `contrasts.csv` (family and per-task), `per_seed.csv`
- `bbh_diagnostics.csv`, `cost.csv`, `geometry.csv`, `coverage.csv`
- `predictions.jsonl.gz` (per-question prediction and correctness for all five conditions)
- `checks/smoke_check.json`

Raw per-trajectory outputs are archived in `raw_archive.tar.xz`.

## Status

- **Complete.** 230 / 230 trajectories, 1,150 condition evaluations:
  - BBH Boolean: 20 seeds.
  - NLP-LR: 21 tasks × 10 seeds.
- **Coverage is complete.** Every task has all seeds and 100% of its common questions in all five conditions. Nothing was dropped or counted as wrong.
- **All runner technical checks pass on every trajectory:**
  - C_0 and the training cache are unchanged by evaluation.
  - The training RNG and generator are untouched by evaluation.
  - Question IDs are aligned across conditions, and all five conditions are present.
  - Optimizer and scheduler steps equal T.
- **Smoke gate passed at 14:45** (boolean_expressions, financial_phrasebank and ai2_arc, seed 1000):
  - Interpolation at weight 0 and 1 is bitwise equal to C_0 and C_T, with identical predictions.
  - `icl` matches Stage 1 exactly.
  - `full` differs from the Stage 1 endpoint on 1/200, 4/299 and 25/970 questions. This is the expected BF16 backward nondeterminism; a self-rerun differs by a similar amount.
- **The panel reproduces Stage 1** (D-arm reference values):
  - BBH: icl / full 81.00 / 79.70 (Stage 1: 81.00 → 79.75).
  - Classification: 57.93 / 63.39 (Stage 1: 57.93 → 63.33).
  - Multiple choice: 40.60 / 44.97 (Stage 1: 40.60 → 44.95).
- **The run had the GPU to itself** (serial queue, after the Stage 2 benchmark), 14:44–16:02 EST.

## Five-condition accuracy (points; seed mean per task, tasks equal weight; 95% paired bootstrap)

| family | icl (0 steps) | early_quarter (T/4) | early_half (T/2) | full (T) | half_delta (T + lerp) |
|---|---|---|---|---|---|
| BBH Boolean (Llama-3-8B-Instruct, T = 16) | 81.00 | 80.15 | 79.75 | 79.70 | **81.25** |
| NLP-LR classification, 12 tasks (GPT-2 Large, T = 200) | 57.93 | **63.56** | 63.44 | 63.39 | 61.48 |
| NLP-LR multiple choice, 9 tasks | 40.60 | **45.43** | 45.06 | 44.97 | 44.03 |
| (classification, runner macro-F1, secondary) | 32.42 | 42.49 | 43.95 | 44.32 | 38.62 |

## Main comparison: half_delta − full

| family | diff | 95% CI | seed-only CI | fixed / broken Q | seeds better / worse / tie | pre-registered reading |
|---|---|---|---|---|---|---|
| BBH | +1.55 | [−0.05, +3.12] | (+0.62, +2.50) | 156 / 94 | 14 / 3 / 3 | **not shown to improve**: the lower bound is −0.05, not above 0 |
| classification | −1.91 | [−3.44, −0.30] | (−2.86, −0.95) | 3020 / 5094 | 46 / 47 / 27 | **gain not kept**: the interval crosses −1.0 and lies entirely below 0 |
| multiple choice | −0.94 | [−1.91, +0.05] | (−1.38, −0.52) | 3257 / 3878 | 34 / 54 / 2 | **uncertain**: the interval crosses −1.0 |

## Answers to the three questions

### 1. Does the half-delta update recover the BBH loss?

**Mostly, but not significantly at the pre-registered bar.**

- The comparison with `icl` shows the recovery:
  - full − icl: −1.30 [−3.97, +1.42]
  - half_delta − icl: +0.25 [−1.45, +1.95]

  So half_delta sits at the ICL level, and the point estimate recovers the whole CT-KV loss.
- Against full, the interval just reaches 0 (lower bound −0.05). 14 of 20 seeds improve and 3 get worse. Resampling over seeds only gives (+0.62, +2.50).
- On the pre-registered rule (lower bound above 0) we **cannot** report an improvement on this panel.
- **Label pattern.** half_delta returns True recall from 0.882 (full) to 0.904, the ICL level of 0.906. False recall rises slightly (0.702 → 0.710). Predicted-True is 0.615, against 0.621 at icl and 0.607 at full. Gold True share is 0.530.
- **Caveat: interpolation in bf16 at a tiny displacement.**
  - On BBH, ‖C_T − C_0‖ / ‖C_0‖ is 0.0006 for K and 0.010 for V. This is far below bf16's relative precision (about 0.4%).
  - So `lerp(C_0, C_T, 0.5)` cast back to bf16 rounds many elements onto one of the two endpoints.
  - The realised displacement has norm ratio 0.525 and cosine 0.922 for K (0.518 / 0.962 for V) relative to C_T − C_0, instead of 0.5 / 1.0.
  - For NLP-LR, where the displacement is larger, it is 0.500 / 0.999.
  - BBH's half_delta is therefore "approximately half the update, with rounding noise in K". This is the production cache dtype, as the guideline specifies.

### 2. How much of the NLP-LR gain does half_delta keep?

**About two-thirds for classification and four-fifths for multiple choice. The loss against full is real for classification.**

- Against icl:
  - classification: half_delta +3.55 [+2.38, +4.78] vs full +5.46
  - multiple choice: half_delta +3.43 [+2.64, +4.20] vs full +4.38
- Against full: classification −1.91 [−3.44, −0.30] (entirely below 0) and multiple choice −0.94 [−1.91, +0.05]. Neither family passes the −1.0-point retention bound.
- **The loss is concentrated in the tasks where CT-KV gains the most:**

  | task | full − icl | half_delta − full |
  |---|---|---|
  | financial_phrasebank | +39.5 | −14.9 [−22.5, −7.4] |
  | sick | +33.7 | −16.2 [−22.6, −9.8] |
  | qasc | +11.7 | −3.4 [−5.4, −1.6] |

- **Where full is below icl, half_delta recovers part of the loss:**

  | task | full − icl | half_delta − full |
  |---|---|---|
  | poem_sentiment | −5.3 | +3.1 |
  | tweet_eval stance (atheism / feminist) | −3.9 / −5.5 | +5.0 / +3.9 |
  | ethos (national_origin / religion) | −1.6 / −1.2 | +1.5 / +1.0 |
  | quarel | −2.1 | +1.4 |

  None of these is individually significant.
- **Reading.** Halving the displacement behaves like a shrinkage toward ICL that is applied uniformly. It helps where the update overshoots (BBH and the small NLP-LR tasks where CT-KV hurts). It costs a lot where the full update is needed (label-calibration tasks such as financial_phrasebank and sick, where ICL sits near 29%). One fixed weight of 0.5 does not fit both.

### 3. Does early stopping reach the same performance, and save adaptation cost?

**Yes on NLP-LR: early stopping is at least as accurate as full, at a quarter of the cost. On BBH it does not help.**

| contrast | BBH | classification | multiple choice |
|---|---|---|---|
| early_quarter − full | +0.45 [−0.65, +1.48] | +0.17 [−0.89, +1.21] | +0.45 [−0.41, +1.34] (seed-only +0.12, +0.79) |
| early_half − full | +0.05 [−0.43, +0.58] | +0.05 [−0.38, +0.50] | +0.09 [−0.27, +0.43] |
| early_half − half_delta | −1.50 [−2.97, −0.08] | +1.96 [+0.32, +3.53] | +1.03 [+0.07, +1.95] |

- **NLP-LR.**
  - early_quarter (50 of 200 steps) and early_half (100 steps) are statistically indistinguishable from full. The point estimates are slightly higher.
  - early_quarter has the highest accuracy of all five conditions for both NLP-LR families (63.56 and 45.43).
  - Measured adaptation cost to the checkpoint is 25.3% (T/4) and 50.2% (T/2) of the full adaptation (median 9.30 s for classification, 8.79 s for multiple choice), as expected from equal-cost steps.
  - The one place the runner macro-F1 disagrees: classification macro-F1 at T/4 is 42.49, against 44.32 at full. Accuracy is the pre-registered metric.
- **BBH.**
  - Early stopping stays at the full level (79.75 at T/2, 80.15 at T/4) and does not recover the ICL level.
  - The real half-way cache is clearly worse than the endpoint midpoint (early_half − half_delta −1.50 [−2.97, −0.08]).
- **Why: displacement geometry.**
  - The trajectory is front-loaded. At T/2 the displacement is already 95–98% of the final one in norm, with cosine above 0.99. At T/4 it is 74–79%, with cosine 0.93–0.95.
  - So early_half is essentially C_T. Even early_quarter is a 3/4-size step in nearly the final direction, while half_delta is a true half-size step.
  - Under the cosine schedule, most of the movement happens in the first quarter. Early stopping therefore cannot imitate shrinkage, and conversely the full trajectory keeps refining the direction but barely grows in norm.

## Cost (medians; GPU alone)

| family | prefill s | adaptation s → T/4 / T/2 / T | interpolation s | evaluation s (5 conditions) | peak MiB |
|---|---|---|---|---|---|
| BBH | 0.028 | 0.68 / 1.34 / 2.64 | 0.0010 | 14.56 | 15,882 |
| classification | 0.010 | 2.34 / 4.67 / 9.30 | 0.0011 | 2.00 | 6,465 |
| multiple choice | 0.009 | 2.21 / 4.41 / 8.79 | 0.0011 | 8.06 | 6,044 |

- **half_delta costs the full T steps** plus about 1 ms of interpolation, and its peak memory is the same as full.
- **Early stopping at T/4 saves 75% of the gradient steps and of the adaptation time.**
- Adaptation times exclude the mid-trajectory evaluations.

## Decision for the next round

1. **No method claim from round 1.** BBH half_delta misses the pre-registered bar (lower bound −0.05), and it fails the NLP-LR retention bound on both families. Because the panel has been evaluated many times, the near-miss on BBH is not evidence by itself.
2. **Candidate for round 2: fixed early stopping at T/4.** On NLP-LR it equals full and costs a quarter of the adaptation. Per the guideline, this is carried forward as a candidate with its real cost, not claimed as a result here. It does not address BBH.
3. **Post-hoc observation, not a method.** The best scale of the update differs by task family. Shrinkage helps when CT-KV hurts, and full steps are needed when ICL is badly miscalibrated. Any rule that adapts the scale (for example, choosing it from a leave-one-out loss on the demonstrations, which needs no test labels) would have to be designed and validated on separate tasks or data. It must not be tuned on this frozen panel.
4. **If half-delta is revisited on BBH,** the interpolation should be computed and stored in float32 (or as an fp32 delta), because the bf16 cache cannot represent half of the BBH K displacement.
