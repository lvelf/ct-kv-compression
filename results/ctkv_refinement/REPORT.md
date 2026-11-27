# CT-KV generic demonstration refinement — REPORT

Guideline: `docs/experiment-guideline/CTKV_Demonstration_Refinement_Guideline.md` (学习时间戳 2026-11-26 18:13:22 EST, sha256 23fc3c0d…). Runs: 2026-11-26 18:34–21:31 EST on one RTX 5090.

## Status

| item | status |
|---|---|
| Frozen selection (`selection.jsonl`, 740 rows) and `nlplr_jobs.jsonl` (660 jobs) | **done**, frozen before scoring (sha256 70834d0a…, 00346f65…) |
| BBH seed-0 history smoke check | **done**: ICL 173/200 = history; Full 165/200 vs history 163 and local original worker 164 (3 flips, BF16 GPU noise, see noise floor) |
| BBH Boolean, 4 arms × 20 seeds × {step 0, step 16} | **done**, 80/80 jobs, 200/200 questions per endpoint, 0 invalid |
| BBH run-to-run noise floor (A/D × 5 seeds × 2 reruns) | **done**, 20/20 |
| NLP-LR classification, 12 tasks × 4 arms × 10 seeds, steps 0/200 | **done**, 480/480 |
| NLP-LR multiple choice, 9 tasks × A/D × 10 seeds, steps 0/200 | **done**, 180/180 |
| Common test IDs | **done**: full test file on 20/21 tasks; glue-wnli common 51/71 (pairwise-intersection sensitivity reported) |
| Jobs with < 16 usable shots after the 1024-token limit | 3, all run and included: glue-wnli B_s1001 (8), C_s1001 (15), D_s1002 (12) |
| Failures / skipped jobs | **none** |
| Adaptation-time and memory breakdown | **done** (all 760 jobs profiled; overhead ≈1%) |
| §4 transfer to a new external dataset | **not done** this round (not run by choice; the selector entry point is generic and ready) |
| Not estimable | C−A, D−C, C−B on multiple-choice tasks (no fixed classes; C ≡ A by design, B undefined); BBH prefix-length scaling (prefix fixed at 180 tokens) |

## Headline (final accuracy, points; 95% paired bootstrap, 5,000 draws, seeds × questions)

| evaluation | tasks | **D − A** | **SD(D) − SD(A)** | C − A | D − C |
|---|---|---|---|---|---|
| BBH Boolean (Llama-3-8B-Instruct, 10-shot, 20 seeds) | 1 | +1.63 [−1.58, +5.08] | +0.57 [−1.69, +2.78] | +0.47 [−1.60, +2.53] | +1.15 [−1.65, +4.18] |
| NLP-LR classification (GPT-2 Large, 16-shot, 10 seeds) | 12 | +1.47 [−0.15, +3.16] | −1.17 [−2.53, +0.48] | +0.53 [−0.74, +1.84] | +0.93 [−0.55, +2.45] |
| NLP-LR multiple choice (GPT-2 Large, 16-shot, 10 seeds) | 9 | +0.16 [−0.75, +1.08] | +0.21 [−0.61, +0.77] | not estimable | not estimable |
| NLP-LR all 21, equal weight | 21 | +0.91 [−0.10, +1.96] | −0.58 [−1.45, +0.37] | — | — |

**Reading (guideline §4 rule: claim a gain only when the interval is above zero):**

- **No evaluation shows a significant accuracy gain for D over A.** All three D−A intervals cross zero. The point estimates are positive: BBH +1.6, classification +1.5, multiple choice +0.2.
- **NLP-LR classification is the closest to significant.** D−A is +1.47, but the full bootstrap interval reaches −0.15. The seed-only interval is [+0.27, +2.71]: given this fixed set of test questions, D beats A, but that does not carry over to new questions. Almost all of the gain comes from climate_fever (+8.6 [+3.7, +13.5], the only task significant on its own). SD falls by 1.17 points but the interval crosses zero, so no stability improvement can be claimed either.
- **Decomposition on fixed-label tasks:**
  - Quota (C−A) and content choice (D−C) are each small and positive, and neither is significant.
  - On classification, D's ICL score (step 0) is significantly higher than C's: +1.55 [+0.16, +3.10]. After 200 steps the final gap shrinks to +0.93. Content refinement mainly improves ICL, and gradient updates then wash part of that out.
- **Class-equal B behaves in opposite directions on the two benchmarks.**
  - On NLP-LR classification, B is far worse than C: C−B +10.87 [+8.51, +13.20]. Many tasks are heavily skewed, and forcing equal classes drags accuracy down.
  - On BBH, B is significantly better than C: C−B −2.10 [−4.23, −0.15]. B's final 80.70 is the highest of the four arms and its SD is the lowest.
  - The reason is that CT-KV makes BBH worse in every arm (gain A −2.65, C −2.03, D −1.25, B −0.25), and a 5/5 demo split suppresses the "predict True" drift during the update (B's True prediction ratio goes 0.638→0.603, while A goes 0.634→0.644).
  - This is a post-hoc observation. Under the guideline, the test results must not be used to change the policy.
- **Noise floor:** with identical inputs, BBH step-16 reruns differ by SD 1.2 questions (0.6 points), and step 0 reproduces exactly. That is much smaller than the ±3-point intervals above, so GPU nondeterminism does not drive any conclusion.

## Selection and adaptation time (medians per job)

| family | selection A/B/C (CPU, fit + select(1)) | selection D (fit + select(32)) | adaptation A | adaptation D | test eval |
|---|---|---|---|---|---|
| BBH (16 steps, Llama-3-8B) | 2.6 ms | 9.0 ms | 2.65 s | 2.65 s | 5.9 s |
| NLP-LR clf (200 steps, GPT-2 L) | 9.8 ms | 19.0 ms | 9.01 s | 9.06 s | 0.8 s |
| NLP-LR mc (200 steps, GPT-2 L) | 10.2 ms | 16.2 ms | 7.65 s | 8.71 s | 3.3 s |

D's extra selection cost is under 10 ms per task, at least 300× less than adaptation. In multiple choice, D's median adaptation is 14% above A's, but the means differ by only 3% (9.18 vs 8.93 s). Each update pads the full batch to the longest demo; D's median padded training tokens are 704 vs A's 624, while the real tokens are nearly identical (347 vs 350), and the mean prefix is 428 vs 418. In every setting adaptation splits into forward ≈ 41–44% and backward ≈ 52–53%; prefill, clip and the optimizer together take under 4%. Memory is dominated by model weights (BBH 15.0 GiB of a 15.4 GiB peak). The KV cache, its gradients and the Adam state together take only 4× the KV size (BBH 90 MiB; NLP-LR ≈ 0.3 GiB). On NLP-LR the peak is set by the forward transient memory (≈3.3–4.2 GiB, ≈8–10 MiB per prefix token); a further ≈1.1–1.5 GiB of "other resident" memory is the previous microbatch's output, which the code keeps referenced (explained below). Details are under "Performance" below.

## Versions and reproduction

- **Hardware and software:** NVIDIA GeForce RTX 5090 (driver 595.99.02), Python 3.10.22, torch 2.7.1+cu128 (CUDA 12.8), transformers 4.47.1, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, BF16, SDPA attention.
- **BBH model:** meta-llama/Meta-Llama-3-8B-Instruct @ `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`. The weights came from the NousResearch mirror @ 53346005. Every file was checked file by file against the official LFS sha256 and git blob hashes; once access was approved, the official endpoint etags were also confirmed to match (`model_mirror_verification.json` in the ctkv-accuracy worktree). Runs used `HF_HUB_OFFLINE=1`.
- **NLP-LR model:** openai-community/gpt2-large, with the MetaICL `*_16_87_test.jsonl` test files.
- **Code:**
  - main `158aca9` and later: `context-tuning/refinement/`, `study_nlplr.py`, `ctkv_perf.py`.
  - ctkv-accuracy worktree `exp/ctkv-accuracy` @ `4e9c810`: `ctkv_study/refine_jobs.py`, `runner.py`, `perf.py`.

```bash
MAIN_ROOT=/home/nuo/projects/ct-kv-compression; RUN_DIR=$MAIN_ROOT/results/ctkv_refinement
cd $MAIN_ROOT/context-tuning/refinement
python benchmark_plan.py --main-root $MAIN_ROOT --bbh-kit <BBH kit source/> --out $RUN_DIR   # -> selection.jsonl, nlplr_jobs.jsonl (appendix B verbatim)
(cd .. && python refinement/precheck.py $RUN_DIR)  # -> precheck.json, common_test_ids.json (CPU)
OMP_NUM_THREADS=1 python selection_timing.py --main-root $MAIN_ROOT --bbh-kit <bbh kit> --out $RUN_DIR
cd /home/nuo/projects/ctkv-accuracy               # BBH (worktree exp/ctkv-accuracy)
python -m ctkv_study.refine_jobs $RUN_DIR/selection.jsonl $RUN_DIR/raw_bbh   # -> jobs_bbh.jsonl (81 jobs incl. smoke_hist_s0)
HF_HUB_OFFLINE=1 python -m ctkv_study.runner $RUN_DIR/raw_bbh/jobs_bbh.jsonl
HF_HUB_OFFLINE=1 python -m ctkv_study.runner $RUN_DIR/raw_bbh_repeat/jobs_bbh_repeat.jsonl
cd $MAIN_ROOT/context-tuning                      # NLP-LR
python study_nlplr.py $RUN_DIR/nlplr_jobs.jsonl
python refinement/analyze.py $RUN_DIR             # -> runs.csv, contrasts.csv, ..., REPORT_results.md
```

The exact command lines and timings of the serial queue are in `queue/plan.txt` and `queue/plan.txt.log`. Per-run raw JSON (raw_bbh/, raw_bbh_repeat/, raw_nlplr/) is packed in `raw_runs.tgz`; per-question predictions are in `predictions.jsonl.gz`. `common_test_ids.json` canonical sha256 (sort_keys) 640e20f9…, written 18:35 before any scoring.

---

# Detailed results (generated by `refinement/analyze.py`)

Guideline: `docs/experiment-guideline/CTKV_Demonstration_Refinement_Guideline.md` (学习时间戳 2026-11-26 18:13:22 EST). Arms: A random, B class-equal, C pool-ratio quota, D refinement (min feature distance over 32 quota-respecting candidates; first candidate = C, or A without fixed classes). All numbers are accuracy in points on the frozen common test IDs; brackets = 95% paired bootstrap (5,000 draws, seeds × questions).

## Headline

| evaluation | tasks | D − A final | SD(D) − SD(A) | C − A final | D − C final |
|---|---|---|---|---|---|
| BBH Boolean (Llama-3-8B-Instruct, 10-shot, 20 seeds) | 1 | +1.63 [-1.58, +5.08] | +0.57 [-1.69, +2.78] | +0.47 [-1.60, +2.53] | +1.15 [-1.65, +4.18] |
| NLP-LR fixed-label classification (GPT-2 Large, 16-shot, 10 seeds) | 12 | +1.47 [-0.15, +3.16] | -1.17 [-2.53, +0.48] | +0.53 [-0.74, +1.84] | +0.93 [-0.55, +2.45] |
| NLP-LR per-question multiple choice (GPT-2 Large, 16-shot, 10 seeds) | 9 | +0.16 [-0.75, +1.08] | +0.21 [-0.61, +0.77] | — | — |
| NLP-LR all 21 tasks, equal weight | 21 | +0.91 [-0.10, +1.96] | -0.58 [-1.45, +0.37] | — | — |

Reading rule (guideline §4): a gain is claimed only if the interval lies above zero; when SD drops while the mean is flat, only a stability improvement is reported.

## BBH Boolean (Llama-3-8B-Instruct, 10-shot, 20 seeds)

### Mean over tasks (seed-mean first, tasks equal weight)

| contrast | final | gain (final − step 0) | ICL (step 0) | SD diff |
|---|---|---|---|---|
| D − A | +1.63 [-1.58, +5.08] (seed-only [-1.10, +4.37]) | +1.40 [-2.50, +5.40] | +0.23 [-1.60, +2.13] | +0.57 [-1.69, +2.78] |
| C − A | +0.47 [-1.60, +2.53] (seed-only [-1.12, +2.10]) | +0.62 [-1.85, +3.15] | -0.15 [-1.52, +1.23] | -0.74 [-2.74, +1.17] |
| D − C | +1.15 [-1.65, +4.18] (seed-only [-1.03, +3.30]) | +0.77 [-2.25, +4.15] | +0.37 [-1.33, +2.08] | +1.32 [-0.80, +3.28] |
| C − B | -2.10 [-4.23, -0.15] (seed-only [-3.45, -0.60]) | -1.77 [-4.15, +0.40] | -0.33 [-1.67, +1.05] | -0.09 [-1.65, +1.60] |

### Per arm (mean over tasks of seed-mean accuracy; SD = mean over tasks of across-seed SD)

| arm | step 0 | final | gain | across-seed SD (final) | adaptation s (median) |
|---|---|---|---|---|---|
| A | 80.78 | 78.12 | -2.65 | 4.19 | 2.6 |
| B | 80.95 | 80.70 | -0.25 | 3.53 | 2.6 |
| C | 80.62 | 78.60 | -2.03 | 3.44 | 2.6 |
| D | 81.00 | 79.75 | -1.25 | 4.76 | 2.6 |

## NLP-LR fixed-label classification (GPT-2 Large, 16-shot, 10 seeds)

### Mean over tasks (seed-mean first, tasks equal weight)

| contrast | final | gain (final − step 0) | ICL (step 0) | SD diff |
|---|---|---|---|---|
| D − A | +1.47 [-0.15, +3.16] (seed-only [+0.27, +2.71]) | +0.71 [-1.47, +2.80] | +0.76 [-0.79, +2.38] | -1.17 [-2.53, +0.48] |
| C − A | +0.53 [-0.74, +1.84] (seed-only [-0.47, +1.55]) | +1.32 [-0.21, +2.94] | -0.79 [-1.92, +0.19] | -1.12 [-2.46, +0.44] |
| D − C | +0.93 [-0.55, +2.45] (seed-only [-0.09, +1.98]) | -0.61 [-2.70, +1.47] | +1.55 [+0.16, +3.10] | -0.05 [-1.22, +1.28] |
| C − B | +10.87 [+8.51, +13.20] (seed-only [+9.71, +12.04]) | +7.29 [+4.99, +9.57] | +3.59 [+1.98, +5.40] | -3.30 [-4.80, -1.97] |

### Per task (D − A)

| task | seeds | common Q | final | gain | ICL | SD diff |
|---|---|---|---|---|---|---|
| climate_fever | 10 | 307 | +8.60 [+3.68, +13.49] | +7.07 [+1.21, +12.80] | +1.53 [-3.29, +6.84] | -0.53 [-3.74, +2.59] |
| ethos-national_origin | 10 | 87 | +0.92 [-3.45, +6.21] | +0.69 [-3.79, +5.98] | +0.23 [-0.69, +1.49] | -3.01 [-8.18, +2.72] |
| ethos-race | 10 | 87 | +2.64 [-0.11, +6.90] | +2.64 [-0.11, +6.90] | +0.00 [+0.00, +0.00] | -4.16 [-8.12, -0.48] |
| ethos-religion | 10 | 87 | +2.76 [-1.84, +7.59] | +2.18 [-1.61, +6.32] | +0.57 [-1.15, +2.64] | -2.88 [-6.64, +1.82] |
| financial_phrasebank | 10 | 970 | +0.43 [-4.62, +5.74] | +7.68 [-0.87, +17.78] | -7.25 [-17.48, +1.01] | +0.08 [-3.27, +2.98] |
| glue-wnli | 10 | 51 | +5.29 [-3.14, +13.14] | +5.49 [-2.75, +13.33] | -0.20 [-1.18, +0.00] | +0.19 [-8.01, +6.63] |
| hate_speech18 | 10 | 2141 | -0.02 [-0.45, +0.32] | -0.02 [-0.45, +0.32] | +0.00 [+0.00, +0.00] | +0.15 [-0.32, +0.57] |
| poem_sentiment | 10 | 105 | -2.00 [-8.00, +4.10] | -4.86 [-10.95, +0.57] | +2.86 [-0.76, +9.33] | +0.47 [-5.13, +6.78] |
| sick | 10 | 495 | -0.12 [-4.57, +4.32] | -0.08 [-4.55, +4.40] | -0.04 [-0.16, +0.00] | +1.80 [-2.70, +5.35] |
| tweet_eval-hate | 10 | 999 | -0.19 [-3.39, +2.78] | -0.07 [-3.35, +2.97] | -0.12 [-0.57, +0.17] | -1.50 [-3.66, +1.35] |
| tweet_eval-stance_atheism | 10 | 52 | +0.96 [-8.85, +11.73] | -11.15 [-29.42, +4.81] | +12.12 [-0.19, +27.70] | -6.10 [-14.23, +4.69] |
| tweet_eval-stance_feminist | 10 | 67 | -1.64 [-9.10, +5.38] | -1.04 [-8.21, +5.82] | -0.60 [-4.18, +2.39] | +1.44 [-5.54, +7.23] |

### Per arm (mean over tasks of seed-mean accuracy; SD = mean over tasks of across-seed SD)

| arm | step 0 | final | gain | across-seed SD (final) | adaptation s (median) |
|---|---|---|---|---|---|
| A | 57.17 | 61.86 | +4.69 | 5.13 | 9.0 |
| B | 52.80 | 51.52 | -1.28 | 7.31 | 9.2 |
| C | 56.39 | 62.40 | +6.01 | 4.01 | 9.0 |
| D | 57.93 | 63.33 | +5.40 | 3.96 | 9.1 |

## NLP-LR per-question multiple choice (GPT-2 Large, 16-shot, 10 seeds)

### Mean over tasks (seed-mean first, tasks equal weight)

| contrast | final | gain (final − step 0) | ICL (step 0) | SD diff |
|---|---|---|---|---|
| D − A | +0.16 [-0.75, +1.08] (seed-only [-0.40, +0.74]) | +0.10 [-0.92, +1.13] | +0.06 [-0.57, +0.69] | +0.21 [-0.61, +0.77] |

### Per task (D − A)

| task | seeds | common Q | final | gain | ICL | SD diff |
|---|---|---|---|---|---|---|
| ai2_arc | 10 | 299 | +0.10 [-2.58, +2.61] | -0.23 [-3.11, +2.58] | +0.33 [-1.30, +1.91] | +0.63 [-1.48, +2.38] |
| codah | 10 | 556 | -0.49 [-3.08, +2.19] | -0.09 [-2.77, +2.52] | -0.40 [-1.69, +0.90] | +0.52 [-1.25, +2.12] |
| commonsense_qa | 10 | 1221 | +0.69 [-0.97, +2.32] | +0.51 [-1.72, +2.76] | +0.18 [-1.48, +1.80] | -0.17 [-1.53, +1.21] |
| openbookqa | 10 | 500 | +0.42 [-1.84, +2.74] | +1.06 [-1.62, +3.84] | -0.64 [-1.82, +0.40] | -0.30 [-1.71, +1.24] |
| qasc | 10 | 926 | -1.37 [-4.24, +1.31] | -1.18 [-3.84, +1.16] | -0.19 [-2.35, +1.92] | -1.20 [-2.97, +1.03] |
| quarel | 10 | 278 | +0.36 [-2.34, +3.13] | +0.07 [-2.88, +3.24] | +0.29 [-2.09, +2.66] | +0.36 [-1.77, +2.31] |
| quartz-no_knowledge | 10 | 384 | -0.68 [-3.67, +2.37] | -0.76 [-4.11, +2.66] | +0.08 [-2.06, +2.32] | -0.06 [-2.17, +2.12] |
| quartz-with_knowledge | 10 | 384 | +0.94 [-1.77, +3.78] | +0.73 [-2.19, +3.83] | +0.21 [-1.54, +1.90] | +1.22 [-1.47, +2.89] |
| superglue-copa | 10 | 100 | +1.50 [-2.20, +5.50] | +0.80 [-3.70, +5.60] | +0.70 [-2.00, +3.50] | +0.90 [-2.72, +3.61] |

### Per arm (mean over tasks of seed-mean accuracy; SD = mean over tasks of across-seed SD)

| arm | step 0 | final | gain | across-seed SD (final) | adaptation s (median) |
|---|---|---|---|---|---|
| A | 40.54 | 44.78 | +4.25 | 1.81 | 7.7 |
| D | 40.60 | 44.95 | +4.35 | 2.02 | 8.7 |

## BBH hypothesis checks (mean over 20 seeds)

| arm | demo True | pred True @0 | pred True @16 | recall T @0/@16 | recall F @0/@16 | invalid @0/@16 | fixed 0→16 | broken 0→16 |
|---|---|---|---|---|---|---|---|---|
| A | 0.59 | 0.634 | 0.644 | 0.917/0.901 | 0.685/0.646 | 0.00/0.00 | 10.8 | 16.1 |
| B | 0.50 | 0.638 | 0.603 | 0.923/0.887 | 0.682/0.717 | 0.00/0.00 | 10.6 | 11.1 |
| C | 0.60 | 0.631 | 0.645 | 0.913/0.907 | 0.686/0.650 | 0.00/0.00 | 10.0 | 14.1 |
| D | 0.60 | 0.621 | 0.607 | 0.906/0.882 | 0.702/0.702 | 0.00/0.00 | 11.2 | 13.7 |

Per-seed scores: `bbh_checks.csv` (acc_0, acc_16 per arm × seed). Prediction-True denominators are all 200 questions.

| seed | A @0/@16 | B @0/@16 | C @0/@16 | D @0/@16 |
|---|---|---|---|---|
| 1000 | 77.5/78.0 | 80.0/73.5 | 79.5/72.5 | 81.5/70.0 |
| 1001 | 75.0/79.5 | 75.0/79.5 | 78.0/79.5 | 77.5/72.5 |
| 1002 | 77.0/72.5 | 81.5/79.5 | 77.0/72.5 | 81.0/77.5 |
| 1003 | 84.0/81.5 | 84.0/84.0 | 83.0/78.0 | 82.5/86.5 |
| 1004 | 79.0/78.5 | 79.0/83.5 | 79.0/82.5 | 81.0/75.0 |
| 1005 | 87.0/81.5 | 85.5/82.0 | 83.5/85.5 | 85.0/84.0 |
| 1006 | 81.5/80.5 | 80.0/79.5 | 80.5/77.5 | 77.5/75.0 |
| 1007 | 81.5/83.0 | 80.5/80.5 | 80.5/79.0 | 80.0/81.0 |
| 1008 | 87.5/79.5 | 86.0/86.0 | 84.0/82.0 | 83.0/79.5 |
| 1009 | 82.5/75.5 | 82.5/75.5 | 82.5/80.0 | 84.0/82.0 |
| 1010 | 81.5/69.5 | 78.0/77.5 | 80.5/77.0 | 77.0/82.0 |
| 1011 | 81.0/82.0 | 81.0/83.0 | 79.0/79.0 | 80.5/84.0 |
| 1012 | 80.0/83.5 | 82.5/83.5 | 80.5/81.0 | 83.5/85.0 |
| 1013 | 77.0/78.0 | 79.5/78.5 | 79.5/81.5 | 76.5/79.0 |
| 1014 | 77.5/78.0 | 78.5/80.0 | 77.5/78.5 | 80.5/79.0 |
| 1015 | 78.5/72.0 | 84.0/85.0 | 84.0/77.0 | 84.0/77.5 |
| 1016 | 83.0/76.5 | 80.0/81.0 | 78.5/80.0 | 84.0/84.0 |
| 1017 | 79.5/83.5 | 79.5/83.0 | 81.0/79.0 | 80.0/73.0 |
| 1018 | 84.0/71.0 | 80.0/74.5 | 83.5/71.5 | 80.5/83.5 |
| 1019 | 81.0/78.5 | 82.0/84.5 | 81.0/78.5 | 80.5/85.0 |

## NLP-LR classification label checks (mean over tasks × seeds)

| arm | demo TV to pool | pred TV to pool @0 / @200 | pred TV to test gold @0 / @200 | mean per-class recall @0 / @200 | macro-F1 @0 / @200 |
|---|---|---|---|---|---|
| A | 0.086 | 0.362 / 0.186 | 0.384 / 0.209 | 0.417 / 0.458 | 32.37 / 42.44 |
| B | 0.245 | 0.363 / 0.276 | 0.373 / 0.248 | 0.434 / 0.522 | 34.36 / 46.15 |
| C | 0.024 | 0.368 / 0.162 | 0.390 / 0.191 | 0.417 / 0.463 | 32.13 / 43.22 |
| D | 0.024 | 0.358 / 0.151 | 0.374 / 0.177 | 0.418 / 0.472 | 32.55 / 44.55 |

Per-class recall per task × arm × seed: `nlplr_class_checks.csv`. A prediction mix closer to the pool is not by itself counted as more correct answers.

## Coverage

| task | family | jobs ok | test file | evaluable union | common | coverage |
|---|---|---|---|---|---|---|
| ai2_arc | mc | 20/20 | 299 | 299 | 299 | 1.00 |
| climate_fever | clf | 40/40 | 307 | 307 | 307 | 1.00 |
| codah | mc | 20/20 | 556 | 556 | 556 | 1.00 |
| commonsense_qa | mc | 20/20 | 1221 | 1221 | 1221 | 1.00 |
| ethos-national_origin | clf | 40/40 | 87 | 87 | 87 | 1.00 |
| ethos-race | clf | 40/40 | 87 | 87 | 87 | 1.00 |
| ethos-religion | clf | 40/40 | 87 | 87 | 87 | 1.00 |
| financial_phrasebank | clf | 40/40 | 970 | 970 | 970 | 1.00 |
| glue-wnli | clf | 40/40 | 71 | 71 | 51 | 0.72 |
| hate_speech18 | clf | 40/40 | 2141 | 2141 | 2141 | 1.00 |
| openbookqa | mc | 20/20 | 500 | 500 | 500 | 1.00 |
| poem_sentiment | clf | 40/40 | 105 | 105 | 105 | 1.00 |
| qasc | mc | 20/20 | 926 | 926 | 926 | 1.00 |
| quarel | mc | 20/20 | 278 | 278 | 278 | 1.00 |
| quartz-no_knowledge | mc | 20/20 | 384 | 384 | 384 | 1.00 |
| quartz-with_knowledge | mc | 20/20 | 384 | 384 | 384 | 1.00 |
| sick | clf | 40/40 | 495 | 495 | 495 | 1.00 |
| superglue-copa | mc | 20/20 | 100 | 100 | 100 | 1.00 |
| tweet_eval-hate | clf | 40/40 | 999 | 999 | 999 | 1.00 |
| tweet_eval-stance_atheism | clf | 40/40 | 52 | 52 | 52 | 1.00 |
| tweet_eval-stance_feminist | clf | 40/40 | 67 | 67 | 67 | 1.00 |

### Sensitivity: pairwise intersections (D − A, NLP-LR)

| task | Q pair | Q global | final (pair) | final (global) |
|---|---|---|---|---|
| climate_fever | 307 | 307 | +8.60 | +8.60 |
| ethos-national_origin | 87 | 87 | +0.92 | +0.92 |
| ethos-race | 87 | 87 | +2.64 | +2.64 |
| ethos-religion | 87 | 87 | +2.76 | +2.76 |
| financial_phrasebank | 970 | 970 | +0.43 | +0.43 |
| glue-wnli | 63 | 51 | +2.70 | +5.29 |
| hate_speech18 | 2141 | 2141 | -0.02 | -0.02 |
| poem_sentiment | 105 | 105 | -2.00 | -2.00 |
| sick | 495 | 495 | -0.12 | -0.12 |
| tweet_eval-hate | 999 | 999 | -0.19 | -0.19 |
| tweet_eval-stance_atheism | 52 | 52 | +0.96 | +0.96 |
| tweet_eval-stance_feminist | 67 | 67 | -1.64 | -1.64 |
| ai2_arc | 299 | 299 | +0.10 | +0.10 |
| codah | 556 | 556 | -0.49 | -0.49 |
| commonsense_qa | 1221 | 1221 | +0.69 | +0.69 |
| openbookqa | 500 | 500 | +0.42 | +0.42 |
| qasc | 926 | 926 | -1.37 | -1.37 |
| quarel | 278 | 278 | +0.36 | +0.36 |
| quartz-no_knowledge | 384 | 384 | -0.68 | -0.68 |
| quartz-with_knowledge | 384 | 384 | +0.94 | +0.94 |
| superglue-copa | 100 | 100 | +1.50 | +1.50 |

## BBH run-to-run noise floor (same demos, same code, reruns)

Each of seeds 1000–1004 × arms A/D was rerun twice with identical inputs. Differences come only from non-deterministic BF16 CUDA kernels (near-tie logits). Δ = rerun − main run (questions correct out of 200); flips = questions whose prediction differs from the main run.

| job | main @0/@16 | rep1 @0/@16 | rep2 @0/@16 | flips rep1 @0/@16 | flips rep2 @0/@16 |
|---|---|---|---|---|---|
| A_s1000 | 155/156 | 155/156 | 155/156 | 0/0 | 0/2 |
| A_s1001 | 150/159 | 150/158 | 150/160 | 0/1 | 0/1 |
| A_s1002 | 154/145 | 154/146 | 154/148 | 0/1 | 0/3 |
| A_s1003 | 168/163 | 168/161 | 168/162 | 0/2 | 0/1 |
| A_s1004 | 158/157 | 158/156 | 158/156 | 0/1 | 0/1 |
| D_s1000 | 163/140 | 163/139 | 163/140 | 0/1 | 0/0 |
| D_s1001 | 155/145 | 155/146 | 155/147 | 0/1 | 0/2 |
| D_s1002 | 162/155 | 162/154 | 162/154 | 0/1 | 0/1 |
| D_s1003 | 165/173 | 165/172 | 165/173 | 0/1 | 0/0 |
| D_s1004 | 162/150 | 162/150 | 162/150 | 0/0 | 0/0 |

Step 0: |Δ| mean 0.00 questions (max 0), Δ SD 0.00 questions = 0.00 points; prediction flips mean 0.00 (max 0).

Step 16: |Δ| mean 0.90 questions (max 3), Δ SD 1.21 questions = 0.60 points; prediction flips mean 1.00 (max 3).

### Seed-0 smoke check (historical demos, manifest seed 0)

| step | history | original worker (local rerun) | this runner | prediction flips vs original worker |
|---|---|---|---|---|
| 0 | 173/200 | 174/200 | 173/200 | 1 |
| 16 | 163/200 | 164/200 | 165/200 | 3 |

## Performance: adaptation-time and memory breakdown

Adaptation = from the start of the prefix forward to a tuned, servable KV cache (test evaluation excluded, reported separately). Each phase is bracketed by `torch.cuda.synchronize()`; the synchronizations are inside the measured time (overhead ≈1% vs. un-instrumented runs). Medians over jobs; memory in MiB of `torch.cuda.memory_allocated` (reserved = caching-allocator peak). Per-job values: `perf.csv`.

Memory columns: *resident in training* = allocated at the start of a forward microbatch (max over steps); *other resident* = that minus weights, KV, KV grads and Adam state. In the NLP-LR runner the previous microbatch's model output stays referenced until the next forward reassigns it; with `use_cache=True` it holds the batch-expanded concatenated KV (36 layers × batch × (prefix + sequence) × 0.176 MiB/token) plus logits; this arithmetic estimate (not measured separately) matches the ≈1.1–1.5 GiB. The BBH runner does not keep it (≈16 MiB). Transient = phase peak − allocated at phase start (activations and workspaces).

### BBH Boolean (Llama-3-8B-Instruct, 10-shot, 20 seeds)

**Time** (seconds per job, median)

| arm | jobs | prefill | prep | forward | backward | clip | optimizer | other | adaptation | step 1 | steady step | eval (not in adaptation) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 20 | 0.027 | 0.000 | 1.172 | 1.381 | 0.003 | 0.023 | 0.044 | 2.65 | 0.163 | 0.164 | 5.85 |
| B | 20 | 0.027 | 0.000 | 1.171 | 1.380 | 0.003 | 0.023 | 0.044 | 2.65 | 0.163 | 0.164 | 5.85 |
| C | 20 | 0.026 | 0.000 | 1.171 | 1.381 | 0.003 | 0.023 | 0.044 | 2.65 | 0.163 | 0.164 | 5.85 |
| D | 20 | 0.026 | 0.000 | 1.172 | 1.382 | 0.003 | 0.023 | 0.044 | 2.65 | 0.163 | 0.164 | 5.85 |

Share of adaptation time (all arms): prefill 1.0%, forward 44.2%, backward 52.1%, clip 0.1%, optimizer 0.9%, other 1.7%.

**Memory** (MiB, median)

| arm | weights | KV cache | KV grads | Adam state | other resident | resident in training | prefill transient | forward transient | backward transient | peak | reserved peak | eval peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 15316.5 | 22.5 | 22.5 | 45.0 | 16.3 | 15422.8 | 43.0 | 391.9 | 16.6 | 15814.7 | 15870.0 | 15451.7 |
| B | 15316.5 | 22.5 | 22.5 | 45.0 | 16.3 | 15422.8 | 43.0 | 391.9 | 16.6 | 15814.7 | 15870.0 | 15451.7 |
| C | 15316.5 | 22.5 | 22.5 | 45.0 | 16.3 | 15422.8 | 43.0 | 391.9 | 16.6 | 15814.7 | 15870.0 | 15451.7 |
| D | 15316.5 | 22.5 | 22.5 | 45.0 | 16.3 | 15422.8 | 43.0 | 391.9 | 16.6 | 15814.7 | 15870.0 | 15451.7 |

Prefix length: median 180 tokens (range 180–180). Linear fits: not estimable (prefix length is the same in every job).

### NLP-LR fixed-label classification (GPT-2 Large, 16-shot, 10 seeds)

**Time** (seconds per job, median)

| arm | jobs | prefill | prep | forward | backward | clip | optimizer | other | adaptation | step 1 | steady step | eval (not in adaptation) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 120 | 0.010 | 0.045 | 3.734 | 4.798 | 0.056 | 0.226 | 0.230 | 9.01 | 0.047 | 0.045 | 0.80 |
| B | 120 | 0.010 | 0.045 | 3.756 | 4.882 | 0.056 | 0.224 | 0.230 | 9.16 | 0.049 | 0.045 | 0.81 |
| C | 120 | 0.010 | 0.045 | 3.725 | 4.816 | 0.056 | 0.226 | 0.230 | 9.02 | 0.047 | 0.045 | 0.80 |
| D | 120 | 0.010 | 0.045 | 3.760 | 4.823 | 0.056 | 0.225 | 0.230 | 9.06 | 0.048 | 0.045 | 0.80 |

Share of adaptation time (all arms): prefill 0.1%, prep 0.5%, forward 41.0%, backward 52.8%, clip 0.6%, optimizer 2.5%, other 2.5%.

**Memory** (MiB, median)

| arm | weights | KV cache | KV grads | Adam state | other resident | resident in training | prefill transient | forward transient | backward transient | peak | reserved peak | eval peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 1512.3 | 80.8 | 80.8 | 161.5 | 1462.2 | 3297.9 | 220.1 | 4213.9 | 148.0 | 7601.6 | 8766.0 | 6846.6 |
| B | 1512.3 | 81.8 | 81.8 | 163.7 | 1497.6 | 3335.5 | 221.9 | 4341.9 | 150.3 | 7843.9 | 8902.0 | 6968.2 |
| C | 1512.3 | 80.4 | 80.4 | 160.8 | 1484.6 | 3322.5 | 219.7 | 4236.4 | 148.0 | 7660.2 | 8653.0 | 6850.1 |
| D | 1512.3 | 80.4 | 80.4 | 160.8 | 1497.4 | 3336.2 | 219.9 | 4316.1 | 148.0 | 7596.5 | 8743.0 | 6956.9 |

Prefix length: median 460 tokens (range 201–968). Linear fits over jobs: adaptation_s: 16.526 ms/prefix token (intercept 2204.2 ms, r=0.83); m_kv_cache_mib: 0.176 MiB/prefix token (intercept 0.0 MiB, r=1.00); m_forward_transient_mib: 8.202 MiB/prefix token (intercept 750.2 MiB, r=0.83); m_peak_mib: 11.820 MiB/prefix token (intercept 2432.9 MiB, r=0.90).

### NLP-LR per-question multiple choice (GPT-2 Large, 16-shot, 10 seeds)

**Time** (seconds per job, median)

| arm | jobs | prefill | prep | forward | backward | clip | optimizer | other | adaptation | step 1 | steady step | eval (not in adaptation) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 90 | 0.009 | 0.045 | 3.264 | 4.007 | 0.050 | 0.148 | 0.230 | 7.65 | 0.041 | 0.038 | 3.28 |
| D | 90 | 0.009 | 0.045 | 3.524 | 4.628 | 0.050 | 0.147 | 0.229 | 8.71 | 0.045 | 0.043 | 3.29 |

Share of adaptation time (all arms): prefill 0.1%, prep 0.5%, forward 41.4%, backward 52.7%, clip 0.6%, optimizer 1.8%, other 2.8%.

**Memory** (MiB, median)

| arm | weights | KV cache | KV grads | Adam state | other resident | resident in training | prefill transient | forward transient | backward transient | peak | reserved peak | eval peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 1512.3 | 61.3 | 61.3 | 122.7 | 1134.4 | 2898.2 | 155.2 | 3423.1 | 116.6 | 6375.9 | 7484.0 | 5377.3 |
| D | 1512.3 | 60.7 | 60.7 | 121.5 | 1126.1 | 2880.2 | 154.4 | 3971.8 | 132.0 | 6954.6 | 7850.0 | 5348.9 |

Prefix length: median 347 tokens (range 215–810). Linear fits over jobs: adaptation_s: 21.059 ms/prefix token (intercept 138.1 ms, r=0.94); m_kv_cache_mib: 0.176 MiB/prefix token (intercept 0.0 MiB, r=1.00); m_forward_transient_mib: 10.107 MiB/prefix token (intercept -186.6 MiB, r=0.94); m_peak_mib: 13.980 MiB/prefix token (intercept 1329.0 MiB, r=0.96).

### Selection cost (CPU, refine.py)

fit = TF-IDF + length buckets + class centres on the pool; select(1) = one candidate (the cost of A/B/C); select(32) − select(1) = D's extra search. Median over seeds, milliseconds.

| family | tasks | pool size (median) | fit | select(1) | select(32) | D extra |
|---|---|---|---|---|---|---|
| bbh | 1 | 50 | 1.7 | 0.9 | 7.3 | 6.4 |
| clf | 12 | 77 | 8.6 | 1.2 | 10.4 | 9.1 |
| mc | 9 | 79 | 9.7 | 0.5 | 6.5 | 6.0 |

