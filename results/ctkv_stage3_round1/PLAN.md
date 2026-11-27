# Stage 3 round 1 plan: half-delta KV update and same-trajectory early stopping

Guideline: `docs/experiment-guideline/CTKV_Stage3_Round1_Guideline.md` (学习时间戳 2026-11-27 11:46:22 EST, pre-stamp sha256 d022d505…). Written before any Stage 3 accuracy existed; the only run so far is a debug run of the ai2_arc smoke job written to a scratch directory (not used as a result).

## Pre-registered main comparison

`half_delta − full`, reported separately for:

- **BBH Boolean** (the improvement target)
- **NLP-LR fixed-label classification**, 12 tasks (retention check)
- **NLP-LR per-question multiple choice**, 9 tasks (retention check)

The bootstrap is a paired accuracy difference in points: 5,000 seed × question draws, shared indices across conditions, tasks equal-weighted within each family.

Decision rules (guideline §5):

- **BBH:** improved if the lower bound of the `half_delta − full` interval is above 0.
- **NLP-LR, gain kept:** if the lower bound of each family's interval against `full` is above −1.0 point. If an interval crosses −1.0, the result is reported as uncertain.
- **Other contrasts:** `half_delta − icl`, `full − icl`, `early_half − full`, `early_quarter − full` and `early_half − half_delta`. Gradient-step costs are reported alongside them.
- **Post-hoc only:** per-seed best step picked with test labels (an upper bound, not a method).

## Conditions (one trajectory per task × demonstration seed)

| condition | cache | BBH step | NLP-LR step |
|---|---|---:|---:|
| icl | C_0 (detached clone right after prefill) | 0 | 0 |
| early_quarter | real C_{T/4}; the LR schedule still spans T | 4 | 50 |
| early_half | real C_{T/2} | 8 | 100 |
| full | C_T | 16 | 200 |
| half_delta | per layer K and V: `torch.lerp(C_0.float(), C_T.float(), 0.5).to(bf16)` | after 16 | after 200 |

## Inputs and code

- **Frozen Stage 1 arm-D jobs:** `results/ctkv_refinement/nlplr_jobs.jsonl` and `raw_bbh/jobs_bbh.jsonl` (`generic_bbh_` only). The demo IDs, order, prompts, raw loss, optimizer seed 0, LR, dropout, mask, scoring and positions are copied verbatim. BBH uses the manifest's 200 `eval_ids`; NLP-LR uses the common test IDs.
- **Panel:** 230 unique trajectories, 1,150 condition evaluations.
  - BBH: 20 trajectories.
  - NLP-LR: 210 trajectories (classification 120, multiple choice 90).
- **Attention path:** production dense throughout (SDPA; Stage 2's `prod`). The Stage 2 sparse wrapper is not installed.
- **NLP-LR runner:** `context-tuning/stage3_nlplr.py` (main repo commit 469f7c8). It is `study_nlplr.run_job` with the raw loss, plus the five conditions.
- **BBH runner:** `ctkv_study/stage3.py` in the worktree `/home/nuo/projects/ctkv-accuracy` (commit 9f71620). It is `ctkv_study.runner.run_job` with the raw loss, plus the five conditions.
- **Job lists:** built by `context-tuning/stage3_jobs.py`. Job IDs are `s3r1_<task>_D_s<seed>` with absolute output paths under `raw_{bbh,nlplr}/`.
- **Mid-trajectory evaluation:** saves and restores the python / numpy / torch CPU / CUDA RNG states and the train flag. For BBH it also asserts that the training generator is untouched. Evaluation runs on expanded copies, so neither C_0 nor the training cache can change (checked with an exact bit-pattern checksum).
- **Recorded per trajectory:**
  - predictions and correctness for all five conditions;
  - demo and eval IDs;
  - optimizer and scheduler step counts and configuration;
  - timing: prefill, cumulative adaptation time to each checkpoint excluding evaluation, interpolation, evaluation;
  - peak memory;
  - displacement geometry: norm ratio and cosine of C_t − C_0 against C_T − C_0, K and V separately.

## Smoke gate

The three smoke jobs are part of the 230: boolean_expressions, financial_phrasebank and ai2_arc at seed 1000. `context-tuning/stage3_check.py` must pass before the full panel starts. It checks:

- the interpolation path at weight 0 and 1 reproduces both endpoints bitwise, with identical predictions;
- the caches are unchanged by evaluation;
- question IDs are aligned and all five conditions are present;
- optimizer and scheduler step counts equal T;
- common-test coverage is complete;
- `icl` is identical to Stage 1 step 0;
- `full` matches the Stage 1 endpoint to within 5% flips.

BF16 backward is nondeterministic: an exact self-rerun of the ai2_arc job flipped 3 of 299 questions at `full`, and the loss already differed at step 2. Accuracy is never part of the gate.

## Queue

The Stage 3 steps are appended to `results/ctkv_stage2/queue/bench.txt`, after the Stage 2 serial benchmark (which keeps the GPU to itself), in this order:

1. `s3r1_smoke_bbh`
2. `s3r1_smoke_nlplr`
3. `s3r1_check`
4. `s3r1_bbh` (gated on the check)
5. `s3r1_nlplr` (gated on the check)

The runners skip existing outputs, write `.tmp` files, validate conditions and question counts, then rename atomically. Accuracy-run timings are taken with the GPU alone, because the queue runs serially.
