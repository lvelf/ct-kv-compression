# Stage 2 plan (pre-registered before any Stage 2 accuracy was seen)

Guideline: `docs/experiment-guideline/CTKV_Stage2_Sparse_Attention_Guideline.md` (学习时间戳 2026-11-26 23:12:19 EST, pre-stamp sha256 04db2a28…). Written 2026-11-26, before any Stage 2 accuracy existed. The only runs so far are the r = 1 / mask checks in `checks/` and a two-job timing probe on the dev cell financial_phrasebank D s1000.

## No independent dev set

- **BBH Boolean:** the data has 250 rows, which are exactly the 50 support rows plus the 200 frozen test rows.
- **NLP-LR:** the MetaICL `*_test.jsonl` files are identical across splits 13/21/42/87/100. The candidate pool is the union of those splits' train files, and there are no dev files.

So no question exists that is disjoint from both the pool and the frozen test set. Following guideline §1, we therefore:

- **Fix a single main setting in advance:** exact dynamic top-k, r = 0.75, applied in both adaptation and prediction. The main contrast is sparse/sparse − dense/dense on arm D.
- **Treat the dev screen as exploration:** 3 tasks (BBH Boolean, financial_phrasebank, ai2_arc) × the first 3 manifest seeds (1000–1002).
- **Confirm on the remaining cells:** BBH seeds 1003–1019; NLP-LR seeds 1003–1009 of the two dev tasks plus all 10 seeds of the other 19 tasks. These are independent seeds and tasks.
- **Report the headline both ways:** on the full panel, and on the confirmation cells only.

Nothing in the selection rule, budget or matrix is changed after looking at test scores.

## Fixed inputs (verbatim from Stage 1)

- `results/ctkv_refinement/selection.jsonl`, `nlplr_jobs.jsonl` and `raw_bbh/jobs_bbh.jsonl`: demo IDs, order, prompts, optimizer seed 0, lr, dropout, schedule (BBH 16 steps; NLP-LR 200), scoring and common test IDs.
- Arm A dense is kept as the background reference, with Stage 1 ICL and CT-KV results.

## Implementation (`context-tuning/ctkv_sparse.py`, shared by both runners)

- **Wrapper:** `scaled_dot_product_attention` is wrapped, so the rule applies per layer × query head × query token, after RoPE / GQA `repeat_kv`, using the model's own mask and scaling.
- **Selection rule:** legal positions are the demonstration prefix positions the original mask allows (after leave-one-out and token dropout). Keep k = min(n, max(1, floor(r·n))) of them, then apply one joint softmax with the instruction and current-input positions. The ranking is not differentiated.
- **Dense cells:** the diagnostic path with r = 1 (float32 attention), so dense and sparse are compared in the same implementation. The production kernel is read as `prod` on every dense checkpoint, and Stage 1 production results are the external reference.

## Matrix (per task × seed)

- **D, dense adapt:** reads at step 0 and the endpoint: prod, dense, topk@.75, topk@.5, fixed@.75, random@.75.
- **D, topk75 / topk50 / fixed75 / rand75 adapt:** reads at the endpoint: dense and the matching sparse read. Step 0 is shared with the dense job, because the cache is identical before any update.
- **D, BBH only:** physical pruning (Ryan's answer_attention, `bbh_eval.ctkv._select`) at 75% and 50%, as the storage-cost check.
- **A:** dense adapt (prod, dense and topk@.75 reads) and topk75 adapt, for the Stage 1 × Stage 2 synergy contrast.
- **Random rule:** mask seed 0. The dev cells are also rerun with mask seeds 1 and 2.

## Analysis

- **Primary metric:** final accuracy, as a paired difference in points.
- **Bootstrap:** 5,000 paired draws. Seeds and the common test questions are resampled within each task, with indices shared across cells; tasks get equal weight; clf and mc are reported separately, as in Stage 1.
- **Also reported:** questions fixed vs broken; differences against ICL (step 0) and against the original CT-KV (A dense); the BBH label-share diagnostics; access ratio; cost.

## Decision rule

- **Quest-style paging:** move to page-level selection only if exact top-k is promising against dense and beats same-budget random.
- **Accuracy claim:** only if the paired interval is entirely above 0.
