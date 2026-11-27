# CT-KV Stage 2: per-query sparse reading of demonstration KV — report

Guideline: `docs/experiment-guideline/CTKV_Stage2_Sparse_Attention_Guideline.md` (学习时间戳 2026-11-26 23:12:19 EST). Plan: `PLAN.md` (pre-registered). All numbers and tables: `REPORT_results.md`, `contrasts.csv`, `cells.csv`, `bbh_label_checks.csv`, `cost.csv`, `predictions.jsonl.gz`; raw per-job outputs in `raw_archive.tar.xz`.

## Status

- **Complete.** BBH Boolean: 20 seeds × all cells (186 jobs, Llama-3-8B-Instruct, 10-shot, 16 steps). NLP-LR: 21 tasks × 10 seeds × all cells (1,482 jobs, GPT-2 Large, 16-shot, 200 steps). Random-rule dev cells were rerun with mask seeds 1 and 2.
- **Implementation checks pass.** The diagnostic dense path at r = 1 against Stage 1 production: BBH +0.12, clf +0.06, mc −0.01 points (all intervals cover 0). Production read vs diagnostic read of the same dense checkpoint: within ±0.10. At step 0, the dense read reproduces Stage 1 ICL within ±0.05.
- **Shared-GPU incidents.** Running both queues together caused OOMs (I added a job-level OOM retry, then serialised the queues). There was also one transient Xid 8 GPU watchdog timeout; the queue retried automatically. No job was dropped. The timings in the accuracy runs are affected by the sharing; the serial benchmark (`bench/`) is the speed reference.

## Main comparison: arm D, exact dynamic top-k 75%, sparse adapt and sparse read − dense/dense

| evaluation | dense/dense | top-k/top-k | diff [95% CI] | confirmation cells only | fixed / broken Q |
|---|---|---|---|---|---|
| BBH Boolean | 79.88 | 79.97 | +0.10 [−0.53, +0.78] | +0.24 [−0.47, +1.00] | 30 / 26 |
| NLP-LR classification (12 tasks) | 63.39 | 63.29 | −0.10 [−0.50, +0.30] | −0.05 [−0.45, +0.33] | 367 / 504 |
| NLP-LR multiple choice (9 tasks) | 44.94 | 45.01 | +0.08 [−0.18, +0.34] | +0.09 [−0.17, +0.36] | 469 / 413 |

**Decision (pre-registered rule: claim a gain only if the whole interval is above 0):** no accuracy gain on any family. Sparse reading of 75% of the demonstration positions is **accuracy-neutral**: every interval lies within about ±0.8 points, and the confirmation cells agree with the full panel. No task stands out. The largest per-task difference is financial_phrasebank (−0.99 [−2.24, +0.01]), which is an exploration cell.

## Where the effect would sit: adapt vs read (75%)

| contrast | BBH | clf | mc |
|---|---|---|---|
| dense adapt, top-k read − dense/dense | +0.10 | +0.19 | −0.03 |
| top-k adapt, dense read − dense/dense | −0.10 | −0.25 | +0.04 |
| top-k adapt, top-k read − top-k adapt, dense read | +0.20 | +0.14 | +0.04 |

None of these intervals excludes 0. Sparsity in adaptation does not damage the cache, and sparsity in prediction neither helps nor hurts. **At 50%**, the only interval that excludes 0 is NLP-LR classification with top-k adaptation and a dense read: −0.82 [−1.58, −0.06]. With the matching 50% read it recovers to −0.15. This means the cache learned under a 50% budget is slightly specialised to being read sparsely.

## Controls

| contrast (75%, sparse/sparse) | BBH | clf | mc |
|---|---|---|---|
| dynamic top-k − random | +0.38 [−1.08, +1.82] | +0.30 [−0.94, +1.52] | +0.32 [−0.53, +1.20] |
| dynamic top-k − fixed answer_attention | −0.88 [−2.45, +0.70] | −0.47 [−1.25, +0.32] | +0.08 [−0.33, +0.52] |
| physical prune 75% (Ryan) − Stage 1 D CT-KV | +0.83 [−1.02, +2.78] | — | — |
| physical prune 50% (Ryan) − Stage 1 D CT-KV | +1.08 [−1.07, +3.30] | — | — |

- **Dynamic top-k does not beat a same-budget random mask.** The point estimates are about +0.3 to +0.4. The random rule is noisy, though: across mask seeds, the dev cells span up to 4.5 points on BBH.
- **The fixed answer_attention selection has the highest point estimates on BBH.**
  - Step 0: +1.00 vs dense.
  - After adaptation: +0.97 vs dense/dense.
  - Ryan's physical pruning at 50%: +1.08.

  None of these is significant on 20 seeds. In the label diagnostics, the fixed selection keeps a slightly larger share of True-labelled demonstration tokens (0.627 vs a legal share of 0.600). It lowers predicted-True (0.597 vs 0.602) and raises False recall (0.724 vs 0.710).
- **Step 0 (no update, read only):** for classification, top-k 75% +0.26 [+0.06, +0.52], top-k 50% +0.44 [+0.06, +0.87] and fixed 75% +0.83 [+0.30, +1.41] vs the dense read. These are small, real ICL-time gains, and adaptation erases them.

## Against zero-update and original CT-KV

- **NLP-LR:** the main sparse/sparse cell keeps all of the CT-KV gain over ICL. Classification is +5.36 [+3.44, +7.19] and multiple choice +4.44 [+3.10, +5.73] vs D ICL. It is equal to Stage 1 D CT-KV (−0.05 and +0.07).
- **BBH:** CT-KV still lowers accuracy relative to ICL (main cell −1.08 [−3.90, +1.80] vs D ICL), and sparse reading does not undo this. This is the motivation for Stage 3.
- **Synergy** (D sparse − D dense) − (A sparse − A dense): +0.12 / −0.14 / +0.01. Stage 1 demo refinement and Stage 2 sparsity do not interact.

## Label-shift hypothesis (BBH)

- **Selection.** Top-k keeps a True-token share 0.007 above the legal share, which is negligible. Across the 20 seeds, the change in predicted-True (sparse − dense) has a mean of +0.003, uncorrelated with the seed's demo label bias.
- **Adaptation.** The drop from ICL to CT-KV comes with lower True recall (0.906 → 0.878) and lower predicted-True (0.620 → 0.602). Sparse routing does not change this pattern.
- **Conclusion.** Sparse routing does not act as a label-rebalancing mechanism on this panel.

## Cost

Serial benchmark (GPU alone, arm D exploration cells, 3 jobs per variant; ratios vs production SDPA of the same family; `REPORT_results.md` has the full table):

| family | adaptation s: prod / dense / top-k 75 / top-k 50 | × prod (top-k 75) | eval ms/item: prod / top-k 75 | × prod | peak MiB prod / top-k 75 |
|---|---|---|---|---|---|
| BBH | 2.75 / 3.04 / 3.27 / 3.27 | 1.19 | 14.7 / 21.7 | 1.47 | 15815 / 16028 |
| NLP-LR clf | 9.03 / 12.64 / 16.12 / 16.14 | 1.79 | 1.28 / 3.02 | 2.36 | 6148 / 9050 |
| NLP-LR mc | 10.61 / 15.95 / 22.26 / 22.24 | 2.10 | 1.30 / 3.78 | 2.91 | 6962 / 10841 |


- **Exact top-k is a diagnostic implementation.** It computes the full score matrix and then masks, so it reads every key and cannot be faster than dense. Resident KV is unchanged: BBH 22.5 MiB, NLP-LR about 60–80 MiB per task.
- **The access ratio is the most a fused kernel could save.** Top-k 75% reads 0.75 of the legal demonstration positions and top-k 50% reads 0.50.
- **Only physical pruning reduces storage.** At 50% it brings BBH resident KV from 22.5 to 12.5 MiB. Against production its adaptation time is 1.00× and its evaluation 0.98×, because at this prefix length (about 1,100 tokens) the attention reads are not the bottleneck.
- **Fixed and random selection cost the same as top-k**: the time goes into masking the full score matrix, not into the ranking. In this implementation, top-k at 50% costs the same as at 75%.

## Quest-style paging: not triggered

The guideline's condition is that exact top-k is promising against dense **and** beats same-budget random. Neither holds:
- the main contrast is +0.10 / −0.10 / +0.08 with intervals centred on 0;
- top-k − random intervals all cover 0.

Page-level selection can only approximate exact top-k, so at best it would match exact top-k, which itself shows no accuracy effect. A speed-only (efficiency) study of Quest-style paging would require a fused sparse kernel. That is a separate engineering task and is not justified by these accuracy results.

## Reading of the literature (for context; not tested beyond the above)

- **Gupta et al. (2021), top-k attention.** They report that exact top-k keeps the accuracy of dense attention at moderate k. We see the same here, including during adaptation through the KV cache.
- **Quest (Tang et al., 2024).** The page-level min/max key bounds approximate exact top-k for long-context decoding, where reads dominate. Our demonstration prefixes are short (about 550–1,100 tokens) and reads are a small part of the cost, so the potential gain is small.
- **KVzip (Kim et al., 2025).** It evicts KV positions that are unimportant for reconstructing the context, which is closer to the physical pruning arm. On BBH that arm has the highest point estimate, but it is not significant.

The Notion meeting notes and the 11/24 interim report linked in the guideline are behind a login and could not be read; this report relies on the guideline text and the GitHub sources.

## Next

Stage 3 round 1 (half-delta update and same-trajectory early stopping) is queued after this benchmark: `results/ctkv_stage3_round1/PLAN.md`.
