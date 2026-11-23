# BBH results on Llama 3 8B Instruct

## Phase 1. Full-cache benchmark

The independent reconstruction covers 27 BBH task files and five demonstration sets. Accuracy is the equally weighted mean across task files, then across sets. The standard deviation is across the five sets.

| Method | Strict accuracy (%) | Normalized accuracy (%) |
|---|---:|---:|
| Standard ICL | 49.13 ± 0.49 | 49.13 ± 0.49 |
| Full CT-KV | 56.38 ± 0.89 | 56.39 ± 0.90 |

## Phase 2. Held-out compression results

The development pilot selected **`answer_attention`** at 50% retention, using accuracy first and mean adaptation time as the first tiebreaker. Test comparisons use the frozen rule on 200 held-out questions per task, averaged across three demonstration sets. Each set uses the same 200 questions. Random and selected runs keep equal cache bytes and use 16 updates. Retention fractions apply to demonstration positions. All instruction positions are retained.

| Retention | Selected accuracy (%) | Random accuracy (%) | Full CT-KV accuracy (%) | Selected − random (pp) | Selected − full (pp) | Task outcomes (ahead / tied / behind random) |
|---:|---:|---:|---:|---:|---:|---:|
| 25% | 60.38 | 62.12 | 63.04 | -1.75 | -2.67 | 1 / 0 / 3 |
| 50% | 61.62 | 62.54 | 63.04 | -0.92 | -1.42 | 1 / 0 / 3 |
| 75% | 63.08 | 63.08 | 63.04 | +0.00 | +0.04 | 2 / 0 / 2 |

Each task has equal weight in this table. Differences are descriptive percentage points (pp).

### Absolute accuracy and costs

Each row summarizes three demonstration sets. Accuracy shows mean ± sample SD. Costs are means per run. Adaptation includes initial cache construction, selection and optimization. Peak GPU memory is PyTorch peak allocated memory across adaptation and inference.

| Task | Method | Retention | Accuracy (%) | Retained KV (MiB) | Peak GPU (MiB) | Adaptation (s) | Inference (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Boolean Expressions | full | 100% | 71.33 ± 9.45 | 22.50 | 15813.9 | 6.46 | 3.76 |
| Boolean Expressions | answer_attention | 25% | 75.67 ± 1.61 | 7.50 | 15583.5 | 6.63 | 4.05 |
| Boolean Expressions | random | 25% | 72.33 ± 2.25 | 7.50 | 15583.5 | 6.15 | 3.67 |
| Boolean Expressions | answer_attention | 50% | 73.33 ± 5.01 | 12.50 | 15656.5 | 6.94 | 4.18 |
| Boolean Expressions | random | 50% | 74.00 ± 3.12 | 12.50 | 15656.5 | 6.75 | 3.69 |
| Boolean Expressions | answer_attention | 75% | 73.17 ± 7.32 | 17.50 | 15715.0 | 6.93 | 3.94 |
| Boolean Expressions | random | 75% | 72.50 ± 8.72 | 17.50 | 15715.0 | 6.84 | 4.15 |
| Date Understanding | full | 100% | 55.83 ± 2.57 | 127.75 | 18275.0 | 8.35 | 12.35 |
| Date Understanding | answer_attention | 25% | 48.83 ± 1.76 | 33.96 | 16937.2 | 7.94 | 11.66 |
| Date Understanding | random | 25% | 54.33 ± 1.76 | 33.96 | 16930.6 | 7.58 | 11.33 |
| Date Understanding | answer_attention | 50% | 52.17 ± 2.47 | 65.21 | 17420.8 | 7.96 | 11.89 |
| Date Understanding | random | 50% | 55.50 ± 1.32 | 65.21 | 17417.4 | 7.60 | 11.64 |
| Date Understanding | answer_attention | 75% | 55.17 ± 2.36 | 96.46 | 17836.6 | 8.15 | 11.95 |
| Date Understanding | random | 75% | 54.83 ± 1.53 | 96.46 | 17835.3 | 8.07 | 12.22 |
| Word Sorting | full | 100% | 57.00 ± 1.32 | 70.62 | 17263.8 | 7.45 | 81.47 |
| Word Sorting | answer_attention | 25% | 56.33 ± 1.89 | 19.04 | 16531.1 | 7.65 | 77.70 |
| Word Sorting | random | 25% | 57.17 ± 1.76 | 19.04 | 16531.4 | 7.12 | 77.58 |
| Word Sorting | answer_attention | 50% | 57.00 ± 2.18 | 36.25 | 16750.3 | 7.57 | 79.35 |
| Word Sorting | random | 50% | 57.33 ± 2.25 | 36.25 | 16750.3 | 7.09 | 78.84 |
| Word Sorting | answer_attention | 75% | 57.00 ± 2.50 | 53.42 | 17000.2 | 7.39 | 80.37 |
| Word Sorting | random | 75% | 57.33 ± 2.75 | 53.42 | 17000.2 | 7.54 | 80.50 |
| Logical Deduction (three objects) | full | 100% | 68.00 ± 4.27 | 136.42 | 18433.4 | 8.55 | 12.72 |
| Logical Deduction (three objects) | answer_attention | 25% | 60.67 ± 2.93 | 36.96 | 16998.8 | 7.87 | 11.85 |
| Logical Deduction (three objects) | random | 25% | 64.67 ± 1.76 | 36.96 | 17004.4 | 7.65 | 11.52 |
| Logical Deduction (three objects) | answer_attention | 50% | 64.00 ± 2.65 | 70.08 | 17508.1 | 8.06 | 11.77 |
| Logical Deduction (three objects) | random | 50% | 63.33 ± 6.05 | 70.08 | 17505.2 | 8.08 | 11.98 |
| Logical Deduction (three objects) | answer_attention | 75% | 67.00 ± 5.89 | 103.21 | 17956.4 | 8.37 | 12.34 |
| Logical Deduction (three objects) | random | 75% | 67.67 ± 3.88 | 103.21 | 17956.9 | 8.40 | 12.22 |

The full-cache comparator in Phase 2 uses the same project tasks and splits as the compressed runs. The results describe one checkpoint, four tasks and three demonstration sets. The task summaries also contain standard ICL, selected-cache zero-update runs and whole-demonstration controls. The claims above cover the core random-versus-selected test only.

Checkpoint: `meta-llama/Meta-Llama-3-8B-Instruct@8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`.

### Development decision

| Candidate | Correct / 480 development predictions | Mean adaptation (s) |
|---|---:|---:|
| answer_attention | 294 / 480 | 7.67 |
| value_norm | 264 / 480 | 7.25 |

Phase 1 evaluation included questions from the four project test sets. The selector was chosen by the development scores recorded in the frozen decision file. The 36 matched test comparisons share questions across demonstration seeds within each task.

Reproduce the project phase from the repository root in the Phase 1 environment:

```bash
OUT="$(mktemp -d)"
python -m bbh_eval.experiment develop --project-model 8b --root "$OUT" --execute &&
python -m bbh_eval.experiment freeze --project-model 8b --root "$OUT" --execute &&
python -m bbh_eval.experiment project-test --project-model 8b --root "$OUT" --execute &&
python -m bbh_eval.summarize_llama_runs --root "$OUT/project-test/8b" --out "$OUT/project-summary"
```
