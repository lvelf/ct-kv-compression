"""Check decoding, optimization and attention-scoring context lengths with the pinned model tokenizer."""
import argparse
import csv
import json
from collections import Counter
from itertools import groupby
from pathlib import Path

from bbh_eval.prepare_benchmark_splits import benchmark_rows
from bbh_eval.run_llama_icl import MODEL, MODEL_REV, encode_example, encode_instruction
from run_matrix import jobs_for


def check(manifest, source_dir, design, shot_tasks, tokenizer, context):
    for task, task_jobs in groupby(sorted(jobs_for(manifest, design, shot_tasks),
                                         key=lambda j: j.task), key=lambda j: j.task):
        conf = manifest['tasks'][task]
        rows = benchmark_rows(task, source_dir)
        spec = tuple(conf[name] for name in ('task_prompt', 'answer_format', 'generation_length'))
        inst = len(encode_instruction(tokenizer, task, spec))
        support = {}
        for i in conf['support_pool_ids']:
            ids, _, end = encode_example(tokenizer, rows[i]['input'], rows[i]['target'])
            support[i] = (len(ids), end - 1)
        longest_q = max(len(encode_example(tokenizer, rows[i]['input'])[0]) for i in conf['test_ids'])
        for job in task_jobs:
            ids = conf['seeds'][str(job.seed)]['demo_ids'][:job.shots]
            prefix = inst + sum(support[i][0] for i in ids)
            decode = prefix + longest_q + spec[2]
            adapt = (prefix + max((support[i][1] for i in ids), default=0)
                     if job.steps or job.method == 'answer_attention' else 0)
            reasons = []
            if decode > context:
                reasons.append('decoding context')
            if adapt > context:
                reasons.append('tuning/scoring context')
            yield dict(task=task, seed=job.seed, shots=job.shots, method=job.method,
                       steps=job.steps, budget=job.budget,
                       n_selected_demos=len(ids), max_decode_positions=decode,
                       max_adapt_positions=adapt, context_limit=context,
                       fits=not reasons, reason=' / '.join(reasons))


def main():
    import transformers
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--design', choices=('shots',), default='shots')
    p.add_argument('--shot-tasks', choices=('two',), default='two')
    a = p.parse_args()
    cfg = transformers.AutoConfig.from_pretrained(MODEL, revision=MODEL_REV, local_files_only=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV, use_fast=True,
                                                    local_files_only=True)
    rows = check(json.loads(a.manifest.read_text()), a.source_dir, a.design, a.shot_tasks,
                 tok, cfg.max_position_embeddings)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    temp = a.out.with_suffix('.part')
    with temp.open('w', newline='') as fh:
        writer = None
        for row in rows:
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            counts[row['reason'] or 'fits context limit'] += 1
    temp.replace(a.out)
    for reason, count in counts.items():
        print(f'{count:6d}  {reason}')
    print(f'All {sum(counts.values())} planned configurations checked. CSV: {a.out}')
    print('This CPU/tokenizer check does not measure GPU memory.')


if __name__ == '__main__':
    main()
