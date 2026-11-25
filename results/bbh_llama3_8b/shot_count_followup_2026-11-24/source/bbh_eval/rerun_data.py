"""Fixed-test, nested 5/10/20/30-shot BBH splits for the compression rerun."""
import argparse
import json
import random
from pathlib import Path

from bbh_eval.prepare_benchmark_splits import (
    BENCHMARK_HASHES, BENCHMARK_SEEDS, SPEC_REVISION, SPEC_SHA, benchmark_rows,
    benchmark_split, load_task_specs,
)
from bbh_eval.prepare_splits import REVISION, TASK_HASHES, TEST_SEED, make_task_split, write_atomically
from bbh_eval.run_llama_icl import TASKS as PROJECT_SPECS

PROTOCOL = "bbh27_fixed_test_nested_shots_v1"
SHOTS = (5, 10, 20, 30)


def make_split(task, rows):
    """Anchor each seed on its original 10; keep test items and demo texts disjoint."""
    n = len(rows)
    anchors = {
        seed: benchmark_split(n, seed, task, "project_demo_pool")[0]
        for seed in BENCHMARK_SEEDS
    }
    if task in TASK_HASHES:
        # Exactly the Phase 2 fixed 200 questions, including original row order.
        test = make_task_split(n)["test_ids"]
        pool = sorted(set(range(n)) - set(test))
    else:
        # Protect every original ten-shot set across seeds. Reserve at least 50
        # distinct support texts so different seeds can draw different 30-shot sets.
        pool = sorted({i for ids in anchors.values() for i in ids})
        seen = {rows[i]["input"] for i in pool}
        support_text_count = len(seen)
        perm = random.Random(TEST_SEED).sample(range(n), n)
        test = []
        for i in perm:
            text = rows[i]["input"]
            if text in seen:
                continue
            seen.add(text)
            if support_text_count < 50:
                pool.append(i)
                support_text_count += 1
            elif len(test) < 200:
                test.append(i)
        pool.sort()
    if not test or len({rows[i]["input"] for i in pool}) < max(SHOTS):
        raise ValueError(f"{task}: too few distinct rows for support and evaluation.")
    test_texts = {rows[i]["input"] for i in test}
    if len(test_texts) != len(test) or test_texts.intersection(rows[i]["input"] for i in pool):
        raise ValueError(f"{task}: test texts repeat or overlap the support pool.")
    seeds = {}
    for seed, original in anchors.items():
        if not set(original).issubset(pool) or len(original) != 10:
            raise ValueError(f"{task} seed {seed}: unexpected historical demo IDs.")
        # The first ten preserve the historical order, including source-text
        # duplicates if the historical seed contains any. Add new texts only.
        ordered = list(original)
        demo_texts = {rows[i]["input"] for i in ordered}
        for i in random.Random(seed).sample(pool, len(pool)):
            if rows[i]["input"] not in demo_texts:
                ordered.append(i)
                demo_texts.add(rows[i]["input"])
                if len(ordered) == max(SHOTS):
                    break
        if len(ordered) != max(SHOTS):
            raise ValueError(f"{task} seed {seed}: insufficient extra demonstrations.")
        seeds[str(seed)] = {"demo_ids": ordered,
                            "duplicate_texts_in_original_ten": 10 - len({rows[i]["input"] for i in original})}
    return {"seeds": seeds, "test_ids": test, "support_pool_ids": pool}


def build_manifest(source_dir=None, spec_source=None):
    specs = load_task_specs(spec_source)
    tasks = {}
    for task in sorted(BENCHMARK_HASHES):
        rows = benchmark_rows(task, source_dir)
        spec = specs[task]
        if task in PROJECT_SPECS:
            if (spec["task_prompt"], spec["answer_format"], spec["generation_length"]) != PROJECT_SPECS[task]:
                raise ValueError(f"Pinned task specification differs from old Phase 2 for {task}.")
        if any(not all(isinstance(row[key], str) and row[key] for key in ("input", "target")) for row in rows):
            raise ValueError(f"{task}: expected nonempty input and target text.")
        tasks[task] = {**spec, "sha256": BENCHMARK_HASHES[task], "n_rows": len(rows), **make_split(task, rows)}
    return {"protocol": PROTOCOL, "dataset_revision": REVISION, "task_spec_sha256": SPEC_SHA,
            "antecedent_revision": SPEC_REVISION, "seeds": list(BENCHMARK_SEEDS),
            "shot_counts": list(SHOTS), "fixed_test_seed": TEST_SEED, "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, help="Optional offline directory with all pinned BBH JSON files.")
    parser.add_argument("--spec-source", type=Path, help="Optional pinned Fewshot-TTT tasks.py.")
    a = parser.parse_args()
    manifest = build_manifest(a.source_dir, a.spec_source)
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    if a.out.exists() and a.out.read_bytes() != raw:
        parser.error(f"Different manifest already exists at {a.out}")
    if not a.out.exists():
        write_atomically(a.out, raw)
    for task, info in manifest["tasks"].items():
        print(f"{task}: {len(info['test_ids'])} fixed test questions, {len(info['support_pool_ids'])} support IDs")
    print(a.out)


if __name__ == "__main__":
    main()
