"""Optional adapters for the two existing CT-KV runners."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from refine import Selector


def dump_lines(path, rows):
    content = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    if path.exists() and path.read_text() != content:
        raise ValueError(f"Existing plan differs: {path}. Choose a new result directory.")
    path.write_text(content)
    print(path.name, len(rows), hashlib.sha256(content.encode()).hexdigest())


def main():
    p = argparse.ArgumentParser()
    for flag in ("main-root", "bbh-kit", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    args = p.parse_args()
    root, kit, out = args.main_root, args.bbh_kit, args.out.resolve()
    split = json.loads((kit/"data/manifest.json").read_text())["tasks"]["boolean_expressions"]
    examples = json.loads((kit/"data/bbh/boolean_expressions.json").read_text())["examples"]
    pool, test = split["support_pool_ids"], split["test_ids"]
    assert len(set(pool)) == 50 and len(set(test)) == 200 and not set(pool) & set(test)
    checked = [{"id": str(i), "input": examples[i]["input"], "output": examples[i]["target"]}
               for i in pool]
    plans, jobs = [], []

    def record(task, family, result):
        for arm, selected in result["arms"].items():
            row = {"task": task, "family": family, "arm": arm, "seed": result["seed"],
                   "shots": result["k"], "pool_counts": result["pool_counts"],
                   **selected, "ids": [int(i) for i in selected["ids"]]}
            plans.append(row)
            yield row

    s = Selector(checked, "classification")
    for seed in range(1000, 1020):
        list(record("boolean_expressions", "bbh", s.select(10, seed)))
    base = root/"results/ctkv_accuracy_study/nlp_lr/e0/jobs_e0.jsonl"
    pools = defaultdict(dict)
    for line in base.read_text().splitlines():
        job = json.loads(line)
        for r in job["demos"]:
            i = r["pool_id"]
            if i is not None:
                prev = pools[job["task"]].setdefault(i, r)
                assert all(prev[k] == r[k] for k in ("input","options","output"))
    for task, source in sorted(pools.items()):
        cfg = json.loads((root/"context-tuning/config/tasks"/(task+".json")).read_text())
        kind = {"classification": "classification", "multi-choice": "local_choice"}[cfg["task_type"]]
        s = Selector([dict(r, id=str(i)) for i, r in source.items()], kind)
        family = "clf" if s.fixed else "mc"
        for seed in range(1000, 1010):
            for row in record(task, family, s.select(16, seed)):
                rid = f"generic_{task}_{row['arm']}_s{seed}"
                jobs.append(dict(id=rid, task=task, demos=[source[i] for i in row["ids"]],
                                 seed=0, loss="raw", epochs=200, checkpoints=[0,200],
                                 out=str(out/"raw_nlplr"/(rid+".json")), meta=row))
    out.mkdir(parents=True, exist_ok=True)
    dump_lines(out/"selection.jsonl", plans)
    dump_lines(out/"nlplr_jobs.jsonl", jobs)


if __name__ == "__main__":
    main()
