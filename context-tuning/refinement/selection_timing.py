"""CPU cost of the selector, per task and seed (pools rebuilt exactly as benchmark_plan.py does).

fit_s = Selector(pool) construction; select1_s = select(k, seed, candidates=1) (what A/B/C need);
select32_s = select(k, seed) with the default 32 candidates (D). Also checks select(32) equals selection.jsonl.
Usage (from refinement/): OMP_NUM_THREADS=1 python selection_timing.py --main-root R --bbh-kit K --out RUN_DIR
"""
import argparse, json, time
from collections import defaultdict
from pathlib import Path

from refine import Selector


def main():
    p = argparse.ArgumentParser()
    for flag in ("main-root", "bbh-kit", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    a = p.parse_args()
    frozen = defaultdict(dict)
    for l in open(a.out / "selection.jsonl"):
        r = json.loads(l)
        frozen[(r["task"], r["seed"])][r["arm"]] = [str(i) for i in r["ids"]]
    split = json.loads((a.bbh_kit / "data/manifest.json").read_text())["tasks"]["boolean_expressions"]
    ex = json.loads((a.bbh_kit / "data/bbh/boolean_expressions.json").read_text())["examples"]
    pools = [("boolean_expressions", "bbh", [{"id": str(i), "input": ex[i]["input"], "output": ex[i]["target"]}
                                             for i in split["support_pool_ids"]], "classification", 10,
              range(1000, 1020))]
    src = defaultdict(dict)
    for line in (a.main_root / "results/ctkv_accuracy_study/nlp_lr/e0/jobs_e0.jsonl").read_text().splitlines():
        job = json.loads(line)
        for r in job["demos"]:
            if r["pool_id"] is not None:
                src[job["task"]].setdefault(r["pool_id"], r)
    for task, rows in sorted(src.items()):
        cfg = json.loads((a.main_root / "context-tuning/config/tasks" / (task + ".json")).read_text())
        kind = {"classification": "classification", "multi-choice": "local_choice"}[cfg["task_type"]]
        pools.append((task, None, [dict(r, id=str(i)) for i, r in rows.items()], kind, 16, range(1000, 1010)))
    out = []
    for task, fam, rows, kind, k, seeds in pools:
        for seed in seeds:
            t = time.perf_counter()
            s = Selector(rows, kind)
            fit = time.perf_counter() - t
            t = time.perf_counter()
            s.select(k, seed, candidates=1)
            one = time.perf_counter() - t
            t = time.perf_counter()
            res = s.select(k, seed)
            full = time.perf_counter() - t
            for arm, v in res["arms"].items():
                assert v["ids"] == frozen[(task, seed)][arm], (task, seed, arm)
            out.append(dict(task=task, family=fam or ("clf" if s.fixed else "mc"), seed=seed, n=len(rows),
                            fit_s=fit, select1_s=one, select32_s=full))
    json.dump(out, open(a.out / "selection_timing.json", "w"), indent=1)
    print(len(out), "timings; all selections reproduce selection.jsonl")


if __name__ == "__main__":
    main()
