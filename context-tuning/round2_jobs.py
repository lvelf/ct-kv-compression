"""Build the accuracy round 2 job lists from the frozen Stage 1 manifest (results/ctkv_refinement/nlplr_jobs.jsonl).

Arms A (random demonstrations) and D (Stage 1 selection), 21 tasks x 10 demonstration seeds each. Every job copies
the Stage 1 job verbatim (demo IDs, order, texts, raw loss, optimizer seed 0, 200-step cosine schedule).

  --timing   OUT/jobs_timing.jsonl: per task, seed 1000, arms A and D, original and ranking loss, 20 steps, no test
             evaluation (timing_only). Used only to fix the equal-GPU-time step budget.
  --budget   reads OUT/raw_timing, writes OUT/rank_budget.json: per task
             N = clip(5 * floor(200 * t_orig / t_rank / 5), 5, 200), t = median step time after 3 warm-up steps,
             averaged over the A and D timing units.
  --main     (needs rank_budget.json) OUT/jobs_{smoke,D_orig,D_rank,A_orig,A_rank}.jsonl
             orig: checkpoints 0 / 50 / 200.  rank: rank_weight 1.0, checkpoints 0 (null scores only) / 50 / N_task,
             stopped at max(50, N_task). Smoke jobs (financial_phrasebank, ai2_arc; seed 1000; D orig and D rank) are
             part of the panel and carry smoke=true in every list.

Usage (from context-tuning/): python round2_jobs.py --stage1 ../results/ctkv_refinement --out ../results/ctkv_accuracy_round2 --timing|--budget|--main
"""
import argparse
import json
from pathlib import Path

import numpy as np

SMOKE = {("financial_phrasebank", 1000), ("ai2_arc", 1000)}
TIMING_STEPS = 20


def base(a):
    rows = [json.loads(l) for l in open(a.stage1 / "nlplr_jobs.jsonl")]
    rows = [r for r in rows if r["meta"]["arm"] in ("A", "D")]
    for r in rows:
        assert r["loss"] == "raw" and r.get("seed", 0) == 0 and r["epochs"] == 200, r["id"]
    return rows


def write(path, jobs):
    with open(path, "w") as f:
        for j in jobs:
            f.write(json.dumps(j, sort_keys=True) + "\n")
    print(path, len(jobs))


def make(b, out, kind, sub, **kw):
    j = dict(task=b["task"], demos=b["demos"], seed=b["seed"], epochs=b["epochs"], loss="raw", meta=b["meta"],
             stage1_id=b["id"], null_templates=["na", "dots"], **kw)
    j["id"] = f"r2_{b['task']}_{b['meta']['arm']}_s{b['meta']['seed']}_{kind}"
    j["out"] = str(out / sub / (j["id"] + ".json"))
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--timing", action="store_true")
    g.add_argument("--budget", action="store_true")
    g.add_argument("--main", action="store_true")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = base(a)
    if a.timing:
        jobs = []
        for b in sorted(rows, key=lambda r: (r["task"], r["meta"]["arm"])):
            if b["meta"]["seed"] != 1000:
                continue
            for kind, w in (("torig", 0.0), ("trank", 1.0)):
                jobs.append(make(b, out, kind, "raw_timing", rank_weight=w, checkpoints=[TIMING_STEPS],
                                 stop_at=TIMING_STEPS, timing_only=True))
        write(out / "jobs_timing.jsonl", jobs)
        return
    if a.budget:
        t = {}
        for p in sorted((out / "raw_timing").glob("*.json")):
            r = json.load(open(p))
            t.setdefault(r["task"], {}).setdefault("rank" if r["rank_weight"] > 0 else "orig", []).append(
                r["timing"]["step_s_median_after_warmup"])
        budget = {}
        for task, v in sorted(t.items()):
            assert len(v["orig"]) == 2 and len(v["rank"]) == 2, (task, v)
            to, tr = float(np.mean(v["orig"])), float(np.mean(v["rank"]))
            n = int(min(200, max(5, 5 * int(200 * to / tr / 5))))
            budget[task] = dict(step_s_orig=to, step_s_rank=tr, ratio=tr / to, budget_steps=n)
        assert len(budget) == 21
        json.dump(budget, open(out / "rank_budget.json", "w"), indent=1)
        for k, v in budget.items():
            print(k, v)
        return
    budget = json.load(open(out / "rank_budget.json"))
    lists = {}
    for b in rows:
        arm, key = b["meta"]["arm"], (b["task"], b["meta"]["seed"])
        smoke = key in SMOKE and arm == "D"
        n = budget[b["task"]]["budget_steps"]
        lists.setdefault(f"{arm}_orig", []).append(make(b, out, "orig", "raw", rank_weight=0.0, checkpoints=[0, 50, 200],
                                                        stop_at=200, smoke=smoke))
        lists.setdefault(f"{arm}_rank", []).append(make(b, out, "rank", "raw", rank_weight=1.0,
                                                        checkpoints=sorted({0, 50, n}), stop_at=max(50, n),
                                                        eval_real=False, budget_steps=n, smoke=smoke))
    for name, jobs in lists.items():
        assert len(jobs) == 210 and len({(j["task"], j["meta"]["seed"]) for j in jobs}) == 210, name
        jobs.sort(key=lambda j: (not j["smoke"], j["task"], j["meta"]["seed"]))
        write(out / f"jobs_{name}.jsonl", jobs)
    write(out / "jobs_smoke.jsonl", [j for n in ("D_orig", "D_rank") for j in lists[n] if j["smoke"]])


if __name__ == "__main__":
    main()
