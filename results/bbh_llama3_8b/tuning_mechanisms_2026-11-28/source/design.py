"""Deterministic, pre-specified BBH design. Uses the 11/24 split manifest as its anchor."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random


DATA = Path(__file__).resolve().parent / "data"
MANIFEST_HASH = "431c8b6e8a96061b9337ac5ed5f585c4df50a74bb398b48096522136deeaa19e"
BOOLEAN = "boolean_expressions"
DATE = "date_understanding"
TASKS = (BOOLEAN, DATE)
SEEDS = (0, 1, 2)
RATIOS = (0.2, 0.5, 0.8)
BUDGETS = (0.25, 0.5, 0.75)


def rng_for(seed, purpose):
    """Independent deterministic streams for support, order and validation."""
    digest = sha256(f"bbh-mechanisms-v1/{seed}/{purpose}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def load_sources():
    payload = (DATA / "manifest.json").read_bytes()
    if sha256(payload).hexdigest() != MANIFEST_HASH:
        raise ValueError("The original study manifest checksum changed.")
    manifest = json.loads(payload)["tasks"]
    examples = {}
    for task in TASKS:
        payload = (DATA / "bbh" / f"{task}.json").read_bytes()
        if sha256(payload).hexdigest() != manifest[task]["sha256"]:
            raise ValueError(f"Unexpected dataset checksum for {task}.")
        rows = json.loads(payload)["examples"]
        pool = manifest[task]["support_pool_ids"]
        test = manifest[task]["test_ids"]
        if (len(set(pool)) != 50 or len(set(test)) != 200 or set(pool) & set(test)
                or len({rows[i]["input"] for i in pool + test}) != 250):
            raise ValueError(f"{task}: expected 50 distinct support and 200 disjoint test texts.")
        examples[task] = rows
    return manifest, examples


def validation_ids(task, rows, available, seed, design):
    """Select ten held-out support rows before evaluation, with Boolean labels balanced."""
    rng = rng_for(seed, f"validation/{design}/{task}")
    available = sorted(available)
    if task == BOOLEAN:
        ids = []
        for label in ("True", "False"):
            ids.extend(rng.sample([i for i in available if rows[i]["target"] == label], 5))
        rng.shuffle(ids)
        return ids
    return rng.sample(available, 10)


def stratified_sets(rows, pool, seed):
    """Use common shuffled per-label lists and randomized prompt positions in two blocks of ten."""
    orders = {}
    for label in ("True", "False"):
        candidate_ids = sorted(i for i in pool if rows[i]["target"] == label)
        orders[label] = rng_for(seed, f"support/{label}").sample(candidate_ids, len(candidate_ids))
    # Use identical position priorities across proportions, with independently randomized halves.
    # The 20-shot prompt starts with exactly the corresponding 10-shot prompt.
    priorities = [rng_for(seed, f"order/{h}").sample(range(10), 10) for h in (0, 1)]
    cells = {}
    for ratio in RATIOS:
        n_true = round(10 * ratio)
        order = []
        counts = Counter()
        for h, priority in enumerate(priorities):
            true_positions = set(priority[:n_true])
            for i in range(10):
                label = "True" if i in true_positions else "False"
                order.append(orders[label][counts[label]])
                counts[label] += 1
            cells[(10 * (h + 1), ratio)] = order.copy()
    return cells


def build_jobs():
    manifest, examples = load_sources()
    jobs = []
    historical = {}
    main = {}

    for task in TASKS:
        spec = manifest[task]
        rows = examples[task]
        pool = spec["support_pool_ids"]
        for seed in SEEDS:
            original = spec["seeds"][str(seed)]["demo_ids"][:20]
            val = validation_ids(task, rows, set(pool) - set(original), seed, "historical")
            for k in (10, 20):
                historical[(task, seed, k)] = (original[:k], val)
            if task == BOOLEAN:
                cells = stratified_sets(rows, pool, seed)
                used = set().union(*(set(ids) for ids in cells.values()))
                main_val = validation_ids(task, rows, set(pool) - used, seed, "main")
                for (k, ratio), ids in cells.items():
                    main[(seed, k, ratio)] = (ids, main_val)

    def add(stage, task, seed, k, method, updates, *, ratio=None, budget=1.0,
            param_mode="both", optimizer_seed=0):
        if stage == "main":
            demo_ids, val_ids = main[(seed, k, ratio)]
        else:
            demo_ids, val_ids = historical[(task, seed, k)]
        tag = f"true{round(100 * ratio)}" if ratio is not None else "original"
        name = f"{stage}-{task}-s{seed}-k{k}-{tag}-{method}-b{round(100*budget)}-u{updates}-{param_mode}-o{optimizer_seed}"
        jobs.append(dict(id=name, stage=stage, task=task, shots=k, demo_seed=seed,
                         ratio=ratio, demo_ids=list(demo_ids), val_ids=list(val_ids),
                         test_ids=list(manifest[task]["test_ids"]), method=method, budget=budget,
                         updates=updates, param_mode=param_mode, optimizer_seed=optimizer_seed,
                         selection_seed=seed))

    for seed in SEEDS:
        for k in (10, 20):
            for ratio in RATIOS:
                add("main", BOOLEAN, seed, k, "icl", 0, ratio=ratio)
                for optim in (0, 1):
                    add("main", BOOLEAN, seed, k, "full", 16, ratio=ratio, optimizer_seed=optim)
                    add("main", BOOLEAN, seed, k, "answer_attention", 16, ratio=ratio,
                        budget=0.25, optimizer_seed=optim)
    for task in TASKS:
        for seed in SEEDS:
            for steps in (4, 8, 32):
                add("steps", task, seed, 10, "full", steps)
    for task in TASKS:
        for seed in SEEDS:
            for k in (10, 20):
                for mode in ("key", "value"):
                    add("params", task, seed, k, "full", 16, param_mode=mode)
    for task in TASKS:
        for seed in SEEDS:
            for budget in BUDGETS:
                add("random", task, seed, 20, "random", 16, budget=budget)
    for task in TASKS:
        for seed in SEEDS:
            for k in (10, 20):
                add("controls", task, seed, k, "full", 16)
            for budget in BUDGETS:
                add("controls", task, seed, 20, "answer_attention", 16, budget=budget)
    validate_jobs(jobs, manifest, examples)
    return jobs


def validate_jobs(jobs, manifest, examples):
    if Counter(j["stage"] for j in jobs) != dict(main=90, steps=18, params=24, random=18, controls=30):
        raise ValueError("Run count differs from the pre-specified design.")
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate job IDs.")
    for j in jobs:
        task = j["task"]
        ids, val, test = (j[k] for k in ("demo_ids", "val_ids", "test_ids"))
        k = j["shots"]
        if (len(set(ids)) != k or len(ids) != k or len(val) != 10 or len(set(val)) != 10
                or (set(ids) | set(val)) - set(manifest[task]["support_pool_ids"])
                or set(ids) & set(val) or set(test) & (set(ids) | set(val))
                or test != manifest[task]["test_ids"]):
            raise ValueError(f'{j["id"]}: invalid data split.')
        if j["ratio"] is not None:
            count = sum(examples[task][i]["target"] == "True" for i in ids)
            if count != round(j["ratio"] * k):
                raise ValueError(f'{j["id"]}: incorrect label composition.')
        else:
            if ids != manifest[task]["seeds"][str(j["demo_seed"])]["demo_ids"][:k]:
                raise ValueError(f'{j["id"]}: historical demonstration order changed.')
    for seed in SEEDS:
        subset = [j for j in jobs if j["stage"] == "main" and j["demo_seed"] == seed]
        if len({tuple(j["val_ids"]) for j in subset}) != 1:
            raise ValueError(f"Main seed {seed}: validation IDs differ across methods or cells.")
        for ratio in RATIOS:
            at_ratio = [j for j in subset if j["ratio"] == ratio]
            ids20 = next(j["demo_ids"] for j in at_ratio if j["shots"] == 20)
            if any(j["demo_ids"] != ids20[:j["shots"]] for j in at_ratio):
                raise ValueError(f"Main seed {seed}: unintended demonstration order change.")


if __name__ == "__main__":
    grid = build_jobs()
    print(f"{len(grid)} formal runs")
    for stage, count in Counter(job["stage"] for job in grid).items():
        print(f"{stage}: {count}")
