"""Build job files for the NLP-LR side of the CT-KV accuracy study (see study_nlplr.py for the job format).

Candidate pool per task: the union of the five MetaICL 16-shot train files (splits 13, 21, 42, 87, 100, in that
order), deduplicated by input text, minus any example whose input text appears in the (shared) test file.
pool_id is the position in that union. The natural 16-shot set of split s is that split's train file, in file
order, exactly as in the paper.

Usage (from context-tuning/): .venv/bin/python study_nlplr_jobs.py EXPERIMENT OUT_DIR
"""
import json
import math
import os
import random
import sys
from collections import Counter

from transformers import AutoTokenizer

DATA = "./metaicl-data/data"
SPLITS = (13, 21, 42, 87, 100)
# the 21 tasks that survive the 1024-token filter (the other 5 lose every test example)
TASKS = ["ai2_arc", "climate_fever", "codah", "commonsense_qa", "ethos-national_origin", "ethos-race", "ethos-religion",
         "financial_phrasebank", "glue-wnli", "hate_speech18", "openbookqa", "poem_sentiment", "qasc", "quarel",
         "quartz-no_knowledge", "quartz-with_knowledge", "sick", "superglue-copa", "tweet_eval-hate",
         "tweet_eval-stance_atheism", "tweet_eval-stance_feminist"]
EPOCHS = 200
CPS = [0, 50, 100, 200]
_tok = None


def tok_len(text):
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained("openai-community/gpt2-large", cache_dir="./.cache")
    return len(_tok(text)["input_ids"])


def is_clf(task):
    return json.load(open(f"config/tasks/{task}.json"))["task_type"] == "classification"


def read(path):
    return [json.loads(l) for l in open(path)]


def natural(task, split):
    return [dict(d, pool_id=None) for d in read(f"{DATA}/{task}/{task}_16_{split}_train.jsonl")]


def pool(task):
    test_inputs = {d["input"] for d in read(f"{DATA}/{task}/{task}_16_87_test.jsonl")}
    seen, out = set(), []
    for s in SPLITS:
        for d in read(f"{DATA}/{task}/{task}_16_{s}_train.jsonl"):
            if d["input"] in seen or d["input"] in test_inputs:
                continue
            seen.add(d["input"])
            out.append(dict(d, pool_id=len(out)))
    return out


def with_pool_ids(task, demos):
    """Attach pool ids to natural demos where the example is in the pool (None if excluded as a test overlap)."""
    index = {d["input"]: d["pool_id"] for d in pool(task)}
    return [dict(d, pool_id=index.get(d["input"])) for d in demos]


def strip(d):
    return {k: d[k] for k in ("input", "output", "options", "pool_id") if k in d}


def job(jid, out_dir, task, demos, loss="raw", checkpoints=CPS, epochs=EPOCHS, **kw):
    j = dict(id=jid, out=os.path.join(out_dir, f"{jid}.json"), task=task, demos=[strip(d) for d in demos], loss=loss,
             epochs=epochs, checkpoints=checkpoints, seed=0)
    j.update(kw)
    return j


# ---------------------------------------------------------------- experiments

def exp_e0(out_dir):
    """Natural 16-shot sets of all five splits, all 21 tasks, original loss, checkpoints 0/50/100/200."""
    return [job(f"e0_s{s}_{t}", out_dir, t, with_pool_ids(t, natural(t, s)), meta=dict(exp="e0", split=s))
            for s in SPLITS for t in TASKS]


def binary_tasks(min_per_class=13):
    """Binary classification tasks whose pool has at least min_per_class examples of each label."""
    out = []
    for t in TASKS:
        if not is_clf(t):
            continue
        c = Counter(d["output"] for d in pool(t))
        if len(c) == 2 and min(c.values()) >= min_per_class:
            out.append(t)
    return out


def matched_pairs(cands_a, cands_b, rng, n):
    """Draw n of each label at random, then pair them (min total |length difference|, exact for small n via
    greedy on sorted lengths, which is optimal for 1-D absolute distance)."""
    a = rng.sample(cands_a, n)
    b = rng.sample(cands_b, n)
    a.sort(key=lambda d: d["_len"])
    b.sort(key=lambda d: d["_len"])
    return list(zip(a, b))


def label_ratio_groups(task, seed):
    """Section 2.1 analogue for 16 shots: 6 common examples (3 + 3) in fixed slots and 10 length-matched A/B
    pairs in the remaining slots. Label-A counts 3, 8, 13 of 16 (19%, 50%, 81%; BBH uses 20/50/80% of 10)."""
    rng = random.Random(1000 + seed)
    p = pool(task)
    for d in p:
        d["_len"] = tok_len(d["input"])
    labels = sorted({d["output"] for d in p}, key=lambda x: p[0]["options"].index(x))
    A, B = labels
    pa, pb = [d for d in p if d["output"] == A], [d for d in p if d["output"] == B]
    common = rng.sample(pa, 3) + rng.sample(pb, 3)
    pa = [d for d in pa if d not in common]
    pb = [d for d in pb if d not in common]
    pairs = matched_pairs(pa, pb, rng, 10)
    rng.shuffle(pairs)
    slots = list(range(16))
    rng.shuffle(slots)
    common_slots, swap_slots = slots[:6], slots[6:]
    half = set(rng.sample(range(10), 5))
    groups = {}
    for name, choose_a in (("A3", lambda k: False), ("A8", lambda k: k in half), ("A13", lambda k: True)):
        seq = [None] * 16
        for s, d in zip(common_slots, common):
            seq[s] = d
        for k, (s, (da, db)) in enumerate(zip(swap_slots, pairs)):
            seq[s] = da if choose_a(k) else db
        groups[name] = seq
    info = dict(label_A=A, label_B=B, common_slots=common_slots, swap_slots=swap_slots,
                pair_len_diff=[abs(a["_len"] - b["_len"]) for a, b in pairs],
                pairs=[(a["pool_id"], b["pool_id"]) for a, b in pairs], half=sorted(half))
    return groups, info


def exp_h1(out_dir):
    jobs = []
    for t in binary_tasks():
        for seed in range(5):
            groups, info = label_ratio_groups(t, seed)
            for name, seq in groups.items():
                for order in ("fwd", "rev"):
                    demos = seq if order == "fwd" else seq[::-1]
                    jobs.append(job(f"h1_{t}_s{seed}_{name}_{order}", out_dir, t, demos,
                                    meta=dict(exp="h1", seed=seed, ratio=name, order=order, **info)))
    return jobs


def stratified(task, seed, n=16):
    """Class-stratified random draw from the pool (as even as the pool allows), ordered by pool id."""
    rng = random.Random(2000 + seed)
    p = pool(task)
    by = {}
    for d in p:
        by.setdefault(d["output"], []).append(d)
    classes = sorted(by)
    quota = {c: n // len(classes) for c in classes}
    for c in classes[: n - sum(quota.values())]:
        quota[c] += 1
    # move quota from classes that cannot fill it to the others
    short = sum(max(0, quota[c] - len(by[c])) for c in classes)
    for c in classes:
        quota[c] = min(quota[c], len(by[c]))
    for c in sorted(classes, key=lambda c: -len(by[c])):
        while short and quota[c] < len(by[c]):
            quota[c] += 1
            short -= 1
    picked = [d for c in classes for d in rng.sample(by[c], quota[c])]
    return sorted(picked, key=lambda d: d["pool_id"])


def exp_e2(out_dir):
    """2x2: {natural random 16, stratified 16} x {raw, balanced} on classification tasks, 5 seeds.
    natural+raw is E0 itself (reused at analysis time)."""
    jobs = []
    for t in [t for t in TASKS if is_clf(t)]:
        for i, s in enumerate(SPLITS):
            nat = with_pool_ids(t, natural(t, s))
            strat = stratified(t, i)
            jobs.append(job(f"e2_{t}_s{i}_natural_balanced", out_dir, t, nat, loss="balanced",
                            meta=dict(exp="e2", seed=i, split=s, select="natural", loss="balanced")))
            for loss in ("raw", "balanced"):
                jobs.append(job(f"e2_{t}_s{i}_strat_{loss}", out_dir, t, strat, loss=loss,
                                meta=dict(exp="e2", seed=i, select="stratified", loss=loss)))
    return jobs


def assign_min_cost(src, cands):
    """Assign each item in src a distinct candidate minimizing total |length difference| (1-D: sort both)."""
    if len(cands) < len(src):
        return None
    best, best_cost = None, math.inf
    # sliding window over the sorted candidates is optimal for 1-D matching with equal counts; for unequal
    # counts use a small DP over sorted lists
    s = sorted(src, key=lambda d: d["_len"])
    c = sorted(cands, key=lambda d: d["_len"])
    n, m = len(s), len(c)
    INF = math.inf
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = 0
    for i in range(1, n + 1):
        for j in range(i, m + 1):
            dp[i][j] = min(dp[i][j - 1], dp[i - 1][j - 1] + abs(s[i - 1]["_len"] - c[j - 1]["_len"]))
    i, j, match = n, m, {}
    while i > 0:
        if dp[i][j] == dp[i][j - 1] and j > i:
            j -= 1
        else:
            match[id(s[i - 1])] = c[j - 1]
            i, j = i - 1, j - 1
    return [match[id(d)] for d in src], dp[n][m]


def repair_and_sham(task, demos, rng_seed):
    """Section 4.1: replace exactly |k - 8| majority-label demos by minority-label pool items (repair) or by
    same-label pool items (sham), in the same positions, matched on input token length. Returns
    (repair, sham, info) or None if the pool cannot supply the replacements."""
    rng = random.Random(rng_seed)
    p = pool(task)
    for d in p:
        d["_len"] = tok_len(d["input"])
    demos = [dict(d, _len=tok_len(d["input"])) for d in demos]
    labels = sorted({d["output"] for d in p})
    if len(labels) != 2:
        return None
    c = Counter(d["output"] for d in demos)
    maj = max(labels, key=lambda l: (c.get(l, 0), l))
    mnr = [l for l in labels if l != maj][0]
    k = c.get(maj, 0) - len(demos) // 2
    used = {d["input"] for d in demos}
    if k <= 0:
        return dict(n_replaced=0, repair=demos, sham=demos, positions=[])
    minority_pool = [d for d in p if d["output"] == mnr and d["input"] not in used]
    majority_pool = [d for d in p if d["output"] == maj and d["input"] not in used]
    maj_pos = [i for i, d in enumerate(demos) if d["output"] == maj]
    # choose k majority positions and minority candidates with minimum total length distance
    best = None
    pos_sets = [maj_pos]
    rng.shuffle(maj_pos)
    if len(minority_pool) < k or len(majority_pool) < k:
        return None
    # pick positions: try all positions jointly via matching the k majority positions whose lengths are closest
    # to available minority items (DP over all majority positions, choose k of them)
    src = [demos[i] for i in maj_pos]
    cand = sorted(minority_pool, key=lambda d: d["_len"])
    best_cost, best_choice = math.inf, None
    import itertools
    combos = itertools.combinations(range(len(src)), k) if math.comb(len(src), k) <= 5000 else \
        [tuple(sorted(rng.sample(range(len(src)), k))) for _ in range(5000)]
    for combo in combos:
        r = assign_min_cost([src[i] for i in combo], cand)
        if r and r[1] < best_cost:
            best_cost, best_choice = r[1], (combo, r[0])
    combo, repl = best_choice
    positions = [maj_pos[i] for i in combo]
    sham_repl, sham_cost = assign_min_cost([demos[i] for i in positions], majority_pool)
    repair, sham = list(demos), list(demos)
    for pos, r, s in zip(positions, repl, sham_repl):
        repair[pos], sham[pos] = r, s
    return dict(n_replaced=k, repair=repair, sham=sham, positions=positions, majority=maj, minority=mnr,
                repair_cost=best_cost, sham_cost=sham_cost,
                replaced=[(demos[i].get("pool_id"), repair[i]["pool_id"], sham[i]["pool_id"]) for i in positions])


def exp_h2(out_dir):
    """Section 4.1 on (a) natural 16-shot sets of every binary classification task and (b) the skewed H1 groups."""
    jobs = []
    for t in [t for t in TASKS if is_clf(t)]:
        if len({d["output"] for d in pool(t)}) != 2:
            continue
        for i, s in enumerate(SPLITS):
            r = repair_and_sham(t, with_pool_ids(t, natural(t, s)), 3000 + i)
            if r is None:
                print(f"skip natural {t} s{i}: pool too small", file=sys.stderr)
                continue
            info = {k: v for k, v in r.items() if k not in ("repair", "sham")}
            if r["n_replaced"] == 0:
                continue  # already balanced: repair = sham = original (E0)
            for kind in ("repair", "sham"):
                jobs.append(job(f"h2nat_{t}_s{i}_{kind}", out_dir, t, r[kind],
                                meta=dict(exp="h2_natural", seed=i, split=s, kind=kind, **info)))
    for t in binary_tasks():
        for seed in range(5):
            groups, ginfo = label_ratio_groups(t, seed)
            for name in ("A3", "A13"):
                for order in ("fwd", "rev"):
                    seq = groups[name] if order == "fwd" else groups[name][::-1]
                    r = repair_and_sham(t, seq, 4000 + seed)
                    if r is None:
                        print(f"skip skewed {t} s{seed} {name}: pool too small", file=sys.stderr)
                        continue
                    info = {k: v for k, v in r.items() if k not in ("repair", "sham")}
                    for kind in ("repair", "sham"):
                        jobs.append(job(f"h2skew_{t}_s{seed}_{name}_{order}_{kind}", out_dir, t, r[kind],
                                        meta=dict(exp="h2_skewed", seed=seed, ratio=name, order=order, kind=kind, **info)))
    return jobs


def selection_split(task, seed):
    """E3: shuffle the pool with this seed; 30 candidates + 20 labeled selection-dev examples."""
    p = pool(task)
    rng = random.Random(5000 + seed)
    perm = rng.sample(p, len(p))
    return perm[:30], perm[30:50]


def exp_e3(out_dir, seeds=(0, 1, 2, 3, 4), tag="e3"):
    """Short runs: 8 candidate 16-shot subsets of the 30 candidates; dev NLL at steps 0 and 50 (of a 200-step
    schedule; 50/200 mirrors BBH's 4/16). No test evaluation. Also an early-stopping trajectory for the
    pre-specified random group (candidate 0): dev NLL at 0/50/100/200 plus test at the same checkpoints."""
    jobs = []
    for t in TASKS:
        for seed in seeds:
            cands, dev = selection_split(t, seed)
            rng = random.Random(6000 + seed)
            for c in range(8):
                demos = sorted(rng.sample(cands, 16), key=lambda d: d["pool_id"])
                jobs.append(job(f"{tag}sel_{t}_s{seed}_c{c}", out_dir, t, demos, checkpoints=[], stop_at=50,
                                dev=[strip(d) for d in dev], dev_checkpoints=[0, 50], eval_test=False,
                                meta=dict(exp=f"{tag}_select", seed=seed, cand=c)))
                if c == 0:
                    jobs.append(job(f"{tag}es_{t}_s{seed}_c0", out_dir, t, demos, checkpoints=CPS,
                                    dev=[strip(d) for d in dev], dev_checkpoints=CPS,
                                    meta=dict(exp=f"{tag}_earlystop", seed=seed, cand=0)))
    return jobs


def candidate_groups(task, seed):
    cands, dev = selection_split(task, seed)
    rng = random.Random(6000 + seed)
    return [sorted(rng.sample(cands, 16), key=lambda d: d["pool_id"]) for _ in range(8)], dev


def exp_e3final(out_dir, sel_dir="../results/ctkv_accuracy_study/nlp_lr/e3", seeds=(0, 1, 2, 3, 4)):
    """Pick by pre-specified random (c0), lowest dev NLL@0, lowest dev NLL@50 (ties: lower index); evaluate each
    distinct chosen group once (step 0 = ICL, step 200 = Full CT-KV)."""
    jobs, picks = [], []
    for t in TASKS:
        for seed in seeds:
            groups, _ = candidate_groups(t, seed)
            res = [json.load(open(os.path.join(sel_dir, f"e3sel_{t}_s{seed}_c{c}.json"))) for c in range(8)]
            n0 = [r["dev"]["0"]["nll"] for r in res]
            n50 = [r["dev"]["50"]["nll"] for r in res]
            rule = {"random": 0, "nll0": min(range(8), key=lambda c: (n0[c], c)),
                    "nll50": min(range(8), key=lambda c: (n50[c], c))}
            picks.append(dict(task=t, seed=seed, nll0=n0, nll50=n50, rule=rule))
            for c in sorted(set(rule.values())):
                jobs.append(job(f"e3final_{t}_s{seed}_c{c}", out_dir, t, groups[c], checkpoints=[0, 200],
                                meta=dict(exp="e3_final", seed=seed, cand=c, rules=[k for k, v in rule.items() if v == c])))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(picks, open(os.path.join(out_dir, "e3_picks.json"), "w"), indent=1)
    return jobs


def exp_timing(out_dir):
    """Section 6.1 analogue: deployment time of raw (natural) vs refined (minimal replacement) demonstrations,
    stopping at 100 or 200 steps of the 200-step schedule, three repeats, no evaluation."""
    jobs = []
    for t in [t for t in TASKS if is_clf(t) and len({d["output"] for d in pool(t)}) == 2]:
        for i, s in enumerate(SPLITS):
            nat = with_pool_ids(t, natural(t, s))
            r = repair_and_sham(t, nat, 3000 + i)
            if r is None or r["n_replaced"] == 0:
                continue
            for name, demos in (("raw", nat), ("refined", r["repair"])):
                for stop in (100, 200):
                    for rep in range(3):
                        jobs.append(job(f"time_{t}_s{i}_{name}{stop}_r{rep}", out_dir, t, demos, checkpoints=[],
                                        stop_at=stop, eval_test=False,
                                        meta=dict(exp="timing", seed=i, cond=name, stop=stop, rep=rep)))
    return jobs


EXPERIMENTS = dict(e0=exp_e0, h1=exp_h1, e2=exp_e2, h2=exp_h2, e3=exp_e3, e3final=exp_e3final, timing=exp_timing)

if __name__ == "__main__":
    name, out_dir = sys.argv[1], sys.argv[2]
    jobs = EXPERIMENTS[name](out_dir)
    path = os.path.join(out_dir, f"jobs_{name}.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{len(jobs)} jobs -> {path}")
