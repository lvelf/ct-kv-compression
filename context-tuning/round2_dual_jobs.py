"""Accuracy round 2, guideline section 6: build the second demonstration cache H and the random dual-cache control.

Per task x demonstration seed, using only the Stage 1 training pool (study_nlplr_jobs.pool: union of the five MetaICL
16-shot train files minus test overlaps) and, for routing, only the test question text:

  G        the Stage 1 arm-D 16 demonstrations (existing round 2 D ranking trajectory).
  H        TF-IDF (sklearn TfidfVectorizer defaults, fitted on the pool inputs, L2-normalised). For every pool input,
           coverage = max cosine to G's demonstrations; the worst-covered half of the pool is the target subset. H is 16
           pool examples chosen greedily, each step taking the example that most increases the subset's mean max-cosine
           to H (ties: lowest pool_id). Prompt order = selection order.
  R1, R2   random control with the same labelled budget: two random 16-sets from the pool with |R1 ∪ R2| = |G ∪ H|
           (random.Random(f"{task}/{seed}/dualrand")), prompt order as drawn.
  routing  a question goes to the cache whose 3 most similar demonstrations have the higher mean cosine to the question
           text (ties: the first cache, G or R1). Random routing: sha256(f"{task}/{seed}/{qid}") parity (even: G).

Writes OUT/jobs_dual.jsonl (H, R1, R2: ranking loss, 50 steps of the 200-step schedule, null scores at 0 and 50),
OUT/dual_selection.jsonl and OUT/dual_routes.json.gz.
Usage (from context-tuning/): python round2_dual_jobs.py --out ../results/ctkv_accuracy_round2
"""
import argparse
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import study_nlplr_jobs as J
from study_nlplr import load_tests


def top3(q, d):
    """Mean of the 3 largest cosines of every query row to the demonstration rows."""
    s = (q @ d.T).toarray()
    return np.sort(s, axis=1)[:, -3:].mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve()
    gjobs = [json.loads(l) for l in open(out / "jobs_D_rank.jsonl")]
    jobs, sel, routes = [], [], {}
    for g in sorted(gjobs, key=lambda j: (j["task"], j["meta"]["seed"])):
        task, seed = g["task"], g["meta"]["seed"]
        pool = J.pool(task)
        by_id = {d["pool_id"]: d for d in pool}
        gid = [d["pool_id"] for d in g["demos"]]
        assert all(i in by_id and by_id[i]["input"] == d["input"] for i, d in zip(gid, g["demos"])), (task, seed)
        vec = TfidfVectorizer().fit([d["input"] for d in pool])
        X = vec.transform([d["input"] for d in pool])          # rows follow pool order (pool_id = row index)
        S = (X @ X.T).toarray()
        cover = S[:, gid].max(1)
        order = sorted(range(len(pool)), key=lambda i: (cover[i], i))
        target = order[:len(pool) // 2]
        hid, best = [], np.zeros(len(target))
        for _ in range(16):
            gains = [(np.maximum(best, S[target, c]).mean(), -c) for c in range(len(pool)) if c not in hid]
            c = -max(gains)[1]
            hid.append(c)
            best = np.maximum(best, S[target, c])
        rng = random.Random(f"{task}/{seed}/dualrand")
        union = len(set(gid) | set(hid))
        r1 = rng.sample(range(len(pool)), 16)
        r2 = rng.sample(r1, 32 - union) + rng.sample([i for i in range(len(pool)) if i not in r1], union - 16)
        rng.shuffle(r2)
        assert len(set(r1) | set(r2)) == union and len(set(r2)) == 16
        sets = {"G": gid, "H": hid, "R1": r1, "R2": r2}
        tests = load_tests(task)
        Q = vec.transform([t["input"] for _, t in tests])
        score = {k: top3(Q, X[v]) for k, v in sets.items()}
        route = {}
        for n, (qid, _) in enumerate(tests):
            h = int(hashlib.sha256(f"{task}/{seed}/{qid}".encode()).hexdigest(), 16) % 2
            route[qid] = dict(text_GH="H" if score["H"][n] > score["G"][n] else "G",
                              text_R="R2" if score["R2"][n] > score["R1"][n] else "R1", hash=h)
        routes[f"{task}/{seed}"] = route
        lab = lambda ids: dict(Counter(by_id[i]["output"] for i in ids)) if J.is_clf(task) else None
        sel.append(dict(task=task, seed=seed, pool_size=len(pool), G=gid, H=hid, R1=r1, R2=r2,
                        G_and_H=len(set(gid) & set(hid)), G_or_H=union, degenerate=set(gid) == set(hid),
                        labels_G=lab(gid), labels_H=lab(hid), labels_R1=lab(r1), labels_R2=lab(r2),
                        target_subset=target, coverage_target_by_G=float(cover[target].mean()),
                        coverage_target_by_H=float(best.mean()),
                        share_routed_to_H=float(np.mean([r["text_GH"] == "H" for r in route.values()])),
                        share_routed_to_R2=float(np.mean([r["text_R"] == "R2" for r in route.values()]))))
        for name in ("H", "R1", "R2"):
            j = dict(task=task, seed=g["seed"], epochs=200, loss="raw", rank_weight=1.0, checkpoints=[0, 50], stop_at=50,
                     eval_real=False, null_templates=["na", "dots"], smoke=False,
                     demos=[{k: by_id[i][k] for k in ("input", "output", "options", "pool_id")} for i in sets[name]],
                     meta=dict(g["meta"], arm=name, ids=sets[name]))
            j["id"] = f"r2dual_{task}_{name}_s{seed}_rank"
            j["out"] = str(out / "raw_dual" / (j["id"] + ".json"))
            jobs.append(j)
    with open(out / "jobs_dual.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j, sort_keys=True) + "\n")
    with open(out / "dual_selection.jsonl", "w") as f:
        for s in sel:
            f.write(json.dumps(s) + "\n")
    with gzip.open(out / "dual_routes.json.gz", "wt") as f:
        json.dump(routes, f)
    print(len(jobs), "jobs;", "degenerate", sum(s["degenerate"] for s in sel), "mean |G∩H|", np.mean([s["G_and_H"] for s in sel]),
          "mean share to H", np.mean([s["share_routed_to_H"] for s in sel]), "pool sizes", sorted({s["pool_size"] for s in sel}))


if __name__ == "__main__":
    main()
