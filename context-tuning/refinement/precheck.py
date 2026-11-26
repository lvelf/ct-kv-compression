"""CPU pre-check for the generic-refinement NLP-LR jobs (guideline §3 NLP-LR steps 1-3).

1. Rebuilt pool (from the jobs) vs study_nlplr_jobs.pool(task): same IDs, same input/options/output.
2. Per job: evaluable test IDs under the original 1024-token filter + patience rule (build_eval), demo token
   length, and how many demos GSDataset actually keeps for the CT-KV update (effective shots).
3. Per task: the common test-ID set over all planned arm x seed jobs (frozen denominator for the main table).

Usage (from context-tuning/): python refinement/precheck.py RUN_DIR
Writes RUN_DIR/precheck.json and RUN_DIR/common_test_ids.json.
"""
import hashlib, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer  # noqa: E402

import study_nlplr as R  # noqa: E402
import study_nlplr_jobs as J  # noqa: E402
from data_utils import GSDataset  # noqa: E402


def main(run_dir):
    jobs = [json.loads(l) for l in open(os.path.join(run_dir, "nlplr_jobs.jsonl"))]
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
    tok.pad_token = tok.eos_token

    pool_check = {}
    seen = defaultdict(dict)
    for j in jobs:
        for d in j["demos"]:
            seen[j["task"]][d["pool_id"]] = d
    for task, rows in sorted(seen.items()):
        ref = {r["pool_id"]: r for r in J.pool(task)}
        bad = [i for i, r in rows.items()
               if i not in ref or any(ref[i][k] != r[k] for k in ("input", "options", "output"))]
        pool_check[task] = dict(pool_size=len(ref), used=len(rows), mismatched=bad)
        assert not bad, (task, bad[:5])

    tests = {}
    per_job = []
    for n, j in enumerate(jobs):
        task = j["task"]
        if task not in tests:
            tests[task] = R.load_tests(task)
        rec = dict(id=j["id"], task=task, arm=j["meta"]["arm"], seed=j["meta"]["seed"], family=j["meta"]["family"])
        try:
            prefix, starts, items = R.build_eval(tok, j["demos"], tests[task])
            gs = GSDataset(demonstration_pairs={i: p for i, p in enumerate(j["demos"])}, tokenizer=tok,
                           pad_side="right", past_kv_len=prefix.shape[0])
            ids = sorted({x["test_id"] for x in items})
            rec.update(ok=True, prefix_len=int(prefix.shape[0]), n_test_total=len(tests[task]), n_eval=len(ids),
                       effective_shots=len(gs.parsed_examples), eval_ids=ids)
            if not gs.parsed_examples:
                rec.update(ok=False, reason="no demo usable for the CT-KV update")
            elif not ids:
                rec.update(ok=False, reason="no evaluable test question")
        except ValueError as e:
            rec.update(ok=False, reason=str(e))
        per_job.append(rec)
        if (n + 1) % 60 == 0:
            print(f"{n + 1}/{len(jobs)}", flush=True)

    common = {}
    for task in sorted(tests):
        recs = [r for r in per_job if r["task"] == task and r["ok"]]
        sets = [set(r["eval_ids"]) for r in recs]
        inter = sorted(set.intersection(*sets)) if sets else []
        union = set().union(*sets) if sets else set()
        common[task] = dict(family=recs[0]["family"] if recs else None, n_jobs_ok=len(recs),
                            n_jobs=sum(r["task"] == task for r in per_job), n_test_file=len(tests[task]),
                            n_union=len(union), n_common=len(inter),
                            min_eval=min((len(s) for s in sets), default=0), common_ids=inter)
    out = dict(pool_check=pool_check, jobs=[{k: v for k, v in r.items() if k != "eval_ids"} for r in per_job])
    json.dump(out, open(os.path.join(run_dir, "precheck.json"), "w"), indent=1)
    blob = json.dumps(common, sort_keys=True)
    json.dump(common, open(os.path.join(run_dir, "common_test_ids.json"), "w"), indent=1)
    print("common_test_ids.json sha256", hashlib.sha256(blob.encode()).hexdigest())
    bad = [r for r in per_job if not r["ok"]]
    print(f"jobs ok {len(per_job) - len(bad)}/{len(per_job)}; failed: {[(r['id'], r['reason']) for r in bad]}")
    short = [r for r in per_job if r["ok"] and r["effective_shots"] < 16]
    print(f"jobs with effective shots < 16: {len(short)}")
    for t, c in common.items():
        print(f"{t:28s} {c['family']:4s} jobs {c['n_jobs_ok']}/{c['n_jobs']} test {c['n_test_file']} "
              f"union {c['n_union']} common {c['n_common']} min {c['min_eval']}")


if __name__ == "__main__":
    main(sys.argv[1])
