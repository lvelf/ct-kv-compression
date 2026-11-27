"""Build the Stage 3 round 1 job lists (one trajectory per task x demonstration seed, arm D) from the frozen Stage 1
manifests in results/ctkv_refinement.

Every job copies the Stage 1 arm-D job verbatim (demo IDs, order and texts, prompt, raw loss, optimizer seed 0, lr,
dropout, schedule, scoring, BBH eval_ids) and changes only id / out (absolute) and adds stage1_id and smoke.
Smoke jobs (part of the 230): boolean_expressions, financial_phrasebank, ai2_arc at seed 1000.

Usage (from context-tuning/): python stage3_jobs.py --stage1 ../results/ctkv_refinement --out ../results/ctkv_stage3_round1
writes OUT/jobs_smoke_{bbh,nlplr}.jsonl and OUT/jobs_{bbh,nlplr}.jsonl (the full lists include the smoke jobs,
which are skipped there once their outputs exist).
"""
import argparse
import json
from pathlib import Path

SMOKE = {("boolean_expressions", 1000), ("financial_phrasebank", 1000), ("ai2_arc", 1000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    nl = [json.loads(l) for l in open(a.stage1 / "nlplr_jobs.jsonl")]
    bb = [json.loads(l) for l in open(a.stage1 / "raw_bbh/jobs_bbh.jsonl")]
    bb = [j for j in bb if j["id"].startswith("generic_bbh_")]
    for fam, rows in (("nlplr", nl), ("bbh", bb)):
        seen, jobs = set(), []
        for b in rows:
            if b["meta"]["arm"] != "D":
                continue
            key = (b["task"], b["meta"]["seed"])
            assert key not in seen, key
            seen.add(key)
            assert b.get("loss") == "raw" and b.get("seed", b.get("optimizer_seed", 0)) == 0, b["id"]
            j = dict(b)
            j["stage1_id"] = b["id"]
            j["id"] = f"s3r1_{b['task']}_D_s{b['meta']['seed']}"
            j["out"] = str(out / f"raw_{fam}" / (j["id"] + ".json"))
            j["smoke"] = key in SMOKE
            j.pop("checkpoints", None)
            j.pop("stop_at", None)
            jobs.append(j)
        jobs.sort(key=lambda j: (not j["smoke"], j["task"], j["meta"]["seed"]))
        for name, sel in (("smoke", [j for j in jobs if j["smoke"]]), ("", jobs)):
            p = out / (f"jobs_{name}_{fam}.jsonl" if name else f"jobs_{fam}.jsonl")
            with open(p, "w") as f:
                for j in sel:
                    f.write(json.dumps(j, sort_keys=True) + "\n")
            print(p, len(sel))


if __name__ == "__main__":
    main()
