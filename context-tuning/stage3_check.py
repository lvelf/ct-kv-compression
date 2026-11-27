"""Stage 3 round 1 smoke-job technical checks (gate for the full panel; accuracy never decides).

For each smoke output: all runner checks true (interpolation weight 0 / 1 bitwise endpoints and identical
predictions, caches unchanged by evaluation, question IDs aligned, five conditions present), optimizer / scheduler
step counts = T, common-test coverage. Against the Stage 1 production baseline of the same job: `icl` must match the
Stage 1 step-0 predictions (no training involved); `full` is compared with the Stage 1 endpoint and with the Stage 1
rerun spread (BF16 backward is not deterministic), flagged if the flip rate exceeds 5% of the questions.
Writes OUT/checks/smoke_check.json; exits 1 if any technical check fails.

Usage (from context-tuning/): python stage3_check.py ../results/ctkv_stage3_round1 ../results/ctkv_refinement
"""
import json
import os
import sys


def main(run_dir, stage1):
    common = json.load(open(os.path.join(stage1, "common_test_ids.json")))
    report, ok = {}, True
    for fam, fn in (("bbh", "jobs_smoke_bbh.jsonl"), ("nlplr", "jobs_smoke_nlplr.jsonl")):
        for job in map(json.loads, open(os.path.join(run_dir, fn))):
            r = {"id": job["id"]}
            if not os.path.exists(job["out"]):
                r["missing"] = True
                report[job["id"]] = r
                ok = False
                continue
            res = json.load(open(job["out"]))
            T = job.get("steps", 16) if fam == "bbh" else job.get("epochs", 200)
            r["runner_checks"] = res["checks"]
            bad = [k for k, v in res["checks"].items() if v is False]
            r["optimizer_steps"], r["scheduler_steps"] = res["optimizer_steps"], res["scheduler"]["scheduler_steps"]
            if res["optimizer_steps"] != T or res["scheduler"]["scheduler_steps"] != T:
                bad.append("step_count")
            s1 = json.load(open(os.path.join(stage1, f"raw_{fam}", job["stage1_id"] + ".json")))
            if fam == "bbh":
                get = lambda rows: {p["row_id"]: p["prediction"] for p in rows}
                cur = {c: get(v["predictions"]) for c, v in res["conditions"].items()}
                ref = {"icl": get(s1["checkpoints"]["0"]["predictions"]),
                       "full": get(s1["checkpoints"][str(T)]["predictions"])}
                reps = []
                for k in (1, 2):
                    p = os.path.join(stage1, "raw_bbh_repeat", f"{job['stage1_id']}_rep{k}.json")
                    if os.path.exists(p):
                        reps.append(get(json.load(open(p))["checkpoints"][str(T)]["predictions"]))
                r["stage1_rerun_full_flips_vs_stage1"] = [sum(x[i] != ref["full"][i] for i in x) for x in reps]
                r["accuracy"] = {c: v["accuracy"] for c, v in res["conditions"].items()}
                r["stage1_accuracy"] = {"icl": s1["checkpoints"]["0"]["accuracy"],
                                        "full": s1["checkpoints"][str(T)]["accuracy"]}
            else:
                get = lambda rows: {x[0]: x[1] for x in rows}
                cur = {c: get(v["predictions"]) for c, v in res["conditions"].items()}
                ref = {"icl": get(s1["checkpoints"]["0"]["predictions"]),
                       "full": get(s1["checkpoints"][str(T)]["predictions"])}
                ids = set(common[job["task"]]["common_ids"])
                r["common_coverage"] = {c: len(ids & set(v)) / len(ids) for c, v in cur.items()}
                if min(r["common_coverage"].values()) < 1:
                    bad.append("common_coverage")
                r["score"] = {c: v["score"] for c, v in res["conditions"].items()}
                r["stage1_score"] = {"icl": s1["checkpoints"]["0"]["score"], "full": s1["checkpoints"][str(T)]["score"]}
            for c in ("icl", "full"):
                r[f"{c}_flips_vs_stage1"] = sum(cur[c][i] != ref[c][i] for i in ref[c])
                r[f"{c}_n"] = len(ref[c])
            if r["icl_flips_vs_stage1"] != 0:
                bad.append("icl_differs_from_stage1")
            r["full_flip_rate_flag"] = r["full_flips_vs_stage1"] > 0.05 * r["full_n"]
            if r["full_flip_rate_flag"]:
                bad.append("full_flip_rate_over_5pct")
            r["geometry"] = res["geometry"]
            r["timing"] = res["timing"]
            r["failed"] = bad
            ok &= not bad
            report[job["id"]] = r
    report["ALL_TECHNICAL_CHECKS_PASS"] = ok
    os.makedirs(os.path.join(run_dir, "checks"), exist_ok=True)
    json.dump(report, open(os.path.join(run_dir, "checks", "smoke_check.json"), "w"), indent=1)
    for k, r in report.items():
        if isinstance(r, dict):
            print(k, "failed:", r.get("failed"), "icl flips", r.get("icl_flips_vs_stage1"), "full flips",
                  r.get("full_flips_vs_stage1"), "/", r.get("full_n"))
    print("ALL_TECHNICAL_CHECKS_PASS", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
