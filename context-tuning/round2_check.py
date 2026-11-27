"""Accuracy round 2 smoke gate (technical only; accuracy never decides). Writes OUT/checks/smoke_check.json and
OUT/checks/null_samples.json; exits 1 if a technical check fails.

Usage (from context-tuning/): python round2_check.py ../results/ctkv_accuracy_round2 ../results/ctkv_refinement ../results/ctkv_stage3_round1
"""
import hashlib
import json
import os
import sys

import numpy as np

import round2_cal as C
from round2_nlplr import NULL_TEMPLATES, null_input

PERM_TOL = 0.25   # batch-composition noise of the BF16 evaluator was measured at up to 0.09 in loss


def main(run_dir, stage1, stage3):
    common = json.load(open(os.path.join(stage1, "common_test_ids.json")))
    report, ok = {"calibration_unit_tests": C.self_test()}, True
    samples = {}
    for task in sorted(common):
        tests = [json.loads(l) for l in open(f"metaicl-data/data/{task}/{task}_16_87_test.jsonl")]
        for t, filler in NULL_TEMPLATES.items():
            u = sorted({null_input(task, x["input"], filler) for x in tests})
            samples.setdefault(task, {})[t] = [dict(text=s, sha256=hashlib.sha256(s.encode()).hexdigest()) for s in u]
    os.makedirs(os.path.join(run_dir, "checks"), exist_ok=True)
    json.dump(samples, open(os.path.join(run_dir, "checks", "null_samples.json"), "w"), indent=1, ensure_ascii=False)
    for job in map(json.loads, open(os.path.join(run_dir, "jobs_smoke.jsonl"))):
        r, bad = {"id": job["id"]}, []
        report[job["id"]] = r
        if not os.path.exists(job["out"]):
            r["missing"], ok = True, False
            continue
        res = json.load(open(job["out"]))
        ch = res["checks"]
        r["runner_checks"] = ch
        bad += [k for k, v in ch.items() if v is False]
        if ch["rank_only_grad_nonzero_tensors"] < 1 or not ch["rank_only_L_finite"]:
            bad.append("rank_gradient")
        if ch["scorer_legacy_equals_option_losses_max_abs"] != 0:
            bad.append("legacy_scorer_differs_from_option_losses")
        if ch["scorer_tensor_vs_fp32_eval_max_abs"] > PERM_TOL or ch["permuted_candidates_max_abs_dscore"] > PERM_TOL:
            bad.append("scorer_tolerance")
        if res["optimizer_steps"] != job["stop_at"] or res["scheduler"]["scheduler_steps"] != job["stop_at"]:
            bad.append("step_count")
        qids = [q["test_id"] for q in res["questions"]]
        ids = set(common[job["task"]]["common_ids"])
        r["common_coverage"] = len(ids & set(qids)) / len(ids)
        if r["common_coverage"] < 1:
            bad.append("common_coverage")
        cps = res["checkpoints"]
        n0 = cps["0"]["null_fp32_loss"]
        r["lambda0_fp32_vs_legacy_pred_mismatch"], r["n_questions"] = {}, len(qids)
        for k, cp in cps.items():
            if "fp32_loss" not in cp:
                continue
            b = C.shift(cp["null_fp32_loss"], n0)
            if k == "0" and float(np.abs(b).max()) != 0.0:
                bad.append("shift_nonzero_at_C0")
            mism = 0
            for qi, q in enumerate(res["questions"]):
                bq = b[res["question_null_idx"]["na"][qi]]
                leg = int(np.argmin(cp["legacy_loss"][qi]))
                p0 = C.argbest(C.calibrated(cp["fp32_loss"][qi], bq, 0.0))
                if p0 != C.argbest(-np.asarray(cp["fp32_loss"][qi])):
                    bad.append("lambda0_changes_prediction")
                mism += p0 != leg
            r["lambda0_fp32_vs_legacy_pred_mismatch"][k] = mism
            r[f"shift_abs_max@{k}"] = float(np.abs(b).max())
        if res["rank_weight"] == 0:   # original trajectory: compare with the Stage 3 run of the same job
            s3 = json.load(open(os.path.join(stage3, "raw_nlplr", f"s3r1_{job['task']}_D_s{job['meta']['seed']}.json")))
            for k, cond in (("0", "icl"), ("50", "early_quarter"), ("200", "full")):
                ref = {x[0]: x[1] for x in s3["conditions"][cond]["predictions"]}
                cur = {q["test_id"]: q["options"][int(np.argmin(l))] for q, l in zip(res["questions"], cps[k]["legacy_loss"])}
                flips = sum(cur[i] != ref[i] for i in ref)
                r[f"flips_vs_stage3@{k}"] = [flips, len(ref)]
                if (k == "0" and flips != 0) or flips > 0.05 * len(ref):
                    bad.append(f"differs_from_stage3@{k}")
        else:
            r["rank_loss_first_last"] = [res["rank_losses"][0], res["rank_losses"][-1]]
            r["rank_candidates"] = res["rank_candidates"]
        r["timing"] = res["timing"]
        r["failed"] = sorted(set(bad))
        ok &= not bad
    report["ALL_TECHNICAL_CHECKS_PASS"] = ok
    json.dump(report, open(os.path.join(run_dir, "checks", "smoke_check.json"), "w"), indent=1)
    for k, r in report.items():
        if isinstance(r, dict):
            print(k, "failed:", r.get("failed"), {a: b for a, b in r.items() if a.startswith(("flips", "lambda0", "shift"))})
    print("ALL_TECHNICAL_CHECKS_PASS", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
