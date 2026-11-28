"""Accuracy round 2, guideline section 6: dual-cache routing analysis (exploratory).

Reads RUN_DIR/raw (G = arm-D ranking trajectory, A original trajectory for A200, D original for D50), RUN_DIR/raw_dual
(H, R1, R2 ranking trajectories), RUN_DIR/dual_routes.json.gz and RUN_DIR/dual_selection.jsonl. Every cache is read
with the frozen main method (ranking loss, step 50, calibration lambda 0.5, N/A template, its own null scores).

Conditions: A200, D50, G, H, R1, R2, hash_GH (random routing), text_GH (text routing), text_R (R1/R2 text routing),
oracle_GH (correct if either cache is correct; needs both reads and the gold answer, not deployable).
Statistics as in round2_analyze (paired bootstrap, 5,000 draws, seeds and questions within task, RNG seed 0).

Writes RUN_DIR/{dual_coverage.csv, dual_conditions.csv, dual_contrasts.csv, dual_routing.csv, dual_cost.csv,
dual_predictions.jsonl.gz, REPORT_dual_results.md}.
Usage (from context-tuning/): python round2_dual_analyze.py RUN_DIR STAGE1_DIR
"""
import gzip
import json
import os
import sys

import numpy as np

import round2_analyze as A
import round2_cal as C

CONDS = ["A200", "D50", "G", "H", "R1", "R2", "hash_GH", "text_GH", "text_R", "oracle_GH"]
VS = [(c, b) for b in ("A200", "G") for c in CONDS if c not in ("A200", b)] + \
     [("text_GH", "hash_GH"), ("text_GH", "text_R"), ("text_R", "R1"), ("H", "R1"), ("G", "R1"), ("text_GH", "D50")]


def cal_pred(res, lam=A.LAM):
    cp, c0 = res["checkpoints"]["50"], res["checkpoints"]["0"]
    b = C.shift(cp["null_fp32_loss"], c0["null_fp32_loss"])
    idx = res["question_null_idx"]["na"]
    return {q["test_id"]: q["options"][C.argbest(C.calibrated(l, b[i], lam))]
            for q, l, i in zip(res["questions"], cp["fp32_loss"], idx)}


def legacy_pred(res, step):
    return {q["test_id"]: q["options"][int(np.argmin(l))]
            for q, l in zip(res["questions"], res["checkpoints"][str(step)]["legacy_loss"])}


def main(run_dir, stage1):
    common = json.load(open(os.path.join(stage1, "common_test_ids.json")))
    fam = {t: c["family"] for t, c in common.items()}
    routes = json.load(gzip.open(os.path.join(run_dir, "dual_routes.json.gz"), "rt"))
    sel = {(s["task"], s["seed"]): s for s in map(json.loads, open(os.path.join(run_dir, "dual_selection.jsonl")))}
    path = {}
    for name, f in (("A", "jobs_A_orig.jsonl"), ("Do", "jobs_D_orig.jsonl"), ("G", "jobs_D_rank.jsonl")):
        for j in map(json.loads, open(os.path.join(run_dir, f))):
            path[(j["task"], j["meta"]["seed"], name)] = j["out"]
    for j in map(json.loads, open(os.path.join(run_dir, "jobs_dual.jsonl"))):
        path[(j["task"], j["meta"]["seed"], j["meta"]["arm"])] = j["out"]
    tasks = sorted({k[0] for k in path})
    seeds_all = sorted({k[1] for k in path})
    need = ("A", "Do", "G", "H", "R1", "R2")
    data, cov, route_rows, cost_rows = {}, [], [], []
    gz = gzip.open(os.path.join(run_dir, "dual_predictions.jsonl.gz"), "wt")
    for t in tasks:
        ids = list(common[t]["common_ids"])
        good = [s for s in seeds_all if all(os.path.exists(path[(t, s, n)]) for n in need)]
        cov.append(dict(task=t, family=fam[t], seeds_expected=len(seeds_all), seeds_complete=len(good), questions_common=len(ids),
                        **{f"{n}_present": sum(os.path.exists(path[(t, s, n)]) for s in seeds_all) for n in need}))
        if not good:
            continue
        M = np.zeros((len(CONDS), len(good), len(ids)))
        for si, s in enumerate(good):
            res = {n: json.load(open(path[(t, s, n)])) for n in need}
            gold = {q["test_id"]: q["gold"] for q in res["A"]["questions"]}
            p = {"A200": legacy_pred(res["A"], 200), "D50": legacy_pred(res["Do"], 50)}
            for n in ("G", "H", "R1", "R2"):
                p[n] = cal_pred(res[n])
                assert set(ids) <= set(p[n]), (t, s, n)
            r = routes[f"{t}/{s}"]
            rk = {type(ids[0])(k) if not isinstance(ids[0], str) else k: v for k, v in r.items()}
            assert set(ids) <= set(rk), (t, s, "routes")
            ok = {c: np.array([p[c][i] == gold[i] for i in ids], float) for c in ("A200", "D50", "G", "H", "R1", "R2")}
            toH = np.array([rk[i]["text_GH"] == "H" for i in ids])
            toR2 = np.array([rk[i]["text_R"] == "R2" for i in ids])
            hH = np.array([rk[i]["hash"] == 1 for i in ids])
            ok["hash_GH"] = np.where(hH, ok["H"], ok["G"])
            ok["text_GH"] = np.where(toH, ok["H"], ok["G"])
            ok["text_R"] = np.where(toR2, ok["R2"], ok["R1"])
            ok["oracle_GH"] = np.maximum(ok["G"], ok["H"])
            for ci_, c in enumerate(CONDS):
                M[ci_, si] = ok[c]
            m = lambda x, w: float(x[w].mean()) * 100 if w.any() else float("nan")
            x = sel[(t, s)]
            route_rows.append(dict(
                task=t, family=fam[t], seed=s, G_and_H=x["G_and_H"], G_or_H=x["G_or_H"], pool_size=x["pool_size"],
                labels_G=json.dumps(x["labels_G"]), labels_H=json.dumps(x["labels_H"]),
                coverage_target_by_G=x["coverage_target_by_G"], coverage_target_by_H=x["coverage_target_by_H"],
                share_to_H=float(toH.mean()), share_hash_to_H=float(hH.mean()), share_to_R2=float(toR2.mean()),
                acc_G_on_routed_G=m(ok["G"], ~toH), acc_H_on_routed_G=m(ok["H"], ~toH),
                acc_G_on_routed_H=m(ok["G"], toH), acc_H_on_routed_H=m(ok["H"], toH),
                G_and_H_agree=float((ok["G"] == ok["H"]).mean()) * 100,
                only_G_right=float(((ok["G"] == 1) & (ok["H"] == 0)).mean()) * 100,
                only_H_right=float(((ok["G"] == 0) & (ok["H"] == 1)).mean()) * 100))
            for n in ("G", "H", "R1", "R2"):
                cp = res[n]["checkpoints"]
                cost_rows.append(dict(task=t, family=fam[t], seed=s, cache=n, prefill_s=res[n]["timing"]["prefill_s"],
                                      adapt_s_50=cp["50"]["cumulative_adapt_s"], real_eval_s_50=cp["50"].get("real_eval_s"),
                                      null_eval_s_0=cp["0"]["null_eval_s"], null_eval_s_50=cp["50"]["null_eval_s"],
                                      peak_mem_mb=res[n]["timing"]["peak_mem_mb"], n_questions=res[n]["n_test"]))
            for qi, i in enumerate(ids):
                gz.write(json.dumps(dict(task=t, seed=s, qid=i, gold=gold[i], route_text=rk[i]["text_GH"],
                                         route_text_R=rk[i]["text_R"], route_hash="H" if rk[i]["hash"] else "G",
                                         pred={c: p[c][i] for c in ("A200", "D50", "G", "H", "R1", "R2")})) + "\n")
        data[t] = dict(M=M, missing=1 - len(good) / len(seeds_all))
    gz.close()
    k = {c: i for i, c in enumerate(CONDS)}
    rng = np.random.default_rng(0)
    for t, d in data.items():
        _, S, Q = d["M"].shape
        ws, wq = A.boot_weights(S, rng), A.boot_weights(Q, rng)
        d["draws"] = (np.einsum("csq,bq->cbs", d["M"], wq) / Q * 100 * ws[None]).sum(-1) / S
        d["point"] = d["M"].mean((1, 2)) * 100
    cond_rows, con_rows = [], []
    for scope in ("all", "clf", "mc"):
        ts = [t for t in data if scope == "all" or fam[t] == scope]
        if not ts:
            continue
        point, draws = np.mean([data[t]["point"] for t in ts], 0), np.mean([data[t]["draws"] for t in ts], 0)
        for c in CONDS:
            lo, hi = A.ci(draws[k[c]])
            cond_rows.append(dict(scope=scope, condition=c, tasks=len(ts), accuracy=point[k[c]], ci_lo=lo, ci_hi=hi))
        for a, b in VS:
            d = draws[k[a]] - draws[k[b]]
            pt = [data[t]["point"][k[a]] - data[t]["point"][k[b]] for t in ts]
            con_rows.append(dict(scope=scope, task="ALL", a=a, b=b, acc_a=point[k[a]], acc_b=point[k[b]], diff=point[k[a]] - point[k[b]],
                                 ci_lo=A.ci(d)[0], ci_hi=A.ci(d)[1],
                                 fixed=sum(int(((data[t]["M"][k[a]] == 1) & (data[t]["M"][k[b]] == 0)).sum()) for t in ts),
                                 broken=sum(int(((data[t]["M"][k[a]] == 0) & (data[t]["M"][k[b]] == 1)).sum()) for t in ts),
                                 tasks_better=sum(x > 1e-9 for x in pt), tasks_worse=sum(x < -1e-9 for x in pt)))
    for t, d in data.items():
        for a, b in VS:
            dd = d["draws"][k[a]] - d["draws"][k[b]]
            con_rows.append(dict(scope=fam[t], task=t, a=a, b=b, acc_a=d["point"][k[a]], acc_b=d["point"][k[b]],
                                 diff=d["point"][k[a]] - d["point"][k[b]], ci_lo=A.ci(dd)[0], ci_hi=A.ci(dd)[1]))
    A.wcsv(os.path.join(run_dir, "dual_coverage.csv"), cov)
    A.wcsv(os.path.join(run_dir, "dual_conditions.csv"), cond_rows)
    A.wcsv(os.path.join(run_dir, "dual_contrasts.csv"), con_rows)
    A.wcsv(os.path.join(run_dir, "dual_routing.csv"), route_rows)
    A.wcsv(os.path.join(run_dir, "dual_cost.csv"), cost_rows)

    L = ["# Dual-cache routing (guideline §6) — generated tables, exploratory", "",
         f"Units complete: {sum(r['seeds_complete'] for r in cov)} / {sum(r['seeds_expected'] for r in cov)} task × seed.", "",
         "| scope | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for scope in ("all", "clf", "mc"):
        r = {x["condition"]: x for x in cond_rows if x["scope"] == scope}
        if r:
            L.append(f"| {scope} | " + " | ".join(f"{r[c]['accuracy']:.2f}" for c in CONDS) + " |")
    L += ["", "| scope | contrast | diff | 95% CI | fixed / broken | tasks better / worse |", "|---|---|---|---|---|---|"]
    for x in con_rows:
        if x["task"] == "ALL":
            L.append(f"| {x['scope']} | {x['a']} − {x['b']} | {x['diff']:+.2f} | [{x['ci_lo']:+.2f}, {x['ci_hi']:+.2f}] | "
                     f"{x['fixed']} / {x['broken']} | {x['tasks_better']} / {x['tasks_worse']} |")
    if route_rows:
        f = lambda key: float(np.nanmean([r[key] for r in route_rows]))
        L += ["", "Routing diagnostics (mean over task × seed):", "",
              f"- |G ∩ H| {f('G_and_H'):.2f}, |G ∪ H| {f('G_or_H'):.2f}; share routed to H {f('share_to_H'):.3f} (hash {f('share_hash_to_H'):.3f}), to R2 {f('share_to_R2'):.3f}.",
              f"- Questions routed to G: G {f('acc_G_on_routed_G'):.2f}, H {f('acc_H_on_routed_G'):.2f}. Questions routed to H: G {f('acc_G_on_routed_H'):.2f}, H {f('acc_H_on_routed_H'):.2f}.",
              f"- G and H agree in correctness on {f('G_and_H_agree'):.2f}% of questions; only G right {f('only_G_right'):.2f}%, only H right {f('only_H_right'):.2f}%.",
              f"- Target-subset coverage: by G {f('coverage_target_by_G'):.3f}, by H {f('coverage_target_by_H'):.3f}."]
        med = lambda c, key: float(np.median([r[key] for r in cost_rows if r["cache"] == c]))
        L += ["", "Cost (median per task × seed): " + "; ".join(
            f"{c}: adaptation to step 50 {med(c, 'adapt_s_50'):.2f} s, question scoring {med(c, 'real_eval_s_50'):.2f} s, peak {med(c, 'peak_mem_mb'):.0f} MiB"
            for c in ("G", "H"))]
    open(os.path.join(run_dir, "REPORT_dual_results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12 + 3 * len(VS)]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
