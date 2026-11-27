"""Accuracy round 2 analysis (NLP-LR): answer-shift calibration and candidate ranking loss against original CT-KV.

Reads RUN_DIR/raw/*.json (round2_nlplr.py outputs; per arm x task x seed one original-loss and one ranking-loss
trajectory), RUN_DIR/jobs_{A,D}_{orig,rank}.jsonl (expected panel) and STAGE1/common_test_ids.json. Conditions are
derived offline from the stored per-option losses (see PLAN.md); calibration uses round2_cal.

Statistics: paired on task x seed number x question (frozen common IDs); seed mean per task; tasks equal weight inside
classification (12) and multiple choice (9); total = (12 clf + 9 mc) / 21. 5,000 paired bootstrap draws (seeds and
questions resampled within each task, shared by all conditions). Only seeds whose four trajectories are all present
are used; coverage.csv lists the rest as pending, and conditions.csv has a column that counts missing units as wrong.

Writes RUN_DIR/{coverage.csv, conditions.csv, contrasts.csv, flip_groups.csv, label_shift.csv, cost.csv,
predictions.jsonl.gz, REPORT_results.md}.
Usage (from context-tuning/): python round2_analyze.py RUN_DIR STAGE1_DIR
"""
import csv
import glob
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

import round2_cal as C

B = 5000
LAM = 0.5
TRAJ = [("A", "orig"), ("A", "rank"), ("D", "orig"), ("D", "rank")]
MAIN = ("D50_rank_cal", "A200")
CONTRASTS = [
    ("D50_rank_cal", "A200", "MAIN: D50_rank_cal − A200"),
    ("D50", "A200", "D50 − A200"), ("D50_cal", "A200", "D50_cal − A200"), ("D50_rank", "A200", "D50_rank − A200"),
    ("D_rankN_cal", "A200", "equal GPU time: D_rank_cal@N − A200"), ("D_rankN", "A200", "equal GPU time: D_rank@N − A200"),
    ("D200", "A200", "D200 − A200"), ("D0", "A200", "D0 − A200"), ("A0", "A200", "A0 − A200"),
    ("D50_cal", "D50", "D50_cal − D50"), ("D50_rank", "D50", "D50_rank − D50"), ("D50_rank_cal", "D50", "D50_rank_cal − D50"),
    ("D50_rank_cal", "D50_rank", "D50_rank_cal − D50_rank"), ("D_rankN_cal", "D50", "D_rank_cal@N − D50"),
    ("D_rankN", "D50_rank", "D_rank@N − D50_rank"),
    ("A50_cal", "A50", "A50_cal − A50"), ("A200_cal", "A200", "A200_cal − A200"), ("D200_cal", "D200", "D200_cal − D200"),
    ("A50_rank", "A50", "A50_rank − A50"), ("A50_rank_cal", "A50", "A50_rank_cal − A50"),
    ("A50_rank_cal", "A200", "A50_rank_cal − A200"),
    ("D50", "A50", "selection: D50 − A50"), ("D50_cal", "A50_cal", "selection: D50_cal − A50_cal"),
    ("D50_rank", "A50_rank", "selection: D50_rank − A50_rank"),
    ("D50_rank_cal", "A50_rank_cal", "selection: D50_rank_cal − A50_rank_cal"),
    ("D50_rank_cal", "D0", "D50_rank_cal − D0"), ("D50_rank_cal", "A0", "D50_rank_cal − A0"),
    ("D50_rank_cal", "D200", "D50_rank_cal − D200"), ("D50_rank_cal", "A200_cal", "D50_rank_cal − A200_cal"),
    ("D50_cal_l0", "D50", "sensitivity λ=0 (FP32 scores, no correction) − D50"),
    ("D50_cal_l1", "D50", "sensitivity λ=1: D50_cal − D50"), ("D50_rank_cal_l1", "D50_rank", "sensitivity λ=1: D50_rank_cal − D50_rank"),
    ("D50_rank_cal_l1", "A200", "sensitivity λ=1: D50_rank_cal − A200"),
    ("D50_cal_dots", "D50", "sensitivity null '…': D50_cal − D50"),
    ("D50_rank_cal_dots", "A200", "sensitivity null '…': D50_rank_cal − A200"),
    ("D_mix", "D50", "D_mix (0.5 s_D0 + 0.5 s_D50, two cache reads) − D50"), ("D_mix", "A200", "D_mix − A200"),
]
INTERACTION = ("D50_rank_cal", "D50_rank", "D50_cal", "D50")
FLIP_CONDS = ["D50_cal", "D50_rank", "D50_rank_cal", "D_rankN", "D_rankN_cal", "D200", "D_mix"]


def derive(orig, rank, budget_step):
    """{condition: (predictions list, steps note)} for one arm x task x seed from its two trajectories."""
    out = {}
    q = orig["questions"]
    assert [x["test_id"] for x in q] == [x["test_id"] for x in rank["questions"]]

    def legacy(res, step):
        return [x["options"][int(np.argmin(l))] for x, l in zip(res["questions"], res["checkpoints"][str(step)]["legacy_loss"])]

    def cal(res, step, lam, tmpl="na"):
        cp, c0 = res["checkpoints"][str(step)], res["checkpoints"]["0"]
        b = C.shift(cp["null_fp32_loss"], c0["null_fp32_loss"])
        idx = res["question_null_idx"][tmpl]
        return [x["options"][C.argbest(C.calibrated(l, b[i], lam))] for x, l, i in zip(res["questions"], cp["fp32_loss"], idx)]

    for s in (0, 50, 200):
        out[f"{s}"] = legacy(orig, s)
    for s in (50, 200):
        out[f"{s}_cal"] = cal(orig, s, LAM)
        out[f"{s}_cal_l0"], out[f"{s}_cal_l1"] = cal(orig, s, 0.0), cal(orig, s, 1.0)
        out[f"{s}_cal_dots"] = cal(orig, s, LAM, "dots")
    out["50_rank"] = legacy(rank, 50)
    out["50_rank_cal"] = cal(rank, 50, LAM)
    out["50_rank_cal_l0"], out["50_rank_cal_l1"] = cal(rank, 50, 0.0), cal(rank, 50, 1.0)
    out["50_rank_cal_dots"] = cal(rank, 50, LAM, "dots")
    out["_rankN"] = legacy(rank, budget_step)
    out["_rankN_cal"] = cal(rank, budget_step, LAM)
    l0, l50 = orig["checkpoints"]["0"]["fp32_loss"], orig["checkpoints"]["50"]["fp32_loss"]
    out["_mix"] = [x["options"][C.argbest(-0.5 * (np.asarray(a) + np.asarray(b)))] for x, a, b in zip(q, l0, l50)]
    return out


def macro_f1(pred, gold):
    f = []
    for lab in sorted(set(gold)):
        tp = sum(p == lab and g == lab for p, g in zip(pred, gold))
        pp, gg = sum(p == lab for p in pred), sum(g == lab for g in gold)
        pr, rc = (tp / pp if pp else 0.0), (tp / gg if gg else 0.0)
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f))


def boot_weights(n, rng):
    idx = rng.integers(0, n, size=(B, n))
    return np.bincount((idx + np.arange(B)[:, None] * n).ravel(), minlength=B * n).reshape(B, n).astype(float)


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def wcsv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(run_dir, stage1):
    common = json.load(open(os.path.join(stage1, "common_test_ids.json")))
    fam = {t: c["family"] for t, c in common.items()}
    budget = json.load(open(os.path.join(run_dir, "rank_budget.json")))
    expected = {}
    for arm, kind in TRAJ:
        for j in map(json.loads, open(os.path.join(run_dir, f"jobs_{arm}_{kind}.jsonl"))):
            expected[(j["task"], j["meta"]["seed"], arm, kind)] = j["out"]
    tasks = sorted({k[0] for k in expected})
    seeds_all = sorted({k[1] for k in expected})
    raw, status = {}, {}
    for key, path in expected.items():
        if os.path.exists(path):
            try:
                raw[key] = json.load(open(path))
                status[key] = "ok"
            except Exception as e:  # corrupt output: reported, rerun by deleting the file
                status[key] = f"failed: {type(e).__name__}"
        else:
            status[key] = "pending"

    # ---------------- conditions per task: correctness matrices [cond][seed][question]
    conds, data, cov_rows, cost_rows, shift_rows, pred_rows = None, {}, [], [], [], []
    gz = gzip.open(os.path.join(run_dir, "predictions.jsonl.gz"), "wt")
    for t in tasks:
        ids = list(common[t]["common_ids"])
        good = [s for s in seeds_all if all(status[(t, s, a, k)] == "ok" for a, k in TRAJ)]
        cov_rows.append(dict(task=t, family=fam[t], seeds_expected=len(seeds_all), seeds_complete=len(good),
                             questions_common=len(ids), budget_steps=budget[t]["budget_steps"],
                             **{f"{a}_{k}_{st}": sum(status[(t, s, a, k)].split(":")[0] == st for s in seeds_all)
                                for a, k in TRAJ for st in ("ok", "pending", "failed")}))
        per = {}
        for s in good:
            cur = {}
            for arm in ("A", "D"):
                o, r = raw[(t, s, arm, "orig")], raw[(t, s, arm, "rank")]
                d = derive(o, r, budget[t]["budget_steps"])
                pos = {q["test_id"]: i for i, q in enumerate(o["questions"])}
                assert set(ids) <= set(pos), (t, s, arm, "common IDs missing")
                gold = [o["questions"][pos[i]]["gold"] for i in ids]
                for name, p in d.items():
                    cur[arm + name] = [p[pos[i]] for i in ids]
                cur["_gold"] = gold
                cur["_nopt"] = [len(o["questions"][pos[i]]["options"]) for i in ids]
                # cost
                for res, kind in ((o, "orig"), (r, "rank")):
                    cps = res["checkpoints"]
                    cost_rows.append(dict(task=t, family=fam[t], arm=arm, seed=s, trajectory=kind, stop_at=res["stop_at"],
                                          prefill_s=res["timing"]["prefill_s"], adapt_s_total=res["timing"]["adapt_s"],
                                          step_s_median=res["timing"]["step_s_median"], peak_mem_mb=res["timing"]["peak_mem_mb"],
                                          n_questions=res["n_test"], n_real_items=sum(len(q["options"]) for q in res["questions"]),
                                          n_null_items_per_template=len(res["null_items"]) // len(res["question_null_idx"]),
                                          n_rank_candidates=res.get("n_rank_candidates_total"),
                                          **{f"adapt_s@{k}": v["cumulative_adapt_s"] for k, v in cps.items()},
                                          **{f"real_eval_s@{k}": v.get("real_eval_s") for k, v in cps.items()},
                                          **{f"null_eval_s@{k}": v["null_eval_s"] for k, v in cps.items()}))
                # label shift (classification: fixed label set)
                if fam[t] == "clf":
                    labels = sorted({x for q in o["questions"] for x in q["options"]})
                    demo = Counter(o["demo_labels"])
                    pool = (o["meta"].get("pool_counts") or {})
                    for res, kind, step in ((o, "orig", 50), (o, "orig", 200), (r, "rank", 50)):
                        nul = {x[2]: i for i, x in enumerate(res["null_items"]) if x[0] == "na"}
                        b = C.shift(res["checkpoints"][str(step)]["null_fp32_loss"], res["checkpoints"]["0"]["null_fp32_loss"])
                        bl = np.array([b[nul[l]] for l in labels])
                        p0 = cur[arm + "0"]
                        pt = cur[arm + (f"{step}" if kind == "orig" else "50_rank")]
                        pc = cur[arm + (f"{step}_cal" if kind == "orig" else "50_rank_cal")]
                        for l, bv in zip(labels, bl - bl.mean()):
                            g = [x == l for x in gold]
                            rec = lambda p: float(np.mean([x == l for x, y in zip(p, g) if y])) if any(g) else float("nan")
                            shift_rows.append(dict(
                                task=t, arm=arm, seed=s, trajectory=kind, step=step, label=l,
                                demo_share=demo[l] / len(o["demo_labels"]),
                                pool_share=(pool.get(l, 0) / sum(pool.values())) if pool else float("nan"),
                                gold_share=float(np.mean(g)), centred_shift=float(bv),
                                pred_share_C0=float(np.mean([x == l for x in p0])),
                                pred_share_Ct=float(np.mean([x == l for x in pt])),
                                pred_share_Ct_cal=float(np.mean([x == l for x in pc])),
                                recall_C0=rec(p0), recall_Ct=rec(pt), recall_Ct_cal=rec(pc)))
                # per-question output (rounded only here, after all decisions were made)
                for res, kind in ((o, "orig"), (r, "rank")):
                    for qi, q in enumerate(res["questions"]):
                        if q["test_id"] not in pos or q["test_id"] not in set(ids):
                            continue
                        row = dict(task=t, family=fam[t], arm=arm, seed=s, loss=kind, job=res["id"], qid=q["test_id"],
                                   gold=q["gold"], options=q["options"], steps={})
                        for k, cp in res["checkpoints"].items():
                            e = {}
                            if "fp32_loss" in cp:
                                e["s"] = [round(-x, 5) for x in cp["fp32_loss"][qi]]
                                e["s_legacy"] = [round(-x, 5) for x in cp["legacy_loss"][qi]]
                            for tm, idx in res["question_null_idx"].items():
                                e[f"s_null_{tm}"] = [round(-cp["null_fp32_loss"][i], 5) for i in idx[qi]]
                            row["steps"][k] = e
                        gz.write(json.dumps(row) + "\n")
            per[s] = cur
        if not good:
            continue
        names = [k for k in per[good[0]] if not k.startswith("_")]
        conds = conds or names
        gold = per[good[0]]["_gold"]
        M = np.array([[[int(p == g) for p, g in zip(per[s][c], per[s]["_gold"])] for s in good] for c in conds], dtype=float)
        f1 = {c: float(np.mean([macro_f1(per[s][c], per[s]["_gold"]) for s in good])) * 100 for c in conds}
        data[t] = dict(M=M, seeds=good, ids=ids, gold=gold, nopt=per[good[0]]["_nopt"], f1=f1,
                       missing_frac=1 - len(good) / len(seeds_all))
    gz.close()
    k = {c: i for i, c in enumerate(conds)}

    # ---------------- bootstrap
    rng = np.random.default_rng(0)
    for t, d in data.items():
        Cn, S, Q = d["M"].shape
        ws, wq = boot_weights(S, rng), boot_weights(Q, rng)
        acc_s = np.einsum("csq,bq->cbs", d["M"], wq) / Q * 100
        d["draws"] = (acc_s * ws[None]).sum(-1) / S
        d["point"] = d["M"].mean((1, 2)) * 100

    def agg(scope):
        ts = [t for t in data if scope == "all" or fam[t] == scope]
        return ts, np.mean([data[t]["point"] for t in ts], 0), np.mean([data[t]["draws"] for t in ts], 0)

    cond_rows, con_rows = [], []
    for scope in ("all", "clf", "mc"):
        ts, point, draws = agg(scope)
        if not ts:
            continue
        for c in conds:
            lo, hi = ci(draws[k[c]])
            worst = float(np.mean([data[t]["point"][k[c]] * (1 - data[t]["missing_frac"]) for t in ts]))
            cond_rows.append(dict(scope=scope, condition=c, tasks=len(ts), accuracy=point[k[c]], ci_lo=lo, ci_hi=hi,
                                  accuracy_missing_as_wrong=worst,
                                  macro_f1=float(np.mean([data[t]["f1"][c] for t in ts])) if scope == "clf" else ""))
        for a, b, name in CONTRASTS:
            d = draws[k[a]] - draws[k[b]]
            per_task = [data[t]["point"][k[a]] - data[t]["point"][k[b]] for t in ts]
            con_rows.append(dict(scope=scope, task="ALL", contrast=name, a=a, b=b, acc_a=point[k[a]], acc_b=point[k[b]],
                                 diff=point[k[a]] - point[k[b]], ci_lo=ci(d)[0], ci_hi=ci(d)[1],
                                 fixed=sum(int(((data[t]["M"][k[a]] == 1) & (data[t]["M"][k[b]] == 0)).sum()) for t in ts),
                                 broken=sum(int(((data[t]["M"][k[a]] == 0) & (data[t]["M"][k[b]] == 1)).sum()) for t in ts),
                                 tasks_better=sum(x > 1e-9 for x in per_task), tasks_worse=sum(x < -1e-9 for x in per_task)))
        a, b, c_, d_ = (k[x] for x in INTERACTION)
        dd = (draws[a] - draws[b]) - (draws[c_] - draws[d_])
        con_rows.append(dict(scope=scope, task="ALL", contrast="interaction: (D50_rank_cal − D50_rank) − (D50_cal − D50)",
                             a="", b="", diff=(point[a] - point[b]) - (point[c_] - point[d_]), ci_lo=ci(dd)[0], ci_hi=ci(dd)[1]))
    for t, d in data.items():
        for a, b, name in CONTRASTS:
            dd = d["draws"][k[a]] - d["draws"][k[b]]
            con_rows.append(dict(scope=fam[t], task=t, contrast=name, a=a, b=b, acc_a=d["point"][k[a]], acc_b=d["point"][k[b]],
                                 diff=d["point"][k[a]] - d["point"][k[b]], ci_lo=ci(dd)[0], ci_hi=ci(dd)[1],
                                 fixed=int(((d["M"][k[a]] == 1) & (d["M"][k[b]] == 0)).sum()),
                                 broken=int(((d["M"][k[a]] == 0) & (d["M"][k[b]] == 1)).sum())))

    # ---------------- mechanism table: anchors D0 and D50
    groups = [("both_right", 1, 1), ("only_D0_right", 1, 0), ("only_D50_right", 0, 1), ("both_wrong", 0, 0)]
    flip_rows = []

    def flips(split, value, sel):   # sel: {task: boolean mask over questions}
        for c in FLIP_CONDS:
            for g, r0, r50 in groups:
                share, net, acc_num, n = [], [], [], 0
                for t, m in sel.items():
                    d = data[t]
                    if not m.any():
                        continue
                    a0, a50, x = d["M"][k["D0"]][:, m], d["M"][k["D50"]][:, m], d["M"][k[c]][:, m]
                    ing = (a0 == r0) & (a50 == r50)
                    share.append(ing.mean())
                    net.append(((x - a50) * ing).mean())
                    acc_num.append((x * ing).mean())
                    n += int(ing.sum())
                if not share:
                    continue
                sh = float(np.mean(share))
                flip_rows.append(dict(split=split, value=value, condition=c, group=g, count=n, tasks=len(share),
                                      share_pct=100 * sh, accuracy_in_group_pct=100 * float(np.mean(acc_num)) / sh if sh else "",
                                      net_change_vs_D50_points=100 * float(np.mean(net))))

    full = {t: np.ones(len(d["ids"]), bool) for t, d in data.items()}
    flips("all", "all", full)
    for f in ("clf", "mc"):
        flips("family", f, {t: m for t, m in full.items() if fam[t] == f})
    for t in data:
        flips("task", t, {t: full[t]})
    for n in sorted({x for d in data.values() for x in d["nopt"]}):
        flips("n_options", n, {t: np.array(d["nopt"]) == n for t, d in data.items()})
    for t, d in data.items():
        if fam[t] == "clf":
            for lab in sorted(set(d["gold"])):
                flips("gold_label", f"{t}:{lab}", {t: np.array([g == lab for g in d["gold"]])})

    wcsv(os.path.join(run_dir, "coverage.csv"), cov_rows)
    wcsv(os.path.join(run_dir, "conditions.csv"), cond_rows)
    wcsv(os.path.join(run_dir, "contrasts.csv"), con_rows)
    wcsv(os.path.join(run_dir, "flip_groups.csv"), flip_rows)
    wcsv(os.path.join(run_dir, "label_shift.csv"), shift_rows)
    wcsv(os.path.join(run_dir, "cost.csv"), cost_rows)
    report(run_dir, cov_rows, cond_rows, con_rows, flip_rows, shift_rows, cost_rows, budget, status)


def report(run_dir, cov, cond, con, flip, shift, cost, budget, status):
    L = []
    n_ok = sum(v == "ok" for v in status.values())
    complete = sum(r["seeds_complete"] for r in cov)
    L += ["# Accuracy round 2 — result tables (generated by round2_analyze.py)", "",
          f"Trajectories present: {n_ok} / {len(status)} "
          f"(pending {sum(v == 'pending' for v in status.values())}, failed {sum(v.startswith('failed') for v in status.values())}). "
          f"Task × seed units with all four trajectories: {complete} / {sum(r['seeds_expected'] for r in cov)}.",
          "" if complete == sum(r["seeds_expected"] for r in cov) else "**INTERIM: coverage is incomplete; these numbers are provisional.**", ""]
    acc = {(r["scope"], r["condition"]): r for r in cond}
    order = ["A0", "A50", "A200", "A50_cal", "A200_cal", "A50_rank", "A50_rank_cal", "A_rankN", "A_rankN_cal",
             "D0", "D50", "D200", "D50_cal", "D200_cal", "D50_rank", "D50_rank_cal", "D_rankN", "D_rankN_cal", "D_mix"]
    L += ["## Accuracy (points; total = 21 tasks equal weight)", "",
          "| condition | total [95% CI] | classification (12) | multiple choice (9) | clf macro-F1 |", "|---|---|---|---|---|"]
    for c in order:
        if ("all", c) in acc:
            a = acc[("all", c)]
            L.append(f"| {c} | {a['accuracy']:.2f} [{a['ci_lo']:.2f}, {a['ci_hi']:.2f}] | {acc[('clf', c)]['accuracy']:.2f} | "
                     f"{acc[('mc', c)]['accuracy']:.2f} | {acc[('clf', c)]['macro_f1']:.2f} |")
    L += ["", "## Contrasts (points [95% paired bootstrap]; fixed / broken = task × seed × question counts)", "",
          "| contrast | total | classification | multiple choice | fixed / broken (total) | tasks better / worse |", "|---|---|---|---|---|---|"]
    rows = defaultdict(dict)
    for r in con:
        if r["task"] == "ALL":
            rows[r["contrast"]][r["scope"]] = r
    for name, v in rows.items():
        f = lambda r: f"{r['diff']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
        a = v["all"]
        L.append(f"| {name} | {f(a)} | {f(v['clf'])} | {f(v['mc'])} | {a.get('fixed', '')} / {a.get('broken', '')} | "
                 f"{a.get('tasks_better', '')} / {a.get('tasks_worse', '')} |")
    L += ["", "## Per-task main contrast and components", "",
          "| task | A200 | D50 | D50_cal | D50_rank | D50_rank_cal | main diff [95% CI] | N budget |", "|---|---|---|---|---|---|---|---|"]
    pt = defaultdict(dict)
    for r in con:
        if r["task"] != "ALL":
            pt[r["task"]][r["contrast"]] = r
    for t in sorted(pt):
        g = lambda n: pt[t][n]
        m = g("MAIN: D50_rank_cal − A200")
        L.append(f"| {t} | {m['acc_b']:.2f} | {g('D50 − A200')['acc_a']:.2f} | {g('D50_cal − A200')['acc_a']:.2f} | "
                 f"{g('D50_rank − A200')['acc_a']:.2f} | {m['acc_a']:.2f} | {m['diff']:+.2f} [{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}] | "
                 f"{budget[t]['budget_steps']} |")
    L += ["", "## Mechanism: questions grouped by D0 / D50 correctness (all 21 tasks, task-weighted)", "",
          "| condition | group | share % | accuracy in group % | net change vs D50 (points) |", "|---|---|---|---|---|"]
    for r in flip:
        if r["split"] == "all":
            L.append(f"| {r['condition']} | {r['group']} | {r['share_pct']:.2f} | {r['accuracy_in_group_pct']:.2f} | "
                     f"{r['net_change_vs_D50_points']:+.2f} |")
    if shift:
        L += ["", "## Label shift (classification; per task × seed × label, then averaged)", "",
              "| arm | trajectory | step | mean abs(demo share − pool share) | mean abs centred shift | corr(demo − pool, centred shift) | "
              "mean abs(pred share − gold share): C0 / Ct / Ct cal |", "|---|---|---|---|---|---|---|"]
        key = defaultdict(list)
        for r in shift:
            key[(r["arm"], r["trajectory"], r["step"])].append(r)
        for (arm, tr, st), rs in sorted(key.items()):
            db = np.array([r["demo_share"] - r["pool_share"] for r in rs])
            cs = np.array([r["centred_shift"] for r in rs])
            ok = ~np.isnan(db)
            corr = float(np.corrcoef(db[ok], cs[ok])[0, 1]) if ok.sum() > 2 else float("nan")
            gap = [np.mean([abs(r[c] - r["gold_share"]) for r in rs]) for c in ("pred_share_C0", "pred_share_Ct", "pred_share_Ct_cal")]
            L.append(f"| {arm} | {tr} | {st} | {np.nanmean(np.abs(db)):.3f} | {np.mean(np.abs(cs)):.3f} | {corr:+.3f} | "
                     f"{gap[0]:.3f} / {gap[1]:.3f} / {gap[2]:.3f} |")
    L += ["", "## Cost (medians over task × seed; D arm; seconds)", "",
          "| family | trajectory | prefill | adapt → 50 | adapt → end (steps) | step | real eval / checkpoint | null eval / checkpoint (2 templates) | "
          "real items | null items / template | peak MiB |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in ("clf", "mc"):
        for tr in ("orig", "rank"):
            rs = [r for r in cost if r["family"] == f and r["trajectory"] == tr and r["arm"] == "D"]
            if not rs:
                continue
            med = lambda key: float(np.median([r[key] for r in rs if r.get(key) is not None]))
            L.append(f"| {f} | {tr} | {med('prefill_s'):.3f} | {med('adapt_s@50'):.2f} | {med('adapt_s_total'):.2f} "
                     f"({np.median([r['stop_at'] for r in rs]):.0f}) | {med('step_s_median'):.4f} | {med('real_eval_s@50'):.2f} | "
                     f"{med('null_eval_s@50'):.2f} | {med('n_real_items'):.0f} | {med('n_null_items_per_template'):.0f} | {med('peak_mem_mb'):.0f} |")
    L += ["", "## Equal-GPU-time budget (per task)", "", "| task | step s original | step s ranking | ratio | N budget |", "|---|---|---|---|---|"]
    for t, v in sorted(budget.items()):
        L.append(f"| {t} | {v['step_s_orig']:.4f} | {v['step_s_rank']:.4f} | {v['ratio']:.2f} | {v['budget_steps']} |")
    open(os.path.join(run_dir, "REPORT_results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
