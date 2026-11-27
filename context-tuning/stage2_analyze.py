"""Stage 2 analysis: sparse demonstration-KV reading vs dense CT-KV (BBH Boolean + NLP-LR).

Reads RUN_DIR/raw_{bbh,nlplr}/*.json (Stage 2) and STAGE1/raw_{bbh,nlplr}/*.json + STAGE1/common_test_ids.json
(Stage 1 production references: A/D ICL and CT-KV). Accuracy on the frozen common test IDs (invalid = wrong).
Cells are "<adapt>/<read>@<step>", e.g. dense/dense@final, topk75/topk75@final, dense/topk75@0.
Bootstrap (5,000 paired draws): within each task resample seeds (one index shared by all cells and endpoints) and
common questions (one index shared by all seeds and cells); tasks get equal weight; seed-only intervals reported too.
Writes RUN_DIR/{cells.csv, contrasts.csv, bbh_label_checks.csv, cost.csv, predictions.jsonl.gz, REPORT_results.md}.
Usage (from context-tuning/): python stage2_analyze.py RUN_DIR STAGE1_DIR
"""
import csv
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

B = 5000
FINAL = {"bbh": 16, "nlplr": 200}
DEV_TASKS = {"boolean_expressions", "financial_phrasebank", "ai2_arc"}
DEV_SEEDS = {1000, 1001, 1002}
READ_ALIAS = {"prod": "prod", "dense": "dense", "topk@0.75": "topk75", "topk@0.5": "topk50", "fixed@0.75": "fixed75",
              "random@0.75": "rand75"}


def fam_of(task, kind):
    return "bbh" if task == "boolean_expressions" else kind


def load(run_dir, stage1):
    common = json.load(open(os.path.join(stage1, "common_test_ids.json")))
    kinds = {t: c["family"] for t, c in common.items()}
    # correctness[(family, task)][cell][seed] = {qid: 0/1}
    corr = defaultdict(lambda: defaultdict(dict))
    preds = defaultdict(lambda: defaultdict(dict))
    meta = {}
    for fam in ("bbh", "nlplr"):
        for p in sorted(glob.glob(os.path.join(run_dir, f"raw_{fam}", "*.json"))):
            r = json.load(open(p))
            task, arm, seed = r["task"], r["meta"]["arm"], r["meta"]["seed"]
            var = r["meta"]["variant"]
            if "_ms" in r["id"]:
                var = var + "_ms" + r["id"].rsplit("_ms", 1)[1]
            f = "bbh" if fam == "bbh" else kinds[task]
            meta[(f, task, arm, seed, var)] = r
            for key, ck in r["checkpoints"].items():
                step, read = key.split("|")
                stepn = "final" if int(step) == FINAL[fam] else step
                if int(step) == 0:
                    cell = f"{arm}:icl/{READ_ALIAS[read]}@0"
                else:
                    cell = f"{arm}:{var}/{READ_ALIAS[read]}@{stepn}"
                if fam == "bbh":
                    c = {row["row_id"]: int(bool(row["exact_match"])) for row in ck["predictions"]}
                    pr = {row["row_id"]: row["prediction"] for row in ck["predictions"]}
                else:
                    c = {row[0]: int(row[1] == row[2]) for row in ck["predictions"]}
                    pr = {row[0]: row[1] for row in ck["predictions"]}
                corr[(f, task)][cell][seed] = c
                preds[(f, task)][cell][seed] = pr
    # Stage 1 production references
    for fam in ("bbh", "nlplr"):
        for p in sorted(glob.glob(os.path.join(stage1, f"raw_{fam}", "generic_*.json"))):
            r = json.load(open(p))
            task, arm, seed = r["task"], r["meta"]["arm"], r["meta"]["seed"]
            f = "bbh" if fam == "bbh" else kinds[task]
            for step, ck in r["checkpoints"].items():
                tag = "S1icl" if int(step) == 0 else "S1ctkv"
                if fam == "bbh":
                    c = {row["row_id"]: int(bool(row["exact_match"])) for row in ck["predictions"]}
                else:
                    c = {row[0]: int(row[1] == row[2]) for row in ck["predictions"]}
                corr[(f, task)][f"{arm}:{tag}"][seed] = c
    return common, corr, preds, meta


def matrix(corr_t, cells, ids):
    """[cells, seeds, questions] for seeds present in every cell; None if a cell is missing."""
    seeds = None
    for c in cells:
        if c not in corr_t:
            return None, None
        s = set(corr_t[c])
        seeds = s if seeds is None else seeds & s
    seeds = sorted(seeds)
    if not seeds:
        return None, None
    M = np.array([[[corr_t[c][s].get(q, 0) for q in ids] for s in seeds] for c in cells], dtype=float)
    return M, seeds


def boot_weights(n, rng):
    idx = rng.integers(0, n, size=(B, n))
    return np.bincount((idx + np.arange(B)[:, None] * n).ravel(), minlength=B * n).reshape(B, n).astype(float)


def task_stats(M, rng, questions=True):
    """Per cell: point mean, across-seed SD, and bootstrap draws of mean and SD."""
    C, S, Q = M.shape
    ws = boot_weights(S, rng)
    wq = boot_weights(Q, rng) if questions else np.ones((B, Q))
    acc_s = np.einsum("csq,bq->cbs", M, wq) / Q * 100          # [cells, draws, seeds]
    mean_b = (acc_s * ws[None]).sum(-1) / S
    var_b = (ws[None] * (acc_s - mean_b[..., None]) ** 2).sum(-1) / max(S - 1, 1)
    point_s = M.mean(-1) * 100                                     # [cells, seeds]
    return dict(point=point_s.mean(-1), sd=point_s.std(-1, ddof=1) if S > 1 else np.zeros(C),
                mean_b=mean_b, sd_b=np.sqrt(var_b), per_seed=point_s)


def ci(x):
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def contrast_family(corr, common, fam, tasks, a, b, seed_filter=None, rng_seed=0):
    """Mean over tasks of (cell a − cell b); returns dict with point, CI, seed-only CI, SD diff, per-task rows."""
    rng = np.random.default_rng(rng_seed)
    rng2 = np.random.default_rng(rng_seed + 1)
    pts, sds, mb, sdb, mb_s, rows, nt = [], [], [], [], [], [], 0
    for task in tasks:
        ids = common[task]["common_ids"] if fam != "bbh" else None
        ct = corr[(fam, task)]
        if fam == "bbh":
            ids = sorted(next(iter(next(iter(ct.values())).values())).keys())
        cts = {c: ({s: v for s, v in ct[c].items() if seed_filter is None or seed_filter(task, s)} if c in ct else {})
               for c in (a, b)}
        M, seeds = matrix(cts, [a, b], ids)
        if M is None:
            continue
        st = task_stats(M, rng)
        st2 = task_stats(M, rng2, questions=False)
        d = st["point"][0] - st["point"][1]
        pts.append(d)
        sds.append(st["sd"][0] - st["sd"][1])
        mb.append(st["mean_b"][0] - st["mean_b"][1])
        sdb.append(st["sd_b"][0] - st["sd_b"][1])
        mb_s.append(st2["mean_b"][0] - st2["mean_b"][1])
        fixed = int(((M[0] == 1) & (M[1] == 0)).sum())
        broken = int(((M[0] == 0) & (M[1] == 1)).sum())
        rows.append(dict(task=task, seeds=len(seeds), q=len(ids), a=st["point"][0], b=st["point"][1], diff=d,
                         ci=ci(st["mean_b"][0] - st["mean_b"][1]), fixed=fixed, broken=broken,
                         sd_diff=st["sd"][0] - st["sd"][1]))
        nt += 1
    if not nt:
        return None
    mb, sdb, mb_s = np.mean(mb, 0), np.mean(sdb, 0), np.mean(mb_s, 0)
    return dict(fam=fam, a=a, b=b, tasks=nt, diff=float(np.mean(pts)), ci=ci(mb), ci_seed=ci(mb_s),
                sd_diff=float(np.mean(sds)), sd_ci=ci(sdb), rows=rows,
                fixed=sum(r["fixed"] for r in rows), broken=sum(r["broken"] for r in rows),
                a_mean=float(np.mean([r["a"] for r in rows])), b_mean=float(np.mean([r["b"] for r in rows])))


def synergy(corr, common, fam, tasks, rng_seed=7):
    """(D topk75/topk75 − D dense/dense) − (A topk75/topk75 − A dense/dense), seeds/questions resampled per task."""
    rng = np.random.default_rng(rng_seed)
    cells = ["D:topk75/topk75@final", "D:dense/dense@final", "A:topk75/topk75@final", "A:dense/dense@final"]
    pts, mb = [], []
    for task in tasks:
        ct = corr[(fam, task)]
        ids = common[task]["common_ids"] if fam != "bbh" else sorted(next(iter(ct[cells[1]].values())).keys()) \
            if cells[1] in ct else None
        M, seeds = matrix(ct, cells, ids) if ids is not None else (None, None)
        if M is None:
            continue
        st = task_stats(M, rng)
        p = st["point"]
        pts.append((p[0] - p[1]) - (p[2] - p[3]))
        m = st["mean_b"]
        mb.append((m[0] - m[1]) - (m[2] - m[3]))
    if not pts:
        return None
    return dict(diff=float(np.mean(pts)), ci=ci(np.mean(mb, 0)), tasks=len(pts))


def bbh_label_rows(meta):
    """Per job × read: recall, predicted-True ratio, label share of kept demo tokens vs legal candidates."""
    rows = []
    for (f, task, arm, seed, var), r in sorted(meta.items()):
        if f != "bbh" or "_ms" in var:
            continue
        labels = r["demo_targets"]
        demo_true = np.mean([t == "True" for t in labels])
        for key, ck in r["checkpoints"].items():
            step, read = key.split("|")
            pr = ck["predictions"]
            gold_t = [p for p in pr if p["gold"] == "True"]
            gold_f = [p for p in pr if p["gold"] == "False"]
            row = dict(arm=arm, seed=seed, adapt=var, read=READ_ALIAS[read], step=int(step), demo_true=demo_true,
                       acc=np.mean([p["exact_match"] for p in pr]) * 100,
                       pred_true=np.mean([p["prediction"] == "True" for p in pr]),
                       recall_true=np.mean([p["exact_match"] for p in gold_t]),
                       recall_false=np.mean([p["exact_match"] for p in gold_f]),
                       invalid=sum(p["valid_format"] is False for p in pr))
            if pr and "kept_by_demo" in pr[0]:
                kept = np.array([p["kept_by_demo"] for p in pr], float)
                legal = np.array([p["legal_by_demo"] for p in pr], float)
                is_t = np.array([t == "True" for t in labels])
                row["kept_share_true"] = float(np.mean(kept[:, is_t].sum(1) / kept.sum(1)))
                row["legal_share_true"] = float(np.mean(legal[:, is_t].sum(1) / legal.sum(1)))
                frac = kept / np.maximum(legal, 1)
                row["kept_frac_true_demo_equal"] = float(frac[:, is_t].mean()) if is_t.any() else np.nan
                row["kept_frac_false_demo_equal"] = float(frac[:, ~is_t].mean()) if (~is_t).any() else np.nan
            rows.append(row)
    return rows


def cost_rows(meta):
    rows = []
    for (f, task, arm, seed, var), r in sorted(meta.items()):
        if "_ms" in var:
            continue
        perf, tm = r["perf"], r["timing"]
        ph = perf["phases"]
        adapt_s = tm.get("adaptation_s", tm.get("adapt_s"))
        st = perf.get("step_s", [])
        row = dict(family=f, task=task, arm=arm, seed=seed, adapt=var, adapt_s=adapt_s,
                   forward_s=ph.get("forward", {}).get("s", 0), backward_s=ph.get("backward", {}).get("s", 0),
                   select_init_s=ph.get("select_init", {}).get("s", 0),
                   step_median_s=float(np.median(st[1:])) if len(st) > 1 else np.nan,
                   peak_mib=perf["overall_peak_bytes"] / 2**20,
                   kv_resident_mib=r.get("cache_bytes", perf["static_bytes"].get("kv_cache", 0)) / 2**20,
                   kv_full_mib=r.get("full_cache_bytes", perf["static_bytes"].get("kv_cache", 0)) / 2**20,
                   train_access=(r.get("sparse_stats", {}) or {}).get("train", {}) and
                   r["sparse_stats"]["train"]["access_ratio"])
        for key, ck in r["checkpoints"].items():
            n = ck.get("n", ck.get("n_items"))
            row[f"eval_s|{key}"] = ck["eval_s"]
            row[f"eval_per_item_ms|{key}"] = ck["eval_s"] / max(n, 1) * 1e3
            if ck.get("sparse"):
                row[f"access|{key}"] = ck["sparse"]["access_ratio"]
        rows.append(row)
    return rows


def fmt(c, so=True):
    if c is None:
        return "—"
    s = f"{c['diff']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]"
    if so:
        s += f" (seed-only [{c['ci_seed'][0]:+.2f}, {c['ci_seed'][1]:+.2f}])"
    return s


def main(run_dir, stage1):
    common, corr, preds, meta = load(run_dir, stage1)
    fams = {"bbh": ["boolean_expressions"],
            "clf": sorted(t for t, c in common.items() if c["family"] == "clf"),
            "mc": sorted(t for t, c in common.items() if c["family"] == "mc")}
    fams = {f: [t for t in ts if (f, t) in corr] for f, ts in fams.items()}
    NAMES = {"bbh": "BBH Boolean (Llama-3-8B-Instruct, 10-shot, 16 steps)",
             "clf": "NLP-LR classification (GPT-2 Large, 16-shot, 200 steps)",
             "mc": "NLP-LR multiple choice (GPT-2 Large, 16-shot, 200 steps)"}
    F = "@final"
    CON = [  # (label, a, b, group)
        ("**main: sparse/sparse − dense/dense** (top-k 75%)", "D:topk75/topk75" + F, "D:dense/dense" + F, "main"),
        ("dense adapt, top-k read − dense/dense", "D:dense/topk75" + F, "D:dense/dense" + F, "matrix"),
        ("top-k adapt, dense read − dense/dense", "D:topk75/dense" + F, "D:dense/dense" + F, "matrix"),
        ("top-k adapt, top-k read − top-k adapt, dense read", "D:topk75/topk75" + F, "D:topk75/dense" + F, "matrix"),
        ("50%: sparse/sparse − dense/dense", "D:topk50/topk50" + F, "D:dense/dense" + F, "budget"),
        ("50%: dense adapt, top-k read − dense/dense", "D:dense/topk50" + F, "D:dense/dense" + F, "budget"),
        ("50%: top-k adapt, dense read − dense/dense", "D:topk50/dense" + F, "D:dense/dense" + F, "budget"),
        ("dynamic top-k − random (75%, sparse/sparse)", "D:topk75/topk75" + F, "D:rand75/rand75" + F, "control"),
        ("dynamic top-k − fixed answer_attention (75%, sparse/sparse)", "D:topk75/topk75" + F,
         "D:fixed75/fixed75" + F, "control"),
        ("fixed answer_attention 75% − dense/dense", "D:fixed75/fixed75" + F, "D:dense/dense" + F, "control"),
        ("random 75% − dense/dense", "D:rand75/rand75" + F, "D:dense/dense" + F, "control"),
        ("physical prune 75% (Ryan) − Stage 1 D CT-KV (prod)", "D:prune75/prod" + F, "D:S1ctkv", "control"),
        ("physical prune 50% (Ryan) − Stage 1 D CT-KV (prod)", "D:prune50/prod" + F, "D:S1ctkv", "control"),
        ("main sparse/sparse − A ICL (Stage 1, zero-update, random demos)", "D:topk75/topk75" + F, "A:S1icl", "ref"),
        ("main sparse/sparse − D ICL (zero-update, refined demos, dense read)", "D:topk75/topk75" + F,
         "D:icl/dense@0", "ref"),
        ("main sparse/sparse − A CT-KV (Stage 1 original CT-KV)", "D:topk75/topk75" + F, "A:S1ctkv", "ref"),
        ("main sparse/sparse − D CT-KV (Stage 1, prod kernel)", "D:topk75/topk75" + F, "D:S1ctkv", "ref"),
        ("step 0: top-k 75% read − dense read (ICL, refined demos)", "D:icl/topk75@0", "D:icl/dense@0", "step0"),
        ("step 0: top-k 50% read − dense read", "D:icl/topk50@0", "D:icl/dense@0", "step0"),
        ("step 0: random 75% read − dense read", "D:icl/rand75@0", "D:icl/dense@0", "step0"),
        ("step 0: fixed 75% read − dense read", "D:icl/fixed75@0", "D:icl/dense@0", "step0"),
        ("implementation: D dense/dense (diagnostic) − Stage 1 D CT-KV (prod)", "D:dense/dense" + F, "D:S1ctkv",
         "impl"),
        ("implementation: dense checkpoint, prod read − dense read", "D:dense/prod" + F, "D:dense/dense" + F, "impl"),
        ("implementation: step 0, dense read − Stage 1 D ICL (prod)", "D:icl/dense@0", "D:S1icl", "impl"),
        ("A arm: sparse/sparse − dense/dense", "A:topk75/topk75" + F, "A:dense/dense" + F, "synergy"),
    ]
    res, con_rows = {}, []
    confirm = lambda t, s: not (t in DEV_TASKS and s in DEV_SEEDS)
    for f, ts in fams.items():
        for i, (lab, a, b, g) in enumerate(CON):
            c = contrast_family(corr, common, f, ts, a, b, rng_seed=100 + i)
            cc = contrast_family(corr, common, f, ts, a, b, seed_filter=confirm, rng_seed=200 + i) \
                if g == "main" else None
            res[(f, lab)] = (c, cc)
            if c:
                con_rows.append(dict(family=f, contrast=lab.replace("*", ""), a=a, b=b, group=g, tasks=c["tasks"],
                                     diff=c["diff"], lo=c["ci"][0], hi=c["ci"][1], lo_seed=c["ci_seed"][0],
                                     hi_seed=c["ci_seed"][1], sd_diff=c["sd_diff"], sd_lo=c["sd_ci"][0],
                                     sd_hi=c["sd_ci"][1], fixed=c["fixed"], broken=c["broken"],
                                     a_mean=c["a_mean"], b_mean=c["b_mean"]))
                for r in c["rows"]:
                    con_rows.append(dict(family=f, contrast=lab.replace("*", ""), a=a, b=b, group=g + ":task",
                                         task=r["task"], tasks=1, diff=r["diff"], lo=r["ci"][0], hi=r["ci"][1],
                                         sd_diff=r["sd_diff"], fixed=r["fixed"], broken=r["broken"],
                                         a_mean=r["a"], b_mean=r["b"]))
    syn = {f: synergy(corr, common, f, ts) for f, ts in fams.items()}
    write_csv(os.path.join(run_dir, "contrasts.csv"), con_rows)

    # cell means
    cell_rows = []
    for (f, task), ct in sorted(corr.items()):
        ids = common[task]["common_ids"] if f != "bbh" else None
        for cell, by_seed in sorted(ct.items()):
            for seed, c in sorted(by_seed.items()):
                q = ids if ids is not None else sorted(c)
                cell_rows.append(dict(family=f, task=task, cell=cell, seed=seed,
                                      acc=np.mean([c.get(i, 0) for i in q]) * 100, n_q=len(q)))
    write_csv(os.path.join(run_dir, "cells.csv"), cell_rows)
    lab_rows = bbh_label_rows(meta)
    write_csv(os.path.join(run_dir, "bbh_label_checks.csv"), lab_rows)
    cost = cost_rows(meta)
    write_csv(os.path.join(run_dir, "cost.csv"), cost)
    with gzip.open(os.path.join(run_dir, "predictions.jsonl.gz"), "wt") as g:
        for (f, task), ct in sorted(preds.items()):
            for cell, by_seed in sorted(ct.items()):
                for seed, pr in sorted(by_seed.items()):
                    g.write(json.dumps(dict(family=f, task=task, cell=cell, seed=seed,
                                            correct=corr[(f, task)][cell][seed], prediction=pr)) + "\n")
    report(run_dir, fams, NAMES, CON, res, syn, cell_rows, lab_rows, cost, corr)


def cell_mean(cell_rows, f, cell):
    by_task = defaultdict(list)
    for r in cell_rows:
        if r["family"] == f and r["cell"] == cell:
            by_task[r["task"]].append(r["acc"])
    return (float(np.mean([np.mean(v) for v in by_task.values()])), len(by_task)) if by_task else (np.nan, 0)


def report(run_dir, fams, NAMES, CON, res, syn, cell_rows, lab_rows, cost, corr):
    L = ["# CT-KV Stage 2: per-query sparse reading of demonstration KV — results", "",
         "All numbers: accuracy in points on the frozen Stage 1 common test IDs, seed-mean first, tasks equal "
         "weight. Brackets: 95% paired bootstrap (5,000 draws; seeds × questions; seed-only in parentheses). "
         "Cells are `adapt/read`; dense = diagnostic float32 path at r = 1 (same implementation as the sparse "
         "reads); prod = production SDPA kernel; S1 = Stage 1 production runs.", ""]
    L += ["## Main comparison (arm D refined demonstrations, top-k 75%)", "",
          "| evaluation | tasks | dense/dense | top-k/top-k | sparse − dense | confirmation cells only | SD diff | "
          "fixed / broken Q |", "|---|---|---|---|---|---|---|---|"]
    for f in fams:
        c, cc = res.get((f, CON[0][0]), (None, None))
        if not c:
            continue
        L.append(f"| {NAMES[f]} | {c['tasks']} | {c['b_mean']:.2f} | {c['a_mean']:.2f} | {fmt(c, False)} | "
                 f"{fmt(cc, False) if cc else '—'} | {c['sd_diff']:+.2f} [{c['sd_ci'][0]:+.2f}, {c['sd_ci'][1]:+.2f}] | "
                 f"{c['fixed']} / {c['broken']} |")
    L += ["", "Confirmation cells exclude the exploration cells (BBH Boolean, financial_phrasebank, ai2_arc × seeds "
          "1000–1002); see PLAN.md (no independent dev set exists).", ""]
    L += ["## Levels (mean accuracy per cell)", "",
          "| cell | " + " | ".join(f for f in fams) + " |", "|---|" + "---|" * len(fams)]
    show = [("A ICL (Stage 1, prod)", "A:S1icl"), ("A CT-KV (Stage 1, prod)", "A:S1ctkv"),
            ("D ICL (Stage 1, prod)", "D:S1icl"), ("D CT-KV (Stage 1, prod)", "D:S1ctkv"),
            ("D step 0, dense read", "D:icl/dense@0"), ("D step 0, top-k 75% read", "D:icl/topk75@0"),
            ("D step 0, top-k 50% read", "D:icl/topk50@0"), ("D step 0, fixed 75% read", "D:icl/fixed75@0"),
            ("D step 0, random 75% read", "D:icl/rand75@0"),
            ("D dense/dense", "D:dense/dense@final"), ("D dense/prod", "D:dense/prod@final"),
            ("D dense/top-k 75%", "D:dense/topk75@final"), ("D top-k 75%/dense", "D:topk75/dense@final"),
            ("**D top-k 75%/top-k 75% (main)**", "D:topk75/topk75@final"),
            ("D dense/top-k 50%", "D:dense/topk50@final"), ("D top-k 50%/dense", "D:topk50/dense@final"),
            ("D top-k 50%/top-k 50%", "D:topk50/topk50@final"),
            ("D fixed 75%/fixed 75%", "D:fixed75/fixed75@final"), ("D random 75%/random 75%", "D:rand75/rand75@final"),
            ("D physical prune 75% (Ryan)", "D:prune75/prod@final"), ("D physical prune 50% (Ryan)",
                                                                       "D:prune50/prod@final"),
            ("A dense/dense", "A:dense/dense@final"), ("A top-k 75%/top-k 75%", "A:topk75/topk75@final")]
    for lab, cell in show:
        vals = []
        for f in fams:
            m, n = cell_mean(cell_rows, f, cell)
            vals.append(f"{m:.2f}" if n else "—")
        L.append(f"| {lab} | " + " | ".join(vals) + " |")
    L += [""]
    groups = [("matrix", "Run matrix (75%)"), ("budget", "Budget sensitivity (50%)"),
              ("control", "Controls: random, fixed answer_attention, physical pruning"),
              ("ref", "Against zero-update and original CT-KV references"), ("step0", "Step 0 (no update): read only"),
              ("impl", "Implementation checks"), ("synergy", "Arm A (random demonstrations)")]
    for g, title in groups:
        L += [f"## {title}", "", "| contrast | " + " | ".join(f for f in fams) + " |", "|---|" + "---|" * len(fams)]
        for lab, a, b, gg in CON:
            if gg != g:
                continue
            L.append(f"| {lab} | " + " | ".join(fmt(res.get((f, lab), (None,))[0], False) for f in fams) + " |")
        if g == "synergy":
            L.append("| **synergy**: (D sparse − D dense) − (A sparse − A dense) | " + " | ".join(
                (f"{syn[f]['diff']:+.2f} [{syn[f]['ci'][0]:+.2f}, {syn[f]['ci'][1]:+.2f}]" if syn.get(f) else "—")
                for f in fams) + " |")
        L.append("")
    # per task main
    L += ["## Per task: main contrast (D top-k 75% sparse/sparse − dense/dense)", "",
          "| family | task | seeds | Q | dense/dense | sparse/sparse | diff | fixed / broken | SD diff |",
          "|---|---|---|---|---|---|---|---|---|"]
    for f in fams:
        c = res.get((f, CON[0][0]), (None,))[0]
        if not c:
            continue
        for r in c["rows"]:
            L.append(f"| {f} | {r['task']} | {r['seeds']} | {r['q']} | {r['b']:.2f} | {r['a']:.2f} | "
                     f"{r['diff']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {r['fixed']} / {r['broken']} | "
                     f"{r['sd_diff']:+.2f} |")
    L += [""]
    # random mask-seed spread (dev)
    ms = defaultdict(dict)
    for r in cell_rows:
        if "rand75_ms" in r["cell"] or r["cell"] == "D:rand75/rand75@final":
            tag = r["cell"].split("/")[0].split(":")[1]
            ms[(r["family"], r["task"], r["seed"])][tag] = r["acc"]
    rows_ms = [(k, v) for k, v in sorted(ms.items()) if len(v) >= 3]
    if rows_ms:
        L += ["## Random-mask seed spread (exploration cells, random 75% sparse/sparse)", "",
              "| family | task | seed | mask 0 | mask 1 | mask 2 | range |", "|---|---|---|---|---|---|---|"]
        for (f, t, s), v in rows_ms:
            a = [v.get("rand75"), v.get("rand75_ms1"), v.get("rand75_ms2")]
            a = [x for x in a if x is not None]
            L.append(f"| {f} | {t} | {s} | " + " | ".join(f"{x:.2f}" for x in a) + f" | {max(a) - min(a):.2f} |")
        L += [""]
    # BBH label checks
    if lab_rows:
        L += ["## BBH label-shift diagnostics (mean over seeds; arm D unless noted)", "",
              "| adapt / read | step | acc | pred True | recall T | recall F | kept share True | legal share True | "
              "kept frac True demos | kept frac False demos |", "|---|---|---|---|---|---|---|---|---|---|"]
        key = defaultdict(list)
        for r in lab_rows:
            key[(r["arm"], r["adapt"], r["read"], r["step"])].append(r)
        for (arm, ad, rd, st), rs in sorted(key.items()):
            m = lambda k: np.nanmean([x.get(k, np.nan) for x in rs])
            ad_lab = "icl" if st == 0 else ad
            if st == 0 and ad != "dense":
                continue
            L.append(f"| {arm} {ad_lab}/{rd} | {st} | {m('acc'):.2f} | {m('pred_true'):.3f} | {m('recall_true'):.3f} | "
                     f"{m('recall_false'):.3f} | {m('kept_share_true'):.3f} | {m('legal_share_true'):.3f} | "
                     f"{m('kept_frac_true_demo_equal'):.3f} | {m('kept_frac_false_demo_equal'):.3f} |")
        # correlation of change with demo label bias
        dd = {(r["seed"]): r for r in lab_rows if r["arm"] == "D" and r["adapt"] == "dense" and r["read"] == "dense"
              and r["step"] == 16}
        ss = {(r["seed"]): r for r in lab_rows if r["arm"] == "D" and r["adapt"] == "topk75" and r["read"] == "topk75"}
        common_s = sorted(set(dd) & set(ss))
        if len(common_s) > 2:
            bias = np.array([dd[s]["demo_true"] - 0.5 for s in common_s])
            dpt = np.array([ss[s]["pred_true"] - dd[s]["pred_true"] for s in common_s])
            dacc = np.array([ss[s]["acc"] - dd[s]["acc"] for s in common_s])
            shr = np.array([ss[s].get("kept_share_true", np.nan) - ss[s].get("legal_share_true", np.nan)
                            for s in common_s])
            L += ["", f"Across {len(common_s)} seeds (D): demo True-share − 0.5 has range "
                  f"{bias.min():+.2f}…{bias.max():+.2f}; Δ(pred True) sparse/sparse − dense/dense mean {dpt.mean():+.3f}; "
                  f"Δacc mean {dacc.mean():+.2f}; kept − legal True-token share (top-k read) mean {np.nanmean(shr):+.4f}. "
                  + (f"corr(bias, Δpred True) = {np.corrcoef(bias, dpt)[0, 1]:+.2f}, "
                     f"corr(bias, Δacc) = {np.corrcoef(bias, dacc)[0, 1]:+.2f}." if bias.std() > 0 else
                     "All D demo sets have the same label share, so correlations are not estimable."),
                  "Kept share is the share of selected demo positions (summed over layers, heads and query tokens) "
                  "that belong to True-labelled demonstrations; legal share is the same over all legal candidates, "
                  "so a longer True demo does not count as routing preference. Demo-equal columns average the "
                  "kept fraction per demonstration.", ""]
    # cost
    if cost:
        L += cost_lines(cost)
    open(os.path.join(run_dir, "REPORT_results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:20]))


def cost_lines(cost):
    L = ["## Cost (medians over jobs)", "",
         "Timings below come from the accuracy runs, in which the BBH and NLP-LR queues shared the GPU; the "
         "serial benchmark (`bench/`) is the reference for speed.", "",
         "| family | adapt | jobs | adaptation s | forward s | backward s | select init s | step s | peak MiB | "
         "resident KV MiB | train access |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    g = defaultdict(list)
    for r in cost:
        g[(r["family"], r["arm"], r["adapt"])].append(r)
    for (f, arm, ad), rs in sorted(g.items()):
        med = lambda k: np.nanmedian([np.nan if x.get(k) in (None, "", False) else x[k] for x in rs])
        L.append(f"| {f} | {arm} {ad} | {len(rs)} | {med('adapt_s'):.2f} | {med('forward_s'):.2f} | "
                 f"{med('backward_s'):.2f} | {med('select_init_s'):.2f} | {med('step_median_s'):.3f} | "
                 f"{med('peak_mib'):.0f} | {med('kv_resident_mib'):.1f} | {med('train_access'):.3f} |")
    L += [""]
    return L


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else (v if not isinstance(v, tuple) else
                                                                      f"{v[0]:.4g};{v[1]:.4g}")) for k, v in r.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
