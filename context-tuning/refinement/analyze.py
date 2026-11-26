"""Analysis for the generic demonstration-refinement study (CTKV_Demonstration_Refinement_Guideline §4).

Reads RUN_DIR/{selection.jsonl, nlplr_jobs.jsonl, common_test_ids.json, raw_bbh/*.json, raw_nlplr/*.json} and
writes RUN_DIR/{runs.csv, contrasts.csv, bbh_checks.csv, nlplr_class_checks.csv, predictions.jsonl.gz, REPORT.md}.

Accuracy is recomputed from per-question predictions on the frozen common test IDs (invalid = wrong).
Uncertainty: 5,000 paired bootstrap draws; seeds resampled with one index shared by all arms/endpoints of a task,
questions resampled within the task's common IDs with one index shared by all seeds/arms/endpoints. A seed-only
bootstrap (questions fixed) is reported alongside.
Usage (from context-tuning/): python refinement/analyze.py RUN_DIR
"""
import csv, glob, gzip, json, os, sys
from collections import Counter, defaultdict

import numpy as np

B = 5000
FINAL = {"bbh": "16", "clf": "200", "mc": "200"}
CONTRASTS = {"bbh": [("D", "A"), ("C", "A"), ("D", "C"), ("C", "B")],
             "clf": [("D", "A"), ("C", "A"), ("D", "C"), ("C", "B")],
             "mc": [("D", "A")]}
FAMILY_NAME = {"bbh": "BBH Boolean (Llama-3-8B-Instruct, 10-shot, 20 seeds)",
               "clf": "NLP-LR fixed-label classification (GPT-2 Large, 16-shot, 10 seeds)",
               "mc": "NLP-LR per-question multiple choice (GPT-2 Large, 16-shot, 10 seeds)"}


def macro_f1(pred, gold):
    labels = sorted(set(gold) | set(pred))
    f = []
    for y in labels:
        tp = sum(p == y and g == y for p, g in zip(pred, gold))
        fp = sum(p == y and g != y for p, g in zip(pred, gold))
        fn = sum(p != y and g == y for p, g in zip(pred, gold))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else float("nan")


def tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)


def dist(xs):
    c = Counter(xs)
    n = sum(c.values())
    return {k: v / n for k, v in c.items()} if n else {}


# ---------------------------------------------------------------- loading

def load(run_dir):
    sel = [json.loads(l) for l in open(os.path.join(run_dir, "selection.jsonl"))]
    common = json.load(open(os.path.join(run_dir, "common_test_ids.json")))
    runs = {}  # (family, task, arm, seed) -> dict(endpoint -> {qid: (pred, gold, correct)}), meta
    for path in glob.glob(os.path.join(run_dir, "raw_bbh", "generic_bbh_*.json")):
        r = json.load(open(path))
        m = r["meta"]
        eps = {}
        for k, c in r["checkpoints"].items():
            eps[k] = {int(p["row_id"]): (p["prediction"] if p["valid_format"] else None, p["gold"],
                                         bool(p["exact_match"]) and bool(p["valid_format"]))
                      for p in c["predictions"]}
        runs[("bbh", "boolean_expressions", m["arm"], m["seed"])] = dict(
            eps=eps, adapt_s=r["timing"]["adaptation_s"], demo_targets=r["demo_targets"], meta=m,
            perf=r.get("perf"), eval_s=r["timing"]["checkpoint_eval_s"],
            n_invalid={k: c["n_invalid"] for k, c in r["checkpoints"].items()})
    for path in glob.glob(os.path.join(run_dir, "raw_nlplr", "generic_*.json")):
        r = json.load(open(path))
        m = r["meta"]
        eps = {k: {int(t): (p, g, p == g) for t, p, g, _ in c["predictions"]} for k, c in r["checkpoints"].items()}
        runs[(m["family"], r["task"], m["arm"], m["seed"])] = dict(
            eps=eps, adapt_s=r["timing"]["adapt_s"], demo_targets=r["demo_labels"], meta=m,
            perf=r.get("perf"), eval_s=r["timing"]["checkpoint_eval_s"])
    return sel, common, runs


# ---------------------------------------------------------------- per-task arrays and bootstrap

def task_arrays(runs, fam, task, arms, ids):
    """Seeds complete for all `arms`; correctness array [arm, seed, endpoint(0/final), q]."""
    seeds = sorted({k[3] for k in runs if k[0] == fam and k[1] == task})
    seeds = [s for s in seeds if all((fam, task, a, s) in runs for a in arms)]
    if not seeds or not ids:
        return seeds, None
    C = np.zeros((len(arms), len(seeds), 2, len(ids)), dtype=np.float64)
    for ai, a in enumerate(arms):
        for si, s in enumerate(seeds):
            eps = runs[(fam, task, a, s)]["eps"]
            for ei, ep in enumerate(("0", FINAL[fam])):
                C[ai, si, ei] = [eps[ep][q][2] if q in eps[ep] else np.nan for q in ids]
    return seeds, C


def stats(acc):
    """acc [arm(2), seed, endpoint] -> dict of contrasts X - Y (arm 0 minus arm 1), in points."""
    fin, icl = acc[:, :, 1], acc[:, :, 0]
    gain = fin - icl
    sd = fin.std(axis=1, ddof=1) if fin.shape[1] > 1 else np.full(2, np.nan)
    return np.array([fin[0].mean() - fin[1].mean(), gain[0].mean() - gain[1].mean(),
                     icl[0].mean() - icl[1].mean(), sd[0] - sd[1]]) * 100


STAT_NAMES = ("final", "gain", "icl", "sd")


def boot_draws(C, rng, questions=True):
    """Bootstrap stats for one task's 2-arm array; returns [B, 4]."""
    n_s, n_q = C.shape[1], C.shape[3]
    out = np.empty((B, 4))
    for b in range(B):
        s = rng.integers(0, n_s, n_s)
        if questions:
            w = np.bincount(rng.integers(0, n_q, n_q), minlength=n_q) / n_q
            acc = C[:, s] @ w
        else:
            acc = C[:, s].mean(axis=3)
        out[b] = stats(acc)
    return out


def ci(x):
    lo, hi = np.nanpercentile(x, [2.5, 97.5], axis=0)
    return lo, hi


# ---------------------------------------------------------------- main

def main(run_dir):
    sel, common, runs = load(run_dir)
    rng = np.random.default_rng(0)
    sel_by = {(r["task"], r["arm"], r["seed"]): r for r in sel}
    tasks = defaultdict(set)
    for r in sel:
        tasks[r["family"]].add(r["task"])
    bbh_ids = sorted({q for k, v in runs.items() if k[0] == "bbh" for q in v["eps"]["0"]})
    common_ids = {t: c["common_ids"] for t, c in common.items()}
    common_ids["boolean_expressions"] = bbh_ids

    # ---------- runs.csv + predictions
    run_rows = []
    pred_path = os.path.join(run_dir, "predictions.jsonl.gz")
    with gzip.open(pred_path, "wt") as pf:
        for (fam, task, arm, seed), r in sorted(runs.items()):
            ids = set(common_ids.get(task, []))
            for ep, preds in r["eps"].items():
                on = [preds[q] for q in sorted(preds) if q in ids]
                row = dict(family=fam, task=task, arm=arm, seed=seed, step=int(ep), n_eval=len(preds),
                           n_common=len(on), acc_common=np.mean([x[2] for x in on]) * 100 if on else np.nan,
                           acc_own=np.mean([x[2] for x in preds.values()]) * 100,
                           adapt_s=r["adapt_s"])
                if fam != "mc":
                    row["macro_f1_common"] = macro_f1([str(x[0]) for x in on], [x[1] for x in on]) * 100 if on else np.nan
                run_rows.append(row)
                for q in sorted(preds):
                    p, g, c = preds[q]
                    pf.write(json.dumps(dict(family=fam, task=task, arm=arm, seed=seed, step=int(ep), test_id=q,
                                             prediction=p, gold=g, correct=c, in_common=q in ids)) + "\n")
    write_csv(os.path.join(run_dir, "runs.csv"), run_rows)

    # ---------- contrasts
    con_rows, per_task, summary = [], {}, {}
    for fam in ("bbh", "clf", "mc"):
        for X, Y in CONTRASTS[fam]:
            draws, draws_s, points, n_tasks, gaps = [], [], [], 0, []
            for task in sorted(tasks[fam]):
                ids = common_ids.get(task, [])
                seeds, C = task_arrays(runs, fam, task, [X, Y], ids)
                planned = sorted({s for (t, a, s) in sel_by if t == task})
                if len(seeds) < len(planned):
                    gaps.append((task, sorted(set(planned) - set(seeds))))
                if C is None or len(seeds) < 2:
                    con_rows.append(dict(family=fam, task=task, contrast=f"{X}-{Y}", status="not estimable",
                                         n_seeds=len(seeds), n_q=len(ids)))
                    continue
                if np.isnan(C).any():
                    raise SystemExit(f"missing common-ID predictions in {task} {X}/{Y}")
                pt = stats(C.mean(axis=3))
                d = boot_draws(C, rng)
                ds = boot_draws(C, rng, questions=False)
                lo, hi = ci(d)
                slo, shi = ci(ds)
                row = dict(family=fam, task=task, contrast=f"{X}-{Y}", status="ok", n_seeds=len(seeds), n_q=len(ids))
                for i, nm in enumerate(STAT_NAMES):
                    row.update({nm: pt[i], f"{nm}_lo": lo[i], f"{nm}_hi": hi[i],
                                f"{nm}_seedonly_lo": slo[i], f"{nm}_seedonly_hi": shi[i]})
                con_rows.append(row)
                per_task[(fam, task, X, Y)] = row
                draws.append(d), draws_s.append(ds), points.append(pt)
                n_tasks += 1
            if not points:
                continue
            pt = np.mean(points, axis=0)
            lo, hi = ci(np.mean(draws, axis=0))
            slo, shi = ci(np.mean(draws_s, axis=0))
            row = dict(family=fam, task="__mean__", contrast=f"{X}-{Y}", status="ok", n_tasks=n_tasks,
                       missing_seeds=json.dumps(gaps))
            for i, nm in enumerate(STAT_NAMES):
                row.update({nm: pt[i], f"{nm}_lo": lo[i], f"{nm}_hi": hi[i],
                            f"{nm}_seedonly_lo": slo[i], f"{nm}_seedonly_hi": shi[i]})
            con_rows.append(row)
            summary[(fam, X, Y)] = row
            if fam == "clf" and (X, Y) == ("D", "A"):
                clf_draws, clf_pts = draws, points
            if fam == "mc" and (X, Y) == ("D", "A"):
                mc_draws, mc_pts = draws, points
    if ("clf", "D", "A") in summary and ("mc", "D", "A") in summary:
        all_d, all_p = clf_draws + mc_draws, clf_pts + mc_pts
        pt = np.mean(all_p, axis=0)
        lo, hi = ci(np.mean(all_d, axis=0))
        row = dict(family="nlplr21", task="__mean__", contrast="D-A", status="ok", n_tasks=len(all_p))
        for i, nm in enumerate(STAT_NAMES):
            row.update({nm: pt[i], f"{nm}_lo": lo[i], f"{nm}_hi": hi[i]})
        con_rows.append(row)
        summary[("nlplr21", "D", "A")] = row

    # pairwise-intersection sensitivity (NLP-LR): common IDs over the two arms only
    pair_rows = []
    for fam in ("clf", "mc"):
        for X, Y in CONTRASTS[fam]:
            for task in sorted(tasks[fam]):
                keys = [k for k in runs if k[0] == fam and k[1] == task and k[2] in (X, Y)]
                if not keys:
                    continue
                ids = sorted(set.intersection(*[set(runs[k]["eps"]["0"]) & set(runs[k]["eps"][FINAL[fam]])
                                                for k in keys]))
                seeds, C = task_arrays(runs, fam, task, [X, Y], ids)
                if C is None or len(seeds) < 2:
                    continue
                pt = stats(C.mean(axis=3))
                pair_rows.append(dict(family=fam, task=task, contrast=f"{X}-{Y}", n_q_pair=len(ids),
                                      n_q_global=len(common_ids.get(task, [])), final=pt[0], gain=pt[1]))
    for r in pair_rows:
        r["contrast_kind"] = "pairwise_intersection"
    write_csv(os.path.join(run_dir, "contrasts.csv"), con_rows + pair_rows)

    # ---------- BBH checks
    bbh_rows = []
    for (fam, task, arm, seed), r in sorted(runs.items()):
        if fam != "bbh":
            continue
        row = dict(arm=arm, seed=seed, demo_true=sum(t == "True" for t in r["demo_targets"]) / len(r["demo_targets"]))
        for ep in ("0", "16"):
            P = r["eps"][ep]
            n = len(P)
            row[f"pred_true_{ep}"] = sum(p == "True" for p, g, c in P.values()) / n
            for y in ("True", "False"):
                gs = [c for p, g, c in P.values() if g == y]
                row[f"recall_{y}_{ep}"] = np.mean(gs) if gs else np.nan
            row[f"invalid_{ep}"] = r["n_invalid"][ep]
            row[f"acc_{ep}"] = np.mean([c for *_, c in P.values()]) * 100
        row["fixed_0to16"] = sum(not r["eps"]["0"][q][2] and r["eps"]["16"][q][2] for q in r["eps"]["0"])
        row["broken_0to16"] = sum(r["eps"]["0"][q][2] and not r["eps"]["16"][q][2] for q in r["eps"]["0"])
        bbh_rows.append(row)
    write_csv(os.path.join(run_dir, "bbh_checks.csv"), bbh_rows)

    # ---------- NLP-LR classification label checks
    cls_rows = []
    for (fam, task, arm, seed), r in sorted(runs.items()):
        if fam != "clf":
            continue
        s = sel_by[(task, arm, seed)]
        pool = {json.loads(k): v for k, v in s["pool_counts"].items()}
        n = sum(pool.values())
        pool_d = {k: v / n for k, v in pool.items()}
        ids = set(common_ids[task])
        row = dict(task=task, arm=arm, seed=seed, demo_tv_pool=s["label_tv"])
        for ep in ("0", "200"):
            on = [r["eps"][ep][q] for q in sorted(ids)]
            pd_, gd = dist([p for p, g, c in on]), dist([g for p, g, c in on])
            row[f"pred_tv_pool_{ep}"] = tv(pd_, pool_d)
            row[f"pred_tv_gold_{ep}"] = tv(pd_, gd)
            for y in sorted(pool):
                gs = [c for p, g, c in on if g == y]
                row[f"recall[{y}]_{ep}"] = np.mean(gs) if gs else np.nan
            rec = [np.mean([c for p, g, c in on if g == y]) for y in sorted(set(g for _, g, _ in on))]
            row[f"mean_recall_{ep}"] = float(np.mean(rec))
            row[f"pred_dist_{ep}"] = json.dumps(pd_, sort_keys=True)
        cls_rows.append(row)
    write_csv(os.path.join(run_dir, "nlplr_class_checks.csv"), cls_rows)

    perf_rows = perf_table(runs)
    write_csv(os.path.join(run_dir, "perf.csv"), perf_rows)
    report(run_dir, sel, common, runs, tasks, summary, per_task, con_rows, pair_rows, bbh_rows, cls_rows,
           run_rows, perf_lines(run_dir, perf_rows))


MIB = 2 ** 20
TIME_PHASES = ("prefill", "prep", "forward", "backward", "clip", "optimizer")


def perf_table(runs):
    rows = []
    for (fam, task, arm, seed), r in sorted(runs.items()):
        p = r.get("perf")
        if not p:
            continue
        ph, st = p["phases"], p["static_bytes"]
        row = dict(family=fam, task=task, arm=arm, seed=seed, adaptation_s=p["adaptation_s"], eval_s=r["eval_s"],
                   unattributed_s=p["unattributed_s"], prefix_tokens=st.get("prefix_tokens"),
                   train_tokens=st.get("train_tokens"), train_tokens_padded=st.get("train_tokens_padded"),
                   steps=len(p["step_s"]), step_first_s=p["step_s"][0] if p["step_s"] else np.nan,
                   step_median_s=float(np.median(p["step_s"][1:])) if len(p["step_s"]) > 1 else np.nan)
        for k in TIME_PHASES:
            row[f"t_{k}_s"] = ph[k]["s"] if k in ph else 0.0
        for k in ("model_weights", "kv_cache", "grads", "optimizer_state"):
            row[f"m_{k}_mib"] = st.get(k, 0) / MIB
        fw, bw = ph.get("forward", {}), ph.get("backward", {})
        row["m_resident_train_mib"] = fw.get("resident_at_start_bytes", np.nan) / MIB
        row["m_other_resident_mib"] = row["m_resident_train_mib"] - sum(
            row[f"m_{k}_mib"] for k in ("model_weights", "kv_cache", "grads", "optimizer_state"))
        row["m_forward_transient_mib"] = fw.get("transient_bytes", np.nan) / MIB
        row["m_backward_transient_mib"] = bw.get("transient_bytes", np.nan) / MIB
        row["m_prefill_transient_mib"] = ph["prefill"]["transient_bytes"] / MIB
        row["m_eval_peak_mib"] = ph.get("eval", {}).get("peak_bytes", np.nan) / MIB
        row["m_peak_mib"] = p["overall_peak_bytes"] / MIB
        row["m_reserved_peak_mib"] = p["reserved_peak_bytes"] / MIB
        rows.append(row)
    return rows


def perf_lines(run_dir, rows):
    if not rows:
        return []
    L = ["## Performance: adaptation-time and memory breakdown", "",
         "Adaptation = from the start of the prefix forward to a tuned, servable KV cache (test evaluation excluded, "
         "reported separately). Each phase is bracketed by `torch.cuda.synchronize()`; the synchronizations are "
         "inside the measured time (overhead ≈1% vs. un-instrumented runs). Medians over jobs; memory in MiB of "
         "`torch.cuda.memory_allocated` (reserved = caching-allocator peak). Per-job values: `perf.csv`.", ""]
    for fam in ("bbh", "clf", "mc"):
        fr = [r for r in rows if r["family"] == fam]
        if not fr:
            continue
        med = lambda rs, k: float(np.nanmedian([r[k] for r in rs]))
        L += [f"### {FAMILY_NAME[fam]}", "", "**Time** (seconds per job, median)", "",
              "| arm | jobs | prefill | prep | forward | backward | clip | optimizer | other | adaptation | "
              "step 1 | steady step | eval (not in adaptation) |", "|---" * 14 + "|"]
        for arm in "ABCD":
            rs = [r for r in fr if r["arm"] == arm]
            if not rs:
                continue
            L.append(f"| {arm} | {len(rs)} | " + " | ".join(f"{med(rs, f't_{k}_s'):.3f}" for k in TIME_PHASES) +
                     f" | {med(rs, 'unattributed_s'):.3f} | {med(rs, 'adaptation_s'):.2f} | "
                     f"{med(rs, 'step_first_s'):.3f} | {med(rs, 'step_median_s'):.3f} | {med(rs, 'eval_s'):.2f} |")
        tot = sum(med(fr, f"t_{k}_s") for k in TIME_PHASES) + med(fr, "unattributed_s")
        L += ["", "Share of adaptation time (all arms): " + ", ".join(
            f"{k} {100 * med(fr, f't_{k}_s') / tot:.1f}%" for k in TIME_PHASES if med(fr, f"t_{k}_s") > 0) +
              f", other {100 * med(fr, 'unattributed_s') / tot:.1f}%.", "",
              "**Memory** (MiB, median)", "",
              "| arm | weights | KV cache | KV grads | Adam state | other resident | resident in training | "
              "prefill transient | forward transient | backward transient | peak | reserved peak | eval peak |",
              "|---" * 13 + "|"]
        for arm in "ABCD":
            rs = [r for r in fr if r["arm"] == arm]
            if not rs:
                continue
            ks = ("m_model_weights_mib", "m_kv_cache_mib", "m_grads_mib", "m_optimizer_state_mib",
                  "m_other_resident_mib", "m_resident_train_mib", "m_prefill_transient_mib",
                  "m_forward_transient_mib", "m_backward_transient_mib", "m_peak_mib", "m_reserved_peak_mib",
                  "m_eval_peak_mib")
            L.append(f"| {arm} | " + " | ".join(f"{med(rs, k):.1f}" for k in ks) + " |")
        x = np.array([r["prefix_tokens"] for r in fr], float)
        fits = []
        for k, unit, scale in (("adaptation_s", "ms", 1e3), ("m_kv_cache_mib", "MiB", 1),
                               ("m_forward_transient_mib", "MiB", 1), ("m_peak_mib", "MiB", 1)):
            y = np.array([r[k] for r in fr], float)
            ok = np.isfinite(y)
            if ok.sum() > 2 and np.ptp(x[ok]) > 0:
                b, a = np.polyfit(x[ok], y[ok], 1)
                rr = np.corrcoef(x[ok], y[ok])[0, 1]
                fits.append(f"{k}: {b * scale:.3f} {unit}/prefix token (intercept {a * scale:.1f}, r={rr:.2f})")
        L += ["", f"Prefix length: median {np.median(x):.0f} tokens (range {x.min():.0f}–{x.max():.0f}). "
              "Linear fits over jobs: " + "; ".join(fits) + ".", ""]
    sp = os.path.join(run_dir, "selection_timing.json")
    if os.path.exists(sp):
        st = json.load(open(sp))
        L += ["### Selection cost (CPU, refine.py)", "",
              "fit = TF-IDF + length buckets + class centres on the pool; select(1) = one candidate (the cost of "
              "A/B/C); select(32) − select(1) = D's extra search. Median over seeds, milliseconds.", "",
              "| family | tasks | pool size (median) | fit | select(1) | select(32) | D extra |", "|---|---|---|---|---|---|---|"]
        for fam in ("bbh", "clf", "mc"):
            rs = [v for v in st if v["family"] == fam]
            if rs:
                m = lambda k: float(np.median([v[k] for v in rs])) * 1e3
                L.append(f"| {fam} | {len({v['task'] for v in rs})} | {np.median([v['n'] for v in rs]):.0f} | "
                         f"{m('fit_s'):.1f} | {m('select1_s'):.1f} | {m('select32_s'):.1f} | "
                         f"{m('select32_s') - m('select1_s'):.1f} |")
        L += [""]
    return L


def write_csv(path, rows):
    if not rows:
        open(path, "w").close()
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
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def f(row, nm, so=False):
    if row is None or nm not in row:
        return "—"
    s = f"{row[nm]:+.2f} [{row[nm + '_lo']:+.2f}, {row[nm + '_hi']:+.2f}]"
    if so and f"{nm}_seedonly_lo" in row:
        s += f" (seed-only [{row[nm + '_seedonly_lo']:+.2f}, {row[nm + '_seedonly_hi']:+.2f}])"
    return s


def report(run_dir, sel, common, runs, tasks, summary, per_task, con_rows, pair_rows, bbh_rows, cls_rows, run_rows,
           perf_section=()):
    L = ["# CT-KV generic demonstration refinement · results", "",
         "Guideline: `docs/experiment-guideline/CTKV_Demonstration_Refinement_Guideline.md` "
         "(学习时间戳 2026-11-26 18:13:22 EST). Arms: A random, B class-equal, C pool-ratio quota, "
         "D refinement (min feature distance over 32 quota-respecting candidates; first candidate = C, or A "
         "without fixed classes). All numbers are accuracy in points on the frozen common test IDs; "
         "brackets = 95% paired bootstrap (5,000 draws, seeds × questions).", ""]
    L += ["## Headline", "",
          "| evaluation | tasks | D − A final | SD(D) − SD(A) | C − A final | D − C final |", "|---|---|---|---|---|---|"]
    for fam in ("bbh", "clf", "mc", "nlplr21"):
        da = summary.get((fam, "D", "A"))
        if da is None:
            continue
        name = FAMILY_NAME.get(fam, "NLP-LR all 21 tasks, equal weight")
        L.append(f"| {name} | {da.get('n_tasks', '')} | {f(da, 'final')} | {f(da, 'sd')} | "
                 f"{f(summary.get((fam, 'C', 'A')), 'final')} | {f(summary.get((fam, 'D', 'C')), 'final')} |")
    L += ["", "Reading rule (guideline §4): a gain is claimed only if the interval lies above zero; when SD drops "
          "while the mean is flat, only a stability improvement is reported.", ""]

    for fam in ("bbh", "clf", "mc"):
        if not any(k[0] == fam for k in summary):
            continue
        L += [f"## {FAMILY_NAME[fam]}", "", "### Mean over tasks (seed-mean first, tasks equal weight)", "",
              "| contrast | final | gain (final − step 0) | ICL (step 0) | SD diff |", "|---|---|---|---|---|"]
        for X, Y in CONTRASTS[fam]:
            r = summary.get((fam, X, Y))
            if r:
                L.append(f"| {X} − {Y} | {f(r, 'final', True)} | {f(r, 'gain')} | {f(r, 'icl')} | {f(r, 'sd')} |")
        L += [""]
        if fam != "bbh":
            L += ["### Per task (D − A)", "", "| task | seeds | common Q | final | gain | ICL | SD diff |",
                  "|---|---|---|---|---|---|---|"]
            for task in sorted(tasks[fam]):
                r = per_task.get((fam, task, "D", "A"))
                if r is None:
                    L.append(f"| {task} | — | — | not estimable | | | |")
                    continue
                L.append(f"| {task} | {r['n_seeds']} | {r['n_q']} | {f(r, 'final')} | {f(r, 'gain')} | "
                         f"{f(r, 'icl')} | {f(r, 'sd')} |")
            L += [""]
        # per-arm means and SDs
        L += ["### Per arm (mean over tasks of seed-mean accuracy; SD = mean over tasks of across-seed SD)", "",
              "| arm | step 0 | final | gain | across-seed SD (final) | adaptation s (median) |",
              "|---|---|---|---|---|---|"]
        fin = FINAL[fam]
        for arm in "ABCD":
            m0, m1, sds, ad = [], [], [], []
            for task in sorted(tasks[fam]):
                rr0 = [r["acc_common"] for r in run_rows if r["family"] == fam and r["task"] == task
                       and r["arm"] == arm and r["step"] == 0]
                rr1 = [r["acc_common"] for r in run_rows if r["family"] == fam and r["task"] == task
                       and r["arm"] == arm and str(r["step"]) == fin]
                ad += [r["adapt_s"] for r in run_rows if r["family"] == fam and r["task"] == task
                       and r["arm"] == arm and str(r["step"]) == fin]
                if rr1:
                    m0.append(np.mean(rr0)), m1.append(np.mean(rr1))
                    sds.append(np.std(rr1, ddof=1) if len(rr1) > 1 else np.nan)
            if m1:
                L.append(f"| {arm} | {np.mean(m0):.2f} | {np.mean(m1):.2f} | {np.mean(m1) - np.mean(m0):+.2f} | "
                         f"{np.nanmean(sds):.2f} | {np.median(ad):.1f} |")
        L += [""]

    if bbh_rows:
        L += ["## BBH hypothesis checks (mean over 20 seeds)", "",
              "| arm | demo True | pred True @0 | pred True @16 | recall T @0/@16 | recall F @0/@16 | invalid @0/@16 | "
              "fixed 0→16 | broken 0→16 |", "|---|---|---|---|---|---|---|---|---|"]
        for arm in "ABCD":
            rs = [r for r in bbh_rows if r["arm"] == arm]
            if not rs:
                continue
            m = lambda k: np.mean([r[k] for r in rs])
            L.append(f"| {arm} | {m('demo_true'):.2f} | {m('pred_true_0'):.3f} | {m('pred_true_16'):.3f} | "
                     f"{m('recall_True_0'):.3f}/{m('recall_True_16'):.3f} | {m('recall_False_0'):.3f}/"
                     f"{m('recall_False_16'):.3f} | {m('invalid_0'):.2f}/{m('invalid_16'):.2f} | "
                     f"{m('fixed_0to16'):.1f} | {m('broken_0to16'):.1f} |")
        L += ["", "Per-seed scores: `bbh_checks.csv` (acc_0, acc_16 per arm × seed). Prediction-True denominators "
              "are all 200 questions.", ""]
        L += ["| seed | " + " | ".join(f"{a} @0/@16" for a in "ABCD") + " |", "|---" * 5 + "|"]
        for s in sorted({r["seed"] for r in bbh_rows}):
            cells = []
            for a in "ABCD":
                r = [x for x in bbh_rows if x["arm"] == a and x["seed"] == s]
                cells.append(f"{r[0]['acc_0']:.1f}/{r[0]['acc_16']:.1f}" if r else "—")
            L.append(f"| {s} | " + " | ".join(cells) + " |")
        L += [""]

    if cls_rows:
        L += ["## NLP-LR classification label checks (mean over tasks × seeds)", "",
              "| arm | demo TV to pool | pred TV to pool @0 / @200 | pred TV to test gold @0 / @200 | "
              "mean per-class recall @0 / @200 | macro-F1 @0 / @200 |", "|---|---|---|---|---|---|"]
        for arm in "ABCD":
            rs = [r for r in cls_rows if r["arm"] == arm]
            if not rs:
                continue
            m = lambda k: np.mean([r[k] for r in rs])
            mf = lambda st: np.nanmean([r["macro_f1_common"] for r in run_rows if r["family"] == "clf"
                                        and r["arm"] == arm and r["step"] == st])
            L.append(f"| {arm} | {m('demo_tv_pool'):.3f} | {m('pred_tv_pool_0'):.3f} / {m('pred_tv_pool_200'):.3f} | "
                     f"{m('pred_tv_gold_0'):.3f} / {m('pred_tv_gold_200'):.3f} | {m('mean_recall_0'):.3f} / "
                     f"{m('mean_recall_200'):.3f} | {mf(0):.2f} / {mf(200):.2f} |")
        L += ["", "Per-class recall per task × arm × seed: `nlplr_class_checks.csv`. A prediction mix closer to the "
              "pool is not by itself counted as more correct answers.", ""]

    L += ["## Coverage", "", "| task | family | jobs ok | test file | evaluable union | common | coverage |",
          "|---|---|---|---|---|---|---|"]
    for t, c in sorted(common.items()):
        cov = c["n_common"] / c["n_union"] if c["n_union"] else 0
        L.append(f"| {t} | {c['family']} | {c['n_jobs_ok']}/{c['n_jobs']} | {c['n_test_file']} | {c['n_union']} | "
                 f"{c['n_common']} | {cov:.2f} |")
    L += [""]
    if pair_rows:
        L += ["### Sensitivity: pairwise intersections (D − A, NLP-LR)", "",
              "| task | Q pair | Q global | final (pair) | final (global) |", "|---|---|---|---|---|"]
        for r in pair_rows:
            if r["contrast"] != "D-A":
                continue
            g = per_task.get((r["family"], r["task"], "D", "A"))
            L.append(f"| {r['task']} | {r['n_q_pair']} | {r['n_q_global']} | {r['final']:+.2f} | "
                     f"{g['final']:+.2f} |" if g else f"| {r['task']} | {r['n_q_pair']} | {r['n_q_global']} | "
                     f"{r['final']:+.2f} | — |")
        L += [""]
    L += list(perf_section)
    open(os.path.join(run_dir, "REPORT_results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main(sys.argv[1])
