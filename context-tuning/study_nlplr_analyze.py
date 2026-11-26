"""Analysis of the NLP-LR side of the CT-KV accuracy study. Writes REPORT_nlp_lr.md, runs.csv, predictions.jsonl
and hypothesis_contrasts.csv into results/ctkv_accuracy_study/nlp_lr/.

Primary metric: the paper's per-task score (macro-F1 for classification, accuracy otherwise), in points.
Intervals: paired bootstrap (2000 draws) that resamples seeds and test items jointly; the same item draw is shared
across seeds and across the conditions being compared.

Usage (from context-tuning/): .venv/bin/python study_nlplr_analyze.py
"""
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

from study_nlplr_jobs import SPLITS, TASKS, is_clf

ROOT = "../results/ctkv_accuracy_study/nlp_lr"
LOGS = "../results/nlp_lr_repro/logs"
B = 2000
RNG = np.random.default_rng(0)
_cache = {}


def load(path):
    if path not in _cache:
        _cache[path] = json.load(open(path)) if os.path.exists(path) else None
    return _cache[path]


def rid(exp, name):
    return load(os.path.join(ROOT, exp, f"{name}.json"))


def macro_f1(pred, gold, classes):
    f1s = []
    for c in classes:
        g = gold == c
        if not g.any():
            continue
        p = pred == c
        prec = (p & g).sum() / p.sum() if p.any() else 1.0  # MetaICL: precision 1 for never-predicted labels
        rec = (p & g).sum() / g.sum()
        f1s.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def task_score(pred, gold, task, classes):
    return macro_f1(pred, gold, classes) if is_clf(task) else float((pred == gold).mean())


def arrays(res, step, item_ids=None):
    """(pred labels, gold labels) as numpy object arrays ordered by test id, restricted to item_ids."""
    rows = {r[0]: r for r in res["checkpoints"][str(step)]["predictions"]}
    ids = sorted(rows) if item_ids is None else item_ids
    return np.array([rows[i][1] for i in ids], dtype=object), np.array([rows[i][2] for i in ids], dtype=object), ids


def common_items(results, step):
    sets = [set(r[0] for r in res["checkpoints"][str(step)]["predictions"]) for res in results]
    return sorted(set.intersection(*sets))


def paired_diff(pairs, task, step_a, step_b=None, stat="score"):
    """pairs: list over seeds of (result_a, result_b[, ...]). Returns mean difference (a - b) in points with a
    bootstrap 95% interval resampling seeds and items (item draw shared across seeds)."""
    step_b = step_a if step_b is None else step_b
    per_seed = []
    for ra, rb in pairs:
        items = sorted(set(common_items([ra], step_a)) & set(common_items([rb], step_b)))
        pa, ga, _ = arrays(ra, step_a, items)
        pb, gb, _ = arrays(rb, step_b, items)
        per_seed.append((pa, pb, ga))
    if not per_seed:
        return float("nan"), (float("nan"), float("nan")), 0
    classes = sorted({g for _, _, ga in per_seed for g in ga})
    f = (lambda p, g: task_score(p, g, task, classes)) if stat == "score" else stat
    est = np.mean([f(pa, g) - f(pb, g) for pa, pb, g in per_seed]) * 100
    n_items = min(len(g) for _, _, g in per_seed)
    boots = []
    for _ in range(B):
        seeds = RNG.integers(0, len(per_seed), len(per_seed))
        u = RNG.random(n_items)  # shared item draw, mapped onto each seed's item list
        vals = []
        for s in seeds:
            pa, pb, g = per_seed[s]
            idx = np.minimum((u * len(g)).astype(int), len(g) - 1)
            vals.append(f(pa[idx], g[idx]) - f(pb[idx], g[idx]))
        boots.append(np.mean(vals) * 100)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, (lo, hi), len(per_seed)


def fmt_ci(est, ci):
    return f"{est:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"


def log_scores(name, s):
    txt = open(os.path.join(LOGS, f"{name}_{s}.log")).read()
    return {t: float(v) * 100 for t, v in re.findall(r"^(\S+) clf \w+ has a score ([\d.]+)", txt, re.M)}


def frac(pred, label):
    return float(np.mean(pred == label))


# ------------------------------------------------------------------ sections

def section_e0(out, rows_csv):
    out.append("## E0 · natural 16-shot sets (5 splits × 21 tasks), original loss\n")
    out.append("Checkpoints of one 200-step trajectory; step 0 = ICL with the same cache. Paper reproduction logs for reference.\n")
    out.append("| split | step 0 | 50 | 100 | 200 | repro ICL | repro CT-KV |\n|---|---|---|---|---|---|---|")
    allm = defaultdict(list)
    for s in SPLITS:
        res = [rid("e0", f"e0_s{s}_{t}") for t in TASKS]
        if any(r is None for r in res):
            out.append(f"| {s} | incomplete ({sum(r is not None for r in res)}/21) |||||")
            continue
        m = {cp: np.mean([r["checkpoints"][cp]["score"] for r in res]) * 100 for cp in ("0", "50", "100", "200")}
        for cp, v in m.items():
            allm[cp].append(v)
        icl, ct = log_scores("icl", s), log_scores("ctkv", s)
        out.append(f"| {s} | {m['0']:.2f} | {m['50']:.2f} | {m['100']:.2f} | {m['200']:.2f} | "
                   f"{np.mean(list(icl.values())):.2f} | {np.mean(list(ct.values())):.2f} |")
        for r in res:
            for cp, c in r["checkpoints"].items():
                rows_csv.append(dict(exp="e0", task=r["task"], split=s, seed=s, select="natural", loss="raw", step=cp,
                                     score=c["score"] * 100, n_test=r["n_test"], adapt_s=r["timing"]["adapt_s"],
                                     peak_mb=r["timing"]["peak_mem_mb"], run=r["id"]))
    if allm:
        out.append(f"| mean | " + " | ".join(f"{np.mean(allm[c]):.2f}" for c in ("0", "50", "100", "200")) + " | | |")
    # majority-label drift on classification tasks
    out.append("\n### Does CT-KV push predictions toward the demonstrations' majority label? (classification tasks)\n")
    out.append("For each task × split: m = majority label of the 16 demos. Rate = share of test predictions equal to m.\n")
    out.append("| task | demo share of m (mean over splits) | gold share of m | pred share step 0 | step 200 | Δ (200−0) | splits with Δ>0 |")
    out.append("|---|---|---|---|---|---|---|")
    drift_all = []
    for t in [t for t in TASKS if is_clf(t)]:
        vals = []
        for s in SPLITS:
            r = rid("e0", f"e0_s{s}_{t}")
            if r is None:
                continue
            labels = Counter(r["demo_labels"])
            m, cnt = max(sorted(labels.items()), key=lambda x: x[1])
            p0, g, _ = arrays(r, 0)
            p2, _, _ = arrays(r, 200)
            vals.append((cnt / 16, frac(g, m), frac(p0, m), frac(p2, m)))
        if not vals:
            continue
        v = np.array(vals)
        d = v[:, 3] - v[:, 2]
        drift_all += [(t, *x) for x in vals]
        out.append(f"| {t} | {v[:, 0].mean():.2f} | {v[:, 1].mean():.2f} | {v[:, 2].mean():.2f} | {v[:, 3].mean():.2f} | "
                   f"{d.mean() * 100:+.1f} pts | {(d > 0).sum()}/{len(d)} |")
    if drift_all:
        a = np.array([x[1:] for x in drift_all])
        d = a[:, 3] - a[:, 2]
        excess = a[:, 0] - a[:, 1]  # demo share of m minus gold share of m
        r = np.corrcoef(excess, d)[0, 1] if len(d) > 2 else float("nan")
        out.append(f"\nAcross {len(d)} task×split runs: mean Δ {d.mean() * 100:+.1f} pts, Δ>0 in {(d > 0).sum()}; "
                   f"Pearson r between (demo share − gold share of m) and Δ: {r:.2f}.")
    # distance of the predicted label distribution to the demo label distribution and to the gold distribution
    out.append("\n### Predicted label distribution vs demo and gold distributions (classification tasks)\n")
    out.append("TV = total variation distance. If CT-KV copied the demonstrations' label mix, TV(pred, demo) would fall "
               "below TV(pred, gold) after training. Rows: mean over splits.\n")
    out.append("| task | TV(demo, gold) | TV(pred, demo) step 0 → 200 | TV(pred, gold) step 0 → 200 | runs where step-200 pred is closer to demo than to gold |")
    out.append("|---|---|---|---|---|")
    closer, n_all = 0, 0
    for t in [t for t in TASKS if is_clf(t)]:
        vals = []
        for s in SPLITS:
            r = rid("e0", f"e0_s{s}_{t}")
            if r is None:
                continue
            p0, g, _ = arrays(r, 0)
            p2, _, _ = arrays(r, 200)
            labels = sorted(set(g) | set(r["demo_labels"]))
            dist = lambda xs: np.array([np.mean(np.asarray(xs, dtype=object) == l) for l in labels])
            dd, dg = dist(r["demo_labels"]), dist(g)
            tv = lambda a, b: 0.5 * np.abs(a - b).sum()
            v = (tv(dd, dg), tv(dist(p0), dd), tv(dist(p2), dd), tv(dist(p0), dg), tv(dist(p2), dg))
            vals.append(v)
            closer += v[2] < v[4]
            n_all += 1
        if vals:
            v = np.array(vals)
            out.append(f"| {t} | {v[:, 0].mean():.2f} | {v[:, 1].mean():.2f} → {v[:, 2].mean():.2f} | "
                       f"{v[:, 3].mean():.2f} → {v[:, 4].mean():.2f} | {(v[:, 2] < v[:, 4]).sum()}/{len(v)} |")
    if n_all:
        out.append(f"\nOverall: step-200 predictions closer to the demo mix than to the gold mix in {closer}/{n_all} runs.")


def h1_rate(res, step, label):
    p, _, _ = arrays(res, step)
    return frac(p, label)


def section_h1(out, contrasts):
    out.append("\n## §2.1 analogue · label ratio 3/16, 8/16, 13/16 (binary tasks), original loss\n")
    out.append("p_t(r) = share of test predictions equal to label A (all test items in the denominator); "
               "D_t = p_t(A13) − p_t(A3); amplification A = D_t − D_0. Forward and reversed order are averaged within a seed first.\n")
    for t in sorted({os.path.basename(p).split("_s")[0][3:] for p in glob.glob(os.path.join(ROOT, "h1", "h1_*.json"))}):
        per_seed = defaultdict(dict)
        label_a = None
        for seed in range(5):
            for order in ("fwd", "rev"):
                rs = {r: rid("h1", f"h1_{t}_s{seed}_{r}_{order}") for r in ("A3", "A8", "A13")}
                if any(v is None for v in rs.values()):
                    continue
                label_a = rs["A3"]["meta"]["label_A"]
                for step in (0, 50, 100, 200):
                    p = {r: h1_rate(v, step, label_a) for r, v in rs.items()}
                    sc = {r: v["checkpoints"][str(step)]["score"] * 100 for r, v in rs.items()}
                    per_seed[seed].setdefault(step, []).append((p, sc))
                    contrasts.append(dict(exp="h1", task=t, seed=seed, order=order, step=step, label_A=label_a,
                                          p_A3=p["A3"], p_A8=p["A8"], p_A13=p["A13"], D=p["A13"] - p["A3"],
                                          score_A3=sc["A3"], score_A8=sc["A8"], score_A13=sc["A13"]))
        if not per_seed:
            continue
        out.append(f"\n### {t} (label A = {label_a})\n")
        out.append("| step | p(A3) | p(A8) | p(A13) | D | A = D − D0 (mean, seed-bootstrap 95%) | score A3 / A8 / A13 |")
        out.append("|---|---|---|---|---|---|---|")
        D0 = {s: np.mean([x[0]["A13"] - x[0]["A3"] for x in v[0]]) for s, v in per_seed.items()}
        for step in (0, 50, 100, 200):
            ps = {r: np.mean([np.mean([x[0][r] for x in v[step]]) for v in per_seed.values()]) for r in ("A3", "A8", "A13")}
            sc = {r: np.mean([np.mean([x[1][r] for x in v[step]]) for v in per_seed.values()]) for r in ("A3", "A8", "A13")}
            Ds = np.array([np.mean([x[0]["A13"] - x[0]["A3"] for x in v[step]]) - D0[s] for s, v in per_seed.items()])
            bs = [RNG.choice(Ds, len(Ds)).mean() for _ in range(B)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out.append(f"| {step} | {ps['A3']:.3f} | {ps['A8']:.3f} | {ps['A13']:.3f} | {ps['A13'] - ps['A3']:+.3f} | "
                       f"{Ds.mean() * 100:+.1f} pts [{lo * 100:+.1f}, {hi * 100:+.1f}] (n={len(Ds)}) | "
                       f"{sc['A3']:.1f} / {sc['A8']:.1f} / {sc['A13']:.1f} |")
        between = np.mean([np.mean([min(x[0]["A3"], x[0]["A13"]) <= x[0]["A8"] <= max(x[0]["A3"], x[0]["A13"])
                                    for x in v[200]]) for v in per_seed.values()])
        out.append(f"\nShare of (seed, order) where p(A8) lies between p(A3) and p(A13) at step 200: {between:.2f}.")


def section_e2(out, contrasts):
    out.append("\n## E1/E2 analogue · selection (natural vs stratified) × loss (raw vs class-balanced), classification tasks\n")
    out.append("Mean over 5 seeds of the task score at step 200 (and the step-0 → 200 gain). natural+raw is E0.\n")
    out.append("| task | natural raw | natural balanced | stratified raw | stratified balanced | gain nat-raw | gain strat-raw |")
    out.append("|---|---|---|---|---|---|---|")
    cells = defaultdict(list)
    pairs = defaultdict(list)
    for t in [t for t in TASKS if is_clf(t)]:
        row = {}
        for i, s in enumerate(SPLITS):
            nr = rid("e0", f"e0_s{s}_{t}")
            nb = rid("e2", f"e2_{t}_s{i}_natural_balanced")
            sr = rid("e2", f"e2_{t}_s{i}_strat_raw")
            sb = rid("e2", f"e2_{t}_s{i}_strat_balanced")
            if None in (nr, nb, sr, sb):
                continue
            for k, r in (("nr", nr), ("nb", nb), ("sr", sr), ("sb", sb)):
                row.setdefault(k, []).append(r["checkpoints"]["200"]["score"] * 100)
                row.setdefault(k + "0", []).append(r["checkpoints"]["0"]["score"] * 100)
            pairs[(t, "loss")].append((nb, nr))
            pairs[(t, "select")].append((sr, nr))
        if not row:
            continue
        m = {k: np.mean(v) for k, v in row.items()}
        for k in ("nr", "nb", "sr", "sb"):
            cells[k].append(m[k])
        out.append(f"| {t} | {m['nr']:.2f} | {m['nb']:.2f} | {m['sr']:.2f} | {m['sb']:.2f} | {m['nr'] - m['nr0']:+.2f} | "
                   f"{m['sr'] - m['sr0']:+.2f} |")
    if cells:
        out.append(f"| **mean** | {np.mean(cells['nr']):.2f} | {np.mean(cells['nb']):.2f} | {np.mean(cells['sr']):.2f} | "
                   f"{np.mean(cells['sb']):.2f} | | |")
        out.append("\nPaired differences at step 200 (points, bootstrap over seeds × items):\n")
        out.append("| task | balanced − raw (natural demos) | stratified − natural (raw loss) |\n|---|---|---|")
        for t in [t for t in TASKS if is_clf(t)]:
            if not pairs[(t, "loss")]:
                continue
            a = paired_diff(pairs[(t, "loss")], t, 200)
            b = paired_diff(pairs[(t, "select")], t, 200)
            out.append(f"| {t} | {fmt_ci(a[0], a[1])} | {fmt_ci(b[0], b[1])} |")
            contrasts.append(dict(exp="e2", task=t, contrast="balanced-raw", step=200, diff=a[0], lo=a[1][0], hi=a[1][1]))
            contrasts.append(dict(exp="e2", task=t, contrast="stratified-natural", step=200, diff=b[0], lo=b[1][0], hi=b[1][1]))


def section_h2(out, contrasts):
    out.append("\n## §4.1 analogue · minimal replacement to 8/8 vs sham (same positions, same labels), original loss\n")
    out.append("### Natural 16-shot sets (performance)\n")
    out.append("| task | seeds | replaced (mean) | original @200 | repair @200 | sham @200 | repair − original | repair − sham |")
    out.append("|---|---|---|---|---|---|---|---|")
    for t in [t for t in TASKS if is_clf(t)]:
        prs, sh, n_rep, sc = [], [], [], defaultdict(list)
        for i, s in enumerate(SPLITS):
            rr, rs, ro = rid("h2", f"h2nat_{t}_s{i}_repair"), rid("h2", f"h2nat_{t}_s{i}_sham"), rid("e0", f"e0_s{s}_{t}")
            if None in (rr, rs, ro):
                continue
            prs.append((rr, ro))
            sh.append((rr, rs))
            n_rep.append(rr["meta"]["n_replaced"])
            for k, r in (("o", ro), ("r", rr), ("s", rs)):
                sc[k].append(r["checkpoints"]["200"]["score"] * 100)
        if not prs:
            continue
        a, b = paired_diff(prs, t, 200), paired_diff(sh, t, 200)
        out.append(f"| {t} | {len(prs)} | {np.mean(n_rep):.1f} | {np.mean(sc['o']):.2f} | {np.mean(sc['r']):.2f} | "
                   f"{np.mean(sc['s']):.2f} | {fmt_ci(*a[:2])} | {fmt_ci(*b[:2])} |")
        contrasts.append(dict(exp="h2_natural", task=t, contrast="repair-original", step=200, diff=a[0], lo=a[1][0], hi=a[1][1]))
        contrasts.append(dict(exp="h2_natural", task=t, contrast="repair-sham", step=200, diff=b[0], lo=b[1][0], hi=b[1][1]))
    out.append("\n### Skewed groups from §2.1 (mechanism)\n")
    out.append("D_t = p_t(from A13) − p_t(from A3) for raw / sham / repair versions; A = D_200 − D_0. "
               "After repair both groups are 8/8, so A_repair measures what is left of the source skew.\n")
    out.append("| task | A_raw | A_sham | A_repair | A_sham − A_repair | A_raw − A_repair |\n|---|---|---|---|---|---|")
    for t in sorted({os.path.basename(p).split("_s")[0][7:] for p in glob.glob(os.path.join(ROOT, "h2", "h2skew_*.json"))}):
        vals = defaultdict(list)
        for seed in range(5):
            per = defaultdict(list)
            for order in ("fwd", "rev"):
                raw = {r: rid("h1", f"h1_{t}_s{seed}_{r}_{order}") for r in ("A3", "A13")}
                rep = {r: rid("h2", f"h2skew_{t}_s{seed}_{r}_{order}_repair") for r in ("A3", "A13")}
                sh = {r: rid("h2", f"h2skew_{t}_s{seed}_{r}_{order}_sham") for r in ("A3", "A13")}
                if any(v is None for d in (raw, rep, sh) for v in d.values()):
                    continue
                la = raw["A3"]["meta"]["label_A"]
                for name, d in (("raw", raw), ("sham", sh), ("repair", rep)):
                    A = (h1_rate(d["A13"], 200, la) - h1_rate(d["A3"], 200, la)) - (h1_rate(d["A13"], 0, la) - h1_rate(d["A3"], 0, la))
                    per[name].append(A)
            if per:
                for k, v in per.items():
                    vals[k].append(np.mean(v))
        if not vals:
            continue
        v = {k: np.array(x) for k, x in vals.items()}

        def ci(x):
            bs = [RNG.choice(x, len(x)).mean() for _ in range(B)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            return f"{x.mean() * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]"
        out.append(f"| {t} (n={len(v['raw'])}) | {ci(v['raw'])} | {ci(v['sham'])} | {ci(v['repair'])} | "
                   f"{ci(v['sham'] - v['repair'])} | {ci(v['raw'] - v['repair'])} |")
        contrasts.append(dict(exp="h2_skewed", task=t, A_raw=v["raw"].mean(), A_sham=v["sham"].mean(),
                              A_repair=v["repair"].mean()))


def section_e3(out, contrasts):
    out.append("\n## E3/E4 analogue · choosing the whole 16-shot set by dev NLL (30 candidates / 20 dev from the pool)\n")
    picks_path = os.path.join(ROOT, "e3final", "e3_picks.json")
    if not os.path.exists(picks_path):
        out.append("(final evaluations not run yet)")
        return
    picks = json.load(open(picks_path))
    out.append("Rules: pre-specified random group (c0), lowest dev NLL at step 0 (ICL), lowest dev NLL at step 50 of 200. "
               "Final scores: Full CT-KV@200 on the test set (ICL = step 0 of the same run).\n")
    out.append("| task | random @200 | NLL@0 @200 | NLL@50 @200 | NLL@50 − NLL@0 | NLL@50 − random | same pick (0 vs 50) | "
               "early-stop (random) | Spearman(NLL0, NLL50) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    tot = defaultdict(list)
    for t in TASKS:
        ps = [p for p in picks if p["task"] == t]
        sc = defaultdict(list)
        pair_a, pair_b = [], []
        same, rho, es = 0, [], []
        for p in ps:
            r = {k: rid("e3final", f"e3final_{t}_s{p['seed']}_c{c}") for k, c in p["rule"].items()}
            if any(v is None for v in r.values()):
                continue
            for k, v in r.items():
                sc[k].append(v["checkpoints"]["200"]["score"] * 100)
            pair_a.append((r["nll50"], r["nll0"]))
            pair_b.append((r["nll50"], r["random"]))
            same += p["rule"]["nll0"] == p["rule"]["nll50"]
            rank = lambda x: np.argsort(np.argsort(x))
            rho.append(np.corrcoef(rank(p["nll0"]), rank(p["nll50"]))[0, 1])
            e = rid("e3", f"e3es_{t}_s{p['seed']}_c0")
            if e is not None:
                best = min((e["dev"][k]["nll"], int(k)) for k in e["dev"])[1]  # ties -> earlier step
                es.append(e["checkpoints"][str(best)]["score"] * 100)
        if not pair_a:
            continue
        a, b = paired_diff(pair_a, t, 200), paired_diff(pair_b, t, 200)
        m = {k: np.mean(v) for k, v in sc.items()}
        for k, v in m.items():
            tot[k].append(v)
        tot["es"].append(np.mean(es) if es else float("nan"))
        out.append(f"| {t} | {m['random']:.2f} | {m['nll0']:.2f} | {m['nll50']:.2f} | {fmt_ci(*a[:2])} | {fmt_ci(*b[:2])} | "
                   f"{same}/{len(pair_a)} | {np.mean(es):.2f} | {np.mean(rho):.2f} |")
        contrasts.append(dict(exp="e3", task=t, contrast="nll50-nll0", step=200, diff=a[0], lo=a[1][0], hi=a[1][1]))
        contrasts.append(dict(exp="e3", task=t, contrast="nll50-random", step=200, diff=b[0], lo=b[1][0], hi=b[1][1]))
    if tot:
        out.append(f"| **mean** | {np.mean(tot['random']):.2f} | {np.mean(tot['nll0']):.2f} | {np.mean(tot['nll50']):.2f} | | | | "
                   f"{np.nanmean(tot['es']):.2f} | |")


def section_timing(out):
    files = glob.glob(os.path.join(ROOT, "timing", "time_*.json"))
    if not files:
        return
    out.append("\n## §6.1 analogue · deployment time (adaptation only, no evaluation), median of 3 repeats\n")
    t = defaultdict(list)
    for f in files:
        r = load(f)
        m = r["meta"]
        t[(r["task"], m["seed"], m["cond"], m["stop"])].append(r["timing"]["adapt_s"])
    med = {k: float(np.median(v)) for k, v in t.items()}
    out.append("| task | seed | raw@200 s | refined@100 s | ratio | refined@100 − raw@200 (score) |\n|---|---|---|---|---|---|")
    for (task, seed, cond, stop), v in sorted(med.items()):
        if cond != "raw" or stop != 200 or (task, seed, "refined", 100) not in med:
            continue
        rr = rid("h2", f"h2nat_{task}_s{seed}_repair")
        ro = rid("e0", f"e0_s{SPLITS[seed]}_{task}")
        d = (rr["checkpoints"]["100"]["score"] - ro["checkpoints"]["200"]["score"]) * 100 if rr and ro else float("nan")
        out.append(f"| {task} | {seed} | {v:.2f} | {med[(task, seed, 'refined', 100)]:.2f} | "
                   f"{med[(task, seed, 'refined', 100)] / v:.2f} | {d:+.2f} |")


def all_runs_rows():
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "*", "*.json"))):
        if os.path.basename(p).startswith(("jobs_", "e3_picks")):
            continue
        r = load(p)
        if not r or "timing" not in r:
            continue
        m = r.get("meta", {})
        base = dict(run=r["id"], exp=m.get("exp"), task=r["task"], seed=m.get("seed", m.get("split")), loss=r["loss"],
                    select=m.get("select", m.get("kind", m.get("ratio"))), order=m.get("order"), n_shots=len(r["demo_labels"]),
                    prefix_tokens=r["prefix_len"], kv_mib=r["prefix_len"] * 36 * 2 * 1280 * 2 / 2**20,
                    adapt_s=r["timing"]["adapt_s"], peak_mb=r["timing"]["peak_mem_mb"], n_test=r["n_test"],
                    demo_label_counts=json.dumps(Counter(r["demo_labels"])), stop_at=r.get("stop_at"))
        for step, c in r["checkpoints"].items():
            n_ok = sum(x[1] == x[2] for x in c["predictions"])
            rows.append(dict(base, step=int(step), score=c["score"] * 100, n_correct=n_ok, n_scored=len(c["predictions"])))
        for step, d in r["dev"].items():
            rows.append(dict(base, step=int(step), dev_nll=d["nll"], dev_n=d["n"]))
        if not r["checkpoints"] and not r["dev"]:
            rows.append(base)
    return rows


def export_predictions():
    path = os.path.join(ROOT, "predictions.jsonl")
    with open(path, "w") as f:
        for p in sorted(glob.glob(os.path.join(ROOT, "*", "*.json"))):
            if os.path.basename(p).startswith(("jobs_", "e3_picks")):
                continue
            r = load(p)
            if not r or "checkpoints" not in r:
                continue
            for step, c in r["checkpoints"].items():
                for test_id, pred, gold, losses in c["predictions"]:
                    f.write(json.dumps(dict(run=r["id"], task=r["task"], step=int(step), test_id=test_id, gold=gold,
                                            prediction=pred, correct=pred == gold, option_losses=losses)) + "\n")


def main():
    out = ["# CT-KV accuracy study · NLP-LR (GPT-2 Large, 16 shots, 200 CT-KV steps)\n",
           "Generated by `context-tuning/study_nlplr_analyze.py`. Jobs: `context-tuning/study_nlplr_jobs.py`; runner: "
           "`context-tuning/study_nlplr.py`. Scores are the paper's per-task metric in points.\n"]
    rows_csv, contrasts = [], []
    section_e0(out, rows_csv)
    section_h1(out, contrasts)
    section_e2(out, contrasts)
    section_h2(out, contrasts)
    section_e3(out, contrasts)
    section_timing(out)
    rows_csv = all_runs_rows()
    open(os.path.join(ROOT, "REPORT_nlp_lr.md"), "w").write("\n".join(out) + "\n")
    for name, rows in (("runs.csv", rows_csv), ("hypothesis_contrasts.csv", contrasts)):
        if rows:
            keys = sorted({k for r in rows for k in r})
            with open(os.path.join(ROOT, name), "w", newline="") as f:
                w = csv.DictWriter(f, keys)
                w.writeheader()
                w.writerows(rows)
    export_predictions()
    print("\n".join(out))


if __name__ == "__main__":
    main()
