"""Stage 3 round 1 analysis: half-delta KV update and same-trajectory early stopping (arm D, production dense).

Reads RUN_DIR/raw_{bbh,nlplr}/*.json (one trajectory per task x seed, five conditions) and the job lists
RUN_DIR/jobs_{bbh,nlplr}.jsonl (expected panel), plus STAGE1/common_test_ids.json (NLP-LR common test IDs; BBH uses
the job's eval_ids). Main metric: per-question accuracy (points), seed mean per task, tasks equal weight within each
family (bbh / clf / mc). Bootstrap: 5,000 paired draws, within each task one seed index and one question index shared
by all conditions; seed-only intervals reported too. Missing / unscorable questions are not counted as wrong: each
task uses the questions present in every condition of every seed, and coverage.csv records what was dropped.

Writes RUN_DIR/{coverage.csv, conditions.csv, contrasts.csv, per_seed.csv, bbh_diagnostics.csv, cost.csv,
geometry.csv, predictions.jsonl.gz, REPORT_results.md}.
Usage (from context-tuning/): python stage3_analyze.py RUN_DIR STAGE1_DIR
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
CONDS = ["icl", "early_quarter", "early_half", "full", "half_delta"]
FAMS = ["bbh", "clf", "mc"]
FAM_NAME = {"bbh": "BBH Boolean (Llama-3-8B-Instruct, 10-shot, T = 16)",
            "clf": "NLP-LR classification (GPT-2 Large, 16-shot, T = 200)",
            "mc": "NLP-LR multiple choice (GPT-2 Large, 16-shot, T = 200)"}
STEPS = {"bbh": {"icl": 0, "early_quarter": 4, "early_half": 8, "full": 16, "half_delta": 16},
         "nlplr": {"icl": 0, "early_quarter": 50, "early_half": 100, "full": 200, "half_delta": 200}}
CONTRASTS = [("half_delta", "full", "main: half_delta − full"),
             ("half_delta", "icl", "half_delta − icl"),
             ("full", "icl", "full − icl"),
             ("early_half", "full", "early_half − full"),
             ("early_quarter", "full", "early_quarter − full"),
             ("early_half", "half_delta", "early_half − half_delta"),
             ("early_quarter", "icl", "early_quarter − icl"),
             ("early_half", "icl", "early_half − icl")]


def load(run_dir, stage1):
    common = json.load(open(os.path.join(stage1, "common_test_ids.json")))
    kinds = {t: c["family"] for t, c in common.items()}
    corr = defaultdict(lambda: defaultdict(dict))     # [(fam, task)][cond][seed] = {qid: 0/1}
    preds = defaultdict(lambda: defaultdict(dict))    # [(fam, task)][cond][seed] = {qid: pred}
    gold = defaultdict(dict)                          # [(fam, task)][qid] = gold
    meta, expected = {}, {}
    for src in ("bbh", "nlplr"):
        jp = os.path.join(run_dir, f"jobs_{src}.jsonl")
        for j in map(json.loads, open(jp)):
            f = "bbh" if src == "bbh" else kinds[j["task"]]
            expected[(f, j["task"], j["meta"]["seed"])] = j
        for p in sorted(glob.glob(os.path.join(run_dir, f"raw_{src}", "*.json"))):
            r = json.load(open(p))
            task, seed = r["task"], r["meta"]["seed"]
            f = "bbh" if src == "bbh" else kinds[task]
            meta[(f, task, seed)] = r
            for cond, c in r["conditions"].items():
                if src == "bbh":
                    rows = c["predictions"]
                    corr[(f, task)][cond][seed] = {x["row_id"]: int(bool(x["exact_match"])) for x in rows}
                    preds[(f, task)][cond][seed] = {x["row_id"]: x["prediction"] for x in rows}
                    for x in rows:
                        gold[(f, task)][x["row_id"]] = x["gold"]
                else:
                    corr[(f, task)][cond][seed] = {x[0]: int(x[1] == x[2]) for x in c["predictions"]}
                    preds[(f, task)][cond][seed] = {x[0]: x[1] for x in c["predictions"]}
                    for x in c["predictions"]:
                        gold[(f, task)][x[0]] = x[2]
    return common, kinds, corr, preds, gold, meta, expected


def target_ids(fam, task, common, expected):
    if fam == "bbh":
        js = [j for (f, t, s), j in expected.items() if f == fam and t == task]
        return sorted(set(js[0]["eval_ids"]))
    return list(common[task]["common_ids"])


def coverage(common, corr, meta, expected):
    rows, used = [], {}
    tasks = sorted({(f, t) for f, t, s in expected})
    for f, t in tasks:
        exp_seeds = sorted(s for ff, tt, s in expected if (ff, tt) == (f, t))
        got = sorted(s for s in exp_seeds if (f, t, s) in meta)
        target = target_ids(f, t, common, expected)
        ids = set(target)
        for c in CONDS:
            for s in got:
                ids &= set(corr[(f, t)][c].get(s, {}))
        used[(f, t)] = (sorted(ids), got)
        rows.append(dict(family=f, task=t, seeds_expected=len(exp_seeds), seeds_present=len(got),
                         missing_seeds=" ".join(map(str, sorted(set(exp_seeds) - set(got)))),
                         questions_target=len(target), questions_used=len(ids),
                         coverage=len(ids) / len(target) if target else float("nan"),
                         **{f"cov_{c}": float(np.mean([len(set(target) & set(corr[(f, t)][c].get(s, {})))
                                                        / len(target) for s in got])) if got else float("nan")
                            for c in CONDS}))
    return rows, used


def boot_weights(n, rng):
    idx = rng.integers(0, n, size=(B, n))
    return np.bincount((idx + np.arange(B)[:, None] * n).ravel(), minlength=B * n).reshape(B, n).astype(float)


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def family_boot(corr, used, fam, rng_seed=0):
    """Per task [cond, seed, q] matrices and bootstrap draws (shared indices across conditions).
    Returns per-task dicts with point accuracies [cond], per-seed accuracies [cond, seed], draws [cond, B] (seeds x
    questions) and seed-only draws [cond, B]."""
    rng = np.random.default_rng(rng_seed)
    out = []
    for (f, t), (ids, seeds) in sorted(used.items()):
        if f != fam or not ids or not seeds:
            continue
        M = np.array([[[corr[(f, t)][c][s][q] for q in ids] for s in seeds] for c in CONDS], dtype=float)
        C, S, Q = M.shape
        ws, wq = boot_weights(S, rng), boot_weights(Q, rng)
        acc_s = np.einsum("csq,bq->cbs", M, wq) / Q * 100
        draws = (acc_s * ws[None]).sum(-1) / S
        per_seed = M.mean(-1) * 100
        seed_draws = np.einsum("cs,bs->cb", per_seed, ws) / S
        out.append(dict(task=t, seeds=seeds, ids=ids, M=M, point=per_seed.mean(-1), per_seed=per_seed,
                        draws=draws, seed_draws=seed_draws))
    return out


def summarize(fb, fam):
    k = {c: i for i, c in enumerate(CONDS)}
    if not fb:
        return None, [], []
    point = np.mean([x["point"] for x in fb], 0)
    draws = np.mean([x["draws"] for x in fb], 0)
    sdraws = np.mean([x["seed_draws"] for x in fb], 0)
    cond_rows = [dict(family=fam, condition=c, step=STEPS["bbh" if fam == "bbh" else "nlplr"][c], tasks=len(fb),
                      accuracy=point[k[c]], ci_lo=ci(draws[k[c]])[0], ci_hi=ci(draws[k[c]])[1]) for c in CONDS]
    con_rows, seed_rows = [], []
    for a, b, name in CONTRASTS:
        d = draws[k[a]] - draws[k[b]]
        ds = sdraws[k[a]] - sdraws[k[b]]
        fixed = sum(int(((x["M"][k[a]] == 1) & (x["M"][k[b]] == 0)).sum()) for x in fb)
        broken = sum(int(((x["M"][k[a]] == 0) & (x["M"][k[b]] == 1)).sum()) for x in fb)
        sd = [x["per_seed"][k[a]] - x["per_seed"][k[b]] for x in fb]
        allsd = np.concatenate(sd)
        con_rows.append(dict(family=fam, contrast=name, a=a, b=b, task="ALL", seeds=int(sum(len(x["seeds"]) for x in fb)),
                             questions=int(sum(len(x["ids"]) for x in fb)), acc_a=point[k[a]], acc_b=point[k[b]],
                             diff=point[k[a]] - point[k[b]], ci_lo=ci(d)[0], ci_hi=ci(d)[1],
                             seed_ci_lo=ci(ds)[0], seed_ci_hi=ci(ds)[1], fixed=fixed, broken=broken,
                             seeds_better=int((allsd > 1e-9).sum()), seeds_worse=int((allsd < -1e-9).sum()),
                             seeds_tie=int((np.abs(allsd) <= 1e-9).sum())))
        for x, s_d in zip(fb, sd):
            dd = x["draws"][k[a]] - x["draws"][k[b]]
            con_rows.append(dict(family=fam, contrast=name, a=a, b=b, task=x["task"], seeds=len(x["seeds"]),
                                 questions=len(x["ids"]), acc_a=x["point"][k[a]], acc_b=x["point"][k[b]],
                                 diff=x["point"][k[a]] - x["point"][k[b]], ci_lo=ci(dd)[0], ci_hi=ci(dd)[1],
                                 seed_ci_lo=float("nan"), seed_ci_hi=float("nan"),
                                 fixed=int(((x["M"][k[a]] == 1) & (x["M"][k[b]] == 0)).sum()),
                                 broken=int(((x["M"][k[a]] == 0) & (x["M"][k[b]] == 1)).sum()),
                                 seeds_better=int((s_d > 1e-9).sum()), seeds_worse=int((s_d < -1e-9).sum()),
                                 seeds_tie=int((np.abs(s_d) <= 1e-9).sum())))
    for x in fb:
        for i, s in enumerate(x["seeds"]):
            seed_rows.append(dict(family=fam, task=x["task"], seed=s,
                                  **{c: x["per_seed"][k[c]][i] for c in CONDS}))
    return dict(point=point, draws=draws), cond_rows + [], (con_rows, seed_rows)


def bbh_diag(preds, gold, used):
    rows = []
    for (f, t), (ids, seeds) in used.items():
        if f != "bbh":
            continue
        g = gold[(f, t)]
        for c in CONDS:
            per = []
            for s in seeds:
                p = preds[(f, t)][c][s]
                gt = [q for q in ids if str(g[q]).strip() == "True"]
                gf = [q for q in ids if str(g[q]).strip() == "False"]
                per.append(dict(acc=100 * np.mean([str(p[q]).strip() == str(g[q]).strip() for q in ids]),
                                pred_true=np.mean([str(p[q]).strip() == "True" for q in ids]),
                                invalid=np.mean([str(p[q]).strip() not in ("True", "False") for q in ids]),
                                recall_true=np.mean([str(p[q]).strip() == "True" for q in gt]) if gt else np.nan,
                                recall_false=np.mean([str(p[q]).strip() == "False" for q in gf]) if gf else np.nan))
            rows.append(dict(task=t, condition=c, seeds=len(seeds), gold_true_share=np.mean(
                [str(g[q]).strip() == "True" for q in ids]), **{k: float(np.mean([x[k] for x in per])) for k in per[0]}))
    return rows


def cost(meta):
    rows = []
    for fam in FAMS:
        rs = [r for (f, t, s), r in meta.items() if f == fam]
        if not rs:
            continue
        tm = lambda r, k: r["timing"].get(k, np.nan)
        peak = [r["timing"]["peak_mem_mb"] if "peak_mem_mb" in r["timing"]
                else r["timing"].get("peak_allocated_bytes", np.nan) / 2 ** 20 for r in rs]
        row = dict(family=fam, jobs=len(rs), prefill_s=float(np.median([tm(r, "prefill_s") for r in rs])),
                   interp_s=float(np.median([tm(r, "interp_s") for r in rs])),
                   eval_s_total=float(np.median([tm(r, "eval_s_total") for r in rs])),
                   peak_mib=float(np.median(peak)))
        for c in CONDS:
            row[f"adapt_s_{c}"] = float(np.median([r["timing"]["cumulative_adapt_s"][c] for r in rs]))
            row[f"steps_{c}"] = STEPS["bbh" if fam == "bbh" else "nlplr"][c]
        row["adapt_share_early_quarter"] = float(np.median([r["timing"]["cumulative_adapt_s"]["early_quarter"] /
                                                            r["timing"]["cumulative_adapt_s"]["full"] for r in rs]))
        row["adapt_share_early_half"] = float(np.median([r["timing"]["cumulative_adapt_s"]["early_half"] /
                                                         r["timing"]["cumulative_adapt_s"]["full"] for r in rs]))
        rows.append(row)
    return rows


def geometry(meta):
    rows = []
    for fam in FAMS:
        rs = [r for (f, t, s), r in meta.items() if f == fam]
        for c in ("early_quarter", "early_half", "half_delta"):
            for kv in ("K", "V"):
                g = [r["geometry"][c][kv] for r in rs if c in r.get("geometry", {})]
                if g:
                    rows.append(dict(family=fam, condition=c, tensor=kv, jobs=len(g),
                                     **{k: float(np.median([x[k] for x in g])) for k in
                                        ("norm_ratio", "cosine", "rel_to_C0", "norm_t", "norm_T")}))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main(run_dir, stage1):
    common, kinds, corr, preds, gold, meta, expected = load(run_dir, stage1)
    cov, used = coverage(common, corr, meta, expected)
    write_csv(os.path.join(run_dir, "coverage.csv"), cov)
    cond_rows, con_rows, seed_rows, summ = [], [], [], {}
    for i, fam in enumerate(FAMS):
        fb = family_boot(corr, used, fam, rng_seed=i)
        s, cr, extra = summarize(fb, fam)
        if s is None:
            continue
        summ[fam] = s
        cond_rows += cr
        con_rows += extra[0]
        seed_rows += extra[1]
        # secondary: runner score (macro-F1 for clf), seed mean then task mean
        if fam != "bbh":
            for row in cr:
                c = row["condition"]
                row["runner_score"] = float(np.mean([np.mean([meta[(fam, x["task"], sd)]["conditions"][c]["score"]
                                                              for sd in x["seeds"]]) for x in fb])) * 100
    write_csv(os.path.join(run_dir, "conditions.csv"), cond_rows)
    write_csv(os.path.join(run_dir, "contrasts.csv"), con_rows)
    write_csv(os.path.join(run_dir, "per_seed.csv"), seed_rows)
    diag = bbh_diag(preds, gold, used)
    write_csv(os.path.join(run_dir, "bbh_diagnostics.csv"), diag)
    cst = cost(meta)
    write_csv(os.path.join(run_dir, "cost.csv"), cst)
    geo = geometry(meta)
    write_csv(os.path.join(run_dir, "geometry.csv"), geo)
    with gzip.open(os.path.join(run_dir, "predictions.jsonl.gz"), "wt") as f:
        for (fam, t, s), r in sorted(meta.items()):
            f.write(json.dumps(dict(family=fam, task=t, seed=s, id=r["id"],
                                    conditions={c: {str(q): [p, corr[(fam, t)][c][s][q]]
                                                    for q, p in preds[(fam, t)][c][s].items()} for c in CONDS},
                                    gold={str(q): g for q, g in gold[(fam, t)].items()})) + "\n")
    checks_bad = sorted(r["id"] for r in meta.values() if any(v is False for v in r["checks"].values()))
    report(run_dir, cov, cond_rows, con_rows, diag, cst, geo, checks_bad, len(meta), len(expected))


def report(run_dir, cov, cond_rows, con_rows, diag, cst, geo, checks_bad, n_done, n_exp):
    L = ["# CT-KV Stage 3 round 1: half-delta KV update and same-trajectory early stopping — results", "",
         f"Trajectories: {n_done} / {n_exp} expected. Runner technical checks failing: "
         f"{', '.join(checks_bad) if checks_bad else 'none'}.", "",
         "Accuracy in points on the per-task common questions (present in all five conditions of every seed); seed "
         "mean per task, tasks equal weight. Brackets: 95% paired bootstrap (5,000 draws over seeds × questions, shared "
         "across conditions); seed-only interval in parentheses.", "", "## Five-condition accuracy", "",
         "| family | tasks | " + " | ".join(CONDS) + " |", "|---|---|" + "---|" * len(CONDS)]
    for fam in FAMS:
        r = [x for x in cond_rows if x["family"] == fam]
        if r:
            L.append(f"| {FAM_NAME[fam]} | {r[0]['tasks']} | " + " | ".join(
                f"{x['accuracy']:.2f} [{x['ci_lo']:.2f}, {x['ci_hi']:.2f}]" for x in r) + " |")
    for fam in ("clf", "mc"):
        r = [x for x in cond_rows if x["family"] == fam and "runner_score" in x]
        if r:
            L.append(f"| {fam} runner score ({'macro-F1' if fam == 'clf' else 'accuracy'}) | {r[0]['tasks']} | "
                     + " | ".join(f"{x['runner_score']:.2f}" for x in r) + " |")
    L += ["", "Gradient steps: BBH icl 0 / early_quarter 4 / early_half 8 / full 16 / half_delta 16; NLP-LR 0 / 50 / "
          "100 / 200 / 200.", "", "## Contrasts", "",
          "| contrast | family | a | b | diff | 95% CI | seed-only CI | fixed / broken Q | seeds better / worse / tie |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a, b, name in CONTRASTS:
        for fam in FAMS:
            r = [x for x in con_rows if x["family"] == fam and x["contrast"] == name and x["task"] == "ALL"]
            if r:
                x = r[0]
                L.append(f"| {name} | {fam} | {x['acc_a']:.2f} | {x['acc_b']:.2f} | {x['diff']:+.2f} | "
                         f"[{x['ci_lo']:+.2f}, {x['ci_hi']:+.2f}] | ({x['seed_ci_lo']:+.2f}, {x['seed_ci_hi']:+.2f}) | "
                         f"{x['fixed']} / {x['broken']} | {x['seeds_better']} / {x['seeds_worse']} / {x['seeds_tie']} |")
    L += ["", "## Pre-registered decisions", ""]
    main_ = {x["family"]: x for x in con_rows if x["contrast"].startswith("main") and x["task"] == "ALL"}
    if "bbh" in main_:
        x = main_["bbh"]
        L.append(f"- BBH half_delta − full = {x['diff']:+.2f} [{x['ci_lo']:+.2f}, {x['ci_hi']:+.2f}]: "
                 + ("**improved** (lower bound > 0)." if x["ci_lo"] > 0 else "not shown to improve (lower bound ≤ 0)."))
    for fam in ("clf", "mc"):
        if fam in main_:
            x = main_[fam]
            v = ("**gain kept** (lower bound > −1.0)" if x["ci_lo"] > -1 else
                 "**loss beyond 1 point**" if x["ci_hi"] < -1 else "uncertain (interval crosses −1.0)")
            L.append(f"- {fam} half_delta − full = {x['diff']:+.2f} [{x['ci_lo']:+.2f}, {x['ci_hi']:+.2f}]: {v}.")
    L += ["", "## Per task: main contrast (half_delta − full) and full − icl", "",
          "| family | task | seeds | Q | icl | full | half_delta | half_delta − full | full − icl | fixed / broken |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    per = {(x["family"], x["task"], x["contrast"]): x for x in con_rows if x["task"] != "ALL"}
    for (fam, task, name), x in sorted(per.items()):
        if not name.startswith("main"):
            continue
        y = per[(fam, task, "full − icl")]
        L.append(f"| {fam} | {task} | {x['seeds']} | {x['questions']} | {y['acc_b']:.2f} | {x['acc_b']:.2f} | "
                 f"{x['acc_a']:.2f} | {x['diff']:+.2f} [{x['ci_lo']:+.2f}, {x['ci_hi']:+.2f}] | {y['diff']:+.2f} | "
                 f"{x['fixed']} / {x['broken']} |")
    if diag:
        L += ["", "## BBH label diagnostics (mean over seeds)", "",
              "| condition | acc | pred True | recall True | recall False | invalid |", "|---|---|---|---|---|---|"]
        for d in diag:
            L.append(f"| {d['condition']} | {d['acc']:.2f} | {d['pred_true']:.3f} | {d['recall_true']:.3f} | "
                     f"{d['recall_false']:.3f} | {d['invalid']:.3f} |")
        L.append(f"\nGold True share: {diag[0]['gold_true_share']:.3f}.")
    L += ["", "## Cost (medians over trajectories; queue ran serially, GPU not shared)", "",
          "| family | jobs | prefill s | adapt s → early_quarter | → early_half | → full | interp s | eval s (5 conds) | "
          "peak MiB | early_quarter share | early_half share |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cst:
        L.append(f"| {c['family']} | {c['jobs']} | {c['prefill_s']:.3f} | {c['adapt_s_early_quarter']:.2f} | "
                 f"{c['adapt_s_early_half']:.2f} | {c['adapt_s_full']:.2f} | {c['interp_s']:.4f} | "
                 f"{c['eval_s_total']:.2f} | {c['peak_mib']:.0f} | {c['adapt_share_early_quarter']:.3f} | "
                 f"{c['adapt_share_early_half']:.3f} |")
    L += ["", "## Displacement geometry (medians; C_t − C_0 against C_T − C_0)", "",
          "| family | condition | tensor | norm ratio | cosine | ‖C_T − C_0‖ / ‖C_0‖ |", "|---|---|---|---|---|---|"]
    for g in geo:
        L.append(f"| {g['family']} | {g['condition']} | {g['tensor']} | {g['norm_ratio']:.3f} | {g['cosine']:.3f} | "
                 f"{g['rel_to_C0']:.4f} |")
    L += ["", "## Coverage", "", "| family | task | seeds | Q used / target |", "|---|---|---|---|"]
    for c in cov:
        L.append(f"| {c['family']} | {c['task']} | {c['seeds_present']}/{c['seeds_expected']}"
                 f"{' (missing ' + c['missing_seeds'] + ')' if c['missing_seeds'] else ''} | "
                 f"{c['questions_used']} / {c['questions_target']} |")
    open(os.path.join(run_dir, "REPORT_results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
