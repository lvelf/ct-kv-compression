"""Analyze the CT-KV epoch sweep (stress tests A3, A5, B1).

Reads results/stress/a3_epochs/ctkv_e{E}_s{S}.json, plus the ICL logs in results/nlp_lr_repro/logs
as the epoch-0 point. Writes results/stress/a3_epochs/report.md.
Run from the repo root: context-tuning/.venv/bin/python results/stress/analyze_a3.py
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "a3_epochs")
REPRO_LOGS = os.path.join(ROOT, "..", "nlp_lr_repro", "logs")
SPLITS = [13, 21, 42, 87, 100]

# scores[epochs][split] = {task: score in %}
scores = defaultdict(dict)
diags = defaultdict(dict)
for path in glob.glob(os.path.join(D, "ctkv_e*_s*.json")):
    e, s = map(int, re.findall(r"ctkv_e(\d+)_s(\d+)\.json", os.path.basename(path))[0])
    r = json.load(open(path))
    scores[e][s] = {t: v * 100 for t, v in r["task_to_score"].items()}
    diags[e][s] = r["task_to_diag"]
for s in SPLITS:
    log = os.path.join(REPRO_LOGS, f"icl_{s}.log")
    if os.path.exists(log):
        txt = open(log).read()
        scores[0][s] = {t: float(v) * 100 for t, v in re.findall(r"^(\S+) clf \w+ has a score ([\d.]+)", txt, re.M)}

epochs_list = sorted(scores)
out = []
p = out.append


def avg(d):
    return float(np.mean(list(d.values())))


p("# Stress test A3/B1/A5: CT-KV vs number of epochs (NLP-LR, GPT-2 Large)\n")
p("Epoch 0 is ICL (from the reproduction logs). Every run uses lr 1e-3, token dropout 0.05, and a cosine schedule over its own number of epochs.\n")

# ---- A3: average score vs epochs
p("## A3. Average score vs epochs\n")
p("| epochs | " + " | ".join(f"s{s}" for s in SPLITS) + " | mean ± std | splits done |")
p("|" + "---|" * (len(SPLITS) + 3))
for e in epochs_list:
    row = [f"{avg(scores[e][s]):.2f}" if s in scores[e] else "–" for s in SPLITS]
    vals = [avg(scores[e][s]) for s in SPLITS if s in scores[e]]
    ms = f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}" if len(vals) > 1 else (f"{vals[0]:.2f}" if vals else "–")
    p(f"| {e} | " + " | ".join(row) + f" | {ms} | {len(vals)} |")

# ---- B1: per-task oracle epochs (only splits where every epoch value is done)
complete = [s for s in SPLITS if all(s in scores[e] for e in epochs_list)]
p("\n## B1. Per-task oracle number of epochs\n")
if complete:
    tasks = sorted(scores[epochs_list[0]][complete[0]])
    fixed = {e: np.mean([avg(scores[e][s]) for s in complete]) for e in epochs_list}
    best_fixed_e = max(fixed, key=fixed.get)
    # oracle per (task, split): pick best epochs using the test score (upper bound, not a usable method)
    oracle_split = []
    # oracle per task (same epochs for every split of a task): still optimistic but less noisy
    for s in complete:
        oracle_split.append(np.mean([max(scores[e][s][t] for e in epochs_list) for t in tasks]))
    task_best_e = {}
    for t in tasks:
        per_e = {e: np.mean([scores[e][s][t] for s in complete]) for e in epochs_list}
        task_best_e[t] = max(per_e, key=per_e.get)
    oracle_task = np.mean([np.mean([scores[task_best_e[t]][s][t] for t in tasks]) for s in complete])
    p(f"Computed over splits {complete}.\n")
    p(f"- Best single epochs for all tasks: **{best_fixed_e}**, score {fixed[best_fixed_e]:.2f} (epochs=200: {fixed.get(200, float('nan')):.2f})")
    p(f"- Oracle epochs per task (shared across splits): **{oracle_task:.2f}**")
    p(f"- Oracle epochs per task and split (upper bound): **{np.mean(oracle_split):.2f}**\n")
    p("| task | " + " | ".join(f"e{e}" for e in epochs_list) + " | best e | gain best vs e200 |")
    p("|" + "---|" * (len(epochs_list) + 3))
    for t in tasks:
        per_e = [np.mean([scores[e][s][t] for s in complete]) for e in epochs_list]
        b = task_best_e[t]
        e200 = np.mean([scores[200][s][t] for s in complete]) if 200 in scores else float("nan")
        p(f"| {t} | " + " | ".join(f"{v:.1f}" for v in per_e) + f" | {b} | {max(per_e) - e200:+.1f} |")
else:
    p("Waiting for a split where every epoch value has finished.\n")

# ---- A5: K vs V gradients and drift, from the epochs=200 runs
p("\n## A5. Where does CT-KV change the cache? (epochs=200 runs)\n")
if 200 in diags and diags[200]:
    gk0, gv0, gk, gv, dk, dv = [], [], [], [], [], []
    for s, td in diags[200].items():
        for t, d in td.items():
            gk0.append(d["grad_sq_k_epoch0"]); gv0.append(d["grad_sq_v_epoch0"])
            gk.append(d["grad_sq_k_mean"]); gv.append(d["grad_sq_v_mean"])
            dk.append(d["drift"]["200"]["k"]); dv.append(d["drift"]["200"]["v"])
    gk0, gv0, gk, gv, dk, dv = map(np.array, (gk0, gv0, gk, gv, dk, dv))
    p(f"Over {len(gk0)} (task, split) runs.\n")
    p(f"- Mean squared gradient at epoch 0 (diagonal-Fisher proxy): K {gk0.mean():.2e}, V {gv0.mean():.2e}, V/K = {gv0.mean() / gk0.mean():.1f}")
    p(f"- Averaged over training: K {gk.mean():.2e}, V {gv.mean():.2e}, V/K = {gv.mean() / gk.mean():.1f}")
    p(f"- Relative drift ||X-X0||/||X0|| after 200 epochs: K {dk.mean():.4f}, V {dv.mean():.4f}\n")
    p("Per layer (mean over runs):\n")
    p("| layer | grad² K (e0) | grad² V (e0) | V/K | drift K | drift V |")
    p("|---|---|---|---|---|---|")
    for l in range(gk0.shape[1]):
        p(f"| {l} | {gk0[:, l].mean():.2e} | {gv0[:, l].mean():.2e} | {gv0[:, l].mean() / gk0[:, l].mean():.1f} | {dk[:, l].mean():.4f} | {dv[:, l].mean():.4f} |")

    # cost vs prefix length
    pl, tt, mem = [], [], []
    for s, td in diags[200].items():
        for t, d in td.items():
            pl.append(d["prefix_len"]); tt.append(d["train_time_s"]); mem.append(d["train_peak_mem_mb"])
    pl, tt, mem = map(np.array, (pl, tt, mem))
    slope, icpt = np.polyfit(pl, tt, 1)
    p(f"\n- Training time for 200 epochs: mean {tt.mean():.1f}s per task, fit {icpt:.2f}s + {slope * 100:.2f}s per 100 prefix tokens (corr {np.corrcoef(pl, tt)[0, 1]:.2f})")
    p(f"- Training peak memory: mean {mem.mean():.0f} MB, max {mem.max():.0f} MB; prefix length {pl.min()}–{pl.max()} tokens (mean {pl.mean():.0f})")

    # training loss vs test gain over ICL
    p("\n- Final training loss vs gain over ICL, per (task, split):")
    fl, gain = [], []
    for s, td in diags[200].items():
        if s not in scores[0]:
            continue
        for t, d in td.items():
            fl.append(d["epoch_losses"][-1]); gain.append(scores[200][s][t] - scores[0][s][t])
    if fl:
        p(f"  corr(final train loss, gain) = {np.corrcoef(fl, gain)[0, 1]:.2f}; mean final loss {np.mean(fl):.3f}")
else:
    p("Waiting for epochs=200 runs.\n")

open(os.path.join(D, "report.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
