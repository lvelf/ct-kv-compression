"""Analyze stress tests A4 (run-to-run noise of CT-KV) and A5b (tune only K or only V).

A4 reads a4_noise/ctkv_e200_s87_{rep1,rep2,seed1,seed2}.json plus a3_epochs/ctkv_e200_s87.json.
A5b reads a5_trainkv/ctkv_{k,v}_e200_s{split}.json and compares with a3_epochs/ctkv_e200_s{split}.json.
Writes results/stress/extra_report.md.
Run from the repo root: context-tuning/.venv/bin/python results/stress/analyze_extra.py
"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "..", "nlp_lr_repro", "logs")
load = lambda p: json.load(open(p))
out = []
p = out.append


def log_scores(name, s):
    txt = open(os.path.join(LOGS, f"{name}_{s}.log")).read()
    return {t: float(v) * 100 for t, v in re.findall(r"^(\S+) clf \w+ has a score ([\d.]+)", txt, re.M)}


# ---- A4: noise floor on split 87
p("# Stress tests A4 (noise) and A5b (tune only K or only V)\n")
p("## A4. Run-to-run noise of CT-KV (split 87, 200 epochs)\n")
runs = {"a3 (seed 0)": os.path.join(ROOT, "a3_epochs", "ctkv_e200_s87.json")}
for name in ("rep1", "rep2", "seed1", "seed2"):
    runs[name] = os.path.join(ROOT, "a4_noise", f"ctkv_e200_s87_{name}.json")
runs = {k: v for k, v in runs.items() if os.path.exists(v)}
scores = {k: {t: s * 100 for t, s in load(v)["task_to_score"].items()} for k, v in runs.items()}
scores["reproduction log (seed 0)"] = log_scores("ctkv", 87)
p("| run | average |")
p("|---|---|")
for k, v in scores.items():
    p(f"| {k} | {np.mean(list(v.values())):.2f} |")
same_seed = [k for k in scores if "seed 0" in k or k.startswith("rep")]
other = [k for k in scores if k.startswith("seed")]
for label, keys in (("same seed 0 (GPU nondeterminism only)", same_seed), ("all runs incl. seeds 1, 2", list(scores))):
    if len(keys) < 2:
        continue
    avgs = [np.mean(list(scores[k].values())) for k in keys]
    tasks = sorted(scores[keys[0]])
    task_sd = {t: np.std([scores[k][t] for k in keys], ddof=1) for t in tasks}
    worst = max(task_sd, key=task_sd.get)
    p(f"\n- {label}, {len(keys)} runs: average {np.mean(avgs):.2f} ± {np.std(avgs, ddof=1):.2f} (range {min(avgs):.2f}–{max(avgs):.2f}); "
      f"per-task std mean {np.mean(list(task_sd.values())):.2f}, max {task_sd[worst]:.2f} ({worst})")

# ---- A5b: tune only K or only V
p("\n## A5b. Tune only K or only V (no compression, 200 epochs)\n")
p("| split | K+V (paper) | only V | only K | ICL |")
p("|---|---|---|---|---|")
rows = []
for s in (13, 21, 42, 87, 100):
    f = {w: os.path.join(ROOT, "a5_trainkv", f"ctkv_{w}_e200_s{s}.json") for w in ("k", "v")}
    full = os.path.join(ROOT, "a3_epochs", f"ctkv_e200_s{s}.json")
    if not (all(os.path.exists(x) for x in f.values()) and os.path.exists(full)):
        continue
    r = [load(full), load(f["v"]), load(f["k"])]
    icl = np.mean(list(log_scores("icl", s).values()))
    rows.append([x["average_score"] * 100 for x in r] + [icl])
    p(f"| {s} | " + " | ".join(f"{v:.2f}" for v in rows[-1]) + " |")
    mem = [np.mean([d["train_peak_mem_mb"] for d in x["task_to_diag"].values()]) for x in r]
    tt = [np.mean([d["train_time_s"] for d in x["task_to_diag"].values()]) for x in r]
if rows:
    m = np.mean(rows, axis=0)
    p(f"| mean | " + " | ".join(f"{v:.2f}" for v in m) + " |")
    p(f"\nTraining peak memory (last split): K+V {mem[0]:.0f} MB, only V {mem[1]:.0f} MB, only K {mem[2]:.0f} MB; "
      f"time per task {tt[0]:.1f} / {tt[1]:.1f} / {tt[2]:.1f} s.")

open(os.path.join(ROOT, "extra_report.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
