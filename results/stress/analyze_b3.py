"""Analyze stress test B3: compress the initial demonstration KV cache, with and without CT-KV.

Reads results/stress/b3_compress/{method}_r{ratio}_e{epochs}_s{split}.json. Baselines are the uncompressed
ICL (nlp_lr_repro logs) and CT-KV (nlp_lr_repro logs, epochs 200) on the same splits.
Writes results/stress/b3_compress/report.md.
Run from the repo root: context-tuning/.venv/bin/python results/stress/analyze_b3.py
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "b3_compress")
LOGS = os.path.join(ROOT, "..", "nlp_lr_repro", "logs")


def log_scores(name, s):
    txt = open(os.path.join(LOGS, f"{name}_{s}.log")).read()
    return {t: float(v) * 100 for t, v in re.findall(r"^(\S+) clf \w+ has a score ([\d.]+)", txt, re.M)}


runs = defaultdict(dict)  # (method, ratio, epochs) -> split -> result
for path in glob.glob(os.path.join(D, "*.json")):
    m = re.match(r"(\w+?)_r([\d.]+)_e(\d+)_s(\d+)\.json", os.path.basename(path))
    method, ratio, e, s = m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4))
    runs[(method, ratio, e)][s] = json.load(open(path))

splits = sorted({s for v in runs.values() for s in v})
icl = {s: log_scores("icl", s) for s in splits}
ctkv = {s: log_scores("ctkv", s) for s in splits}
avg = lambda d: float(np.mean(list(d.values())))
icl_avg = np.mean([avg(icl[s]) for s in splits]) if splits else float("nan")
ctkv_avg = np.mean([avg(ctkv[s]) for s in splits]) if splits else float("nan")

out = []
p = out.append
p("# Stress test B3: compressing the CT-KV initialization (NLP-LR, GPT-2 Large)\n")
p(f"Splits: {splits}. Uncompressed baselines on these splits: ICL **{icl_avg:.2f}**, CT-KV (200 epochs) **{ctkv_avg:.2f}**.\n")
p("Each demonstration keeps ceil(ratio × its length) tokens. `e0` means compression then ICL (no training), and `e200` means compression then CT-KV.")
p("\"Gain kept\" is (score − ICL) / (uncompressed CT-KV − ICL).\n")
p("| method | keep ratio | kv len | e0 (compressed ICL) | e200 (compressed CT-KV) | gain kept | train s/task | splits |")
p("|---|---|---|---|---|---|---|---|")
keys = sorted({(m, r) for (m, r, e) in runs}, key=lambda x: (x[0], -x[1]))
for method, ratio in keys:
    cells = {}
    common = None
    for e in (0, 200):
        rs = runs.get((method, ratio, e), {})
        if rs:
            cells[e] = np.mean([r["average_score"] * 100 for r in rs.values()])
            common = rs
    if common is None:
        continue
    kvlen = np.mean([d["kv_len"] for r in common.values() for d in r["task_to_diag"].values()])
    r200 = runs.get((method, ratio, 200), {})
    tt = np.mean([d["train_time_s"] for r in r200.values() for d in r["task_to_diag"].values()]) if r200 else float("nan")
    gain = (cells[200] - icl_avg) / (ctkv_avg - icl_avg) * 100 if 200 in cells else float("nan")
    ns = "/".join(str(len(runs.get((method, ratio, e), {}))) for e in (0, 200))
    p(f"| {method} | {ratio} | {kvlen:.0f} | {cells.get(0, float('nan')):.2f} | {cells.get(200, float('nan')):.2f} | {gain:.0f}% | {tt:.1f} | {ns} |")

open(os.path.join(D, "report.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
