"""Analyze stress test B3: compress the initial demonstration KV cache, with and without CT-KV.

Reads results/stress/b3_compress/{method}_r{ratio}_e{epochs}_s{split}.json. Baselines are the uncompressed
ICL (nlp_lr_repro logs) and CT-KV (nlp_lr_repro logs, epochs 200) on the same splits; the uncompressed
memory/time reference comes from none_r1.0_e0 (ICL) and a3_epochs/ctkv_e200 (CT-KV) JSONs.
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
# GPT-2 Large: 36 layers, K and V, 1280 dims, bf16
KV_BYTES_PER_TOKEN = 36 * 2 * 1280 * 2


def log_scores(name, s):
    txt = open(os.path.join(LOGS, f"{name}_{s}.log")).read()
    return {t: float(v) * 100 for t, v in re.findall(r"^(\S+) clf \w+ has a score ([\d.]+)", txt, re.M)}


runs = defaultdict(dict)  # (method, ratio, epochs) -> split -> result
for path in glob.glob(os.path.join(D, "*.json")):
    m = re.match(r"(\w+?)_r([\d.]+)_e(\d+)_s(\d+)\.json", os.path.basename(path))
    method, ratio, e, s = m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4))
    runs[(method, ratio, e)][s] = json.load(open(path))
# uncompressed CT-KV reference with diagnostics, from the epoch sweep
for path in glob.glob(os.path.join(ROOT, "a3_epochs", "ctkv_e200_s*.json")):
    s = int(re.findall(r"_s(\d+)\.json", path)[0])
    runs[("none", 1.0, 200)][s] = json.load(open(path))

all_splits = sorted({s for v in runs.values() for s in v})
icl = {s: log_scores("icl", s) for s in all_splits}
ctkv = {s: log_scores("ctkv", s) for s in all_splits}
avg = lambda d: float(np.mean(list(d.values())))
mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")


def diag_mean(rs, key):
    return mean([d[key] for r in rs.values() for d in r["task_to_diag"].values() if key in d])


out = []
p = out.append
p("# Stress test B3: compressing the CT-KV initialization (NLP-LR, GPT-2 Large)\n")
p("Each demonstration keeps ceil(ratio × its length) tokens; the same tokens in every layer and head. "
  "`e0` = compress, then ICL (no training). `e200` = compress, then CT-KV for 200 epochs.\n")
p("Scores are averaged over the splits listed in the last column (e0/e200). \"Gain kept\" is "
  "(e200 − ICL) / (uncompressed CT-KV − ICL), with both baselines averaged over the same splits as that row's e200 runs.\n")
p("Memory: `KV MB` is the size of the demonstration cache (bf16, all layers). `train peak` and `eval peak` are "
  "torch peak allocated memory per task, including the 1.5 GB model. Times are per task, averaged over tasks.\n")
p("| method | ratio | kv len | KV MB | e0 | e200 | gain kept | sink kept | train s | train peak MB | eval s | eval peak MB | splits e0/e200 |")
p("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
keys = sorted({(m, r) for (m, r, e) in runs}, key=lambda x: (x[0] != "none", x[0], -x[1]))
for method, ratio in keys:
    r0, r200 = runs.get((method, ratio, 0), {}), runs.get((method, ratio, 200), {})
    common = r200 or r0
    e0 = mean([r["average_score"] * 100 for r in r0.values()])
    e200 = mean([r["average_score"] * 100 for r in r200.values()])
    if r200:
        sp = list(r200)
        icl_a, ct_a = mean([avg(icl[s]) for s in sp]), mean([avg(ctkv[s]) for s in sp])
        gain = f"{(e200 - icl_a) / (ct_a - icl_a) * 100:.0f}%"
    else:
        gain = "–"
    kvlen = diag_mean(common, "kv_len")
    sink = diag_mean(common, "kept_token0")
    sink = f"{sink * 100:.0f}%" if not np.isnan(sink) else ("100%" if method == "none" else "–")
    f = lambda x, fmt: "–" if np.isnan(x) else format(x, fmt)
    p(f"| {method} | {ratio} | {kvlen:.0f} | {kvlen * KV_BYTES_PER_TOKEN / 2**20:.1f} | {f(e0, '.2f')} | {f(e200, '.2f')} | {gain} | {sink} "
      f"| {f(diag_mean(r200, 'train_time_s'), '.1f')} | {f(diag_mean(r200, 'train_peak_mem_mb'), '.0f')} "
      f"| {f(diag_mean(common, 'eval_time_s'), '.1f')} | {f(diag_mean(common, 'eval_peak_mem_mb'), '.0f')} | {len(r0)}/{len(r200)} |")

# budget pivot, only over splits where every core method has e200 at every ratio
CORE = ["random", "random_sink", "attn", "l2"]
RATIOS = [0.75, 0.5, 0.25, 0.125]
for e in (200, 0):
    sp = [s for s in all_splits if all(s in runs.get((m, r, e), {}) for m in CORE for r in RATIOS)]
    p(f"\n## Budget grid, epochs {e} (splits {sp if sp else 'none complete yet'})\n")
    if not sp:
        continue
    ref = mean([avg(ctkv[s]) for s in sp]) if e == 200 else mean([avg(icl[s]) for s in sp])
    p("| keep ratio | " + " | ".join(CORE) + " |")
    p("|---|" + "---|" * len(CORE))
    p(f"| 1.0 (uncompressed) | " + " | ".join(f"{ref:.2f}" for _ in CORE) + " |")
    for r in RATIOS:
        p(f"| {r} | " + " | ".join(f"{mean([runs[(m, r, e)][s]['average_score'] * 100 for s in sp]):.2f}" for m in CORE) + " |")
    p(f"\nICL (uncompressed, no training) on these splits: {mean([avg(icl[s]) for s in sp]):.2f}")

open(os.path.join(D, "report.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
