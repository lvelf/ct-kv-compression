"""Write results/stress/full_jobs.txt: every remaining stress-test run, in priority order.

Run from the repo root. gpu_queue.py skips any job whose JSON exists, so this can be regenerated freely.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PRE = ("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/accelerate launch --num_processes 1 "
       "--mixed_precision bf16 train.py --experiment_name stress --batch_size 16 --diag_every 25")
jobs = []


def add(out, args):
    out = f"../results/stress/{out}.json"
    jobs.append(f"{out} {PRE} {args} --output_json {out}")


def comp(method, r, e, s):
    if method == "none":
        add(f"b3_compress/none_r1.0_e{e}_s{s}", f"--epochs {e} --eval_split {s}")
    else:
        add(f"b3_compress/{method}_r{r}_e{e}_s{s}", f"--epochs {e} --eval_split {s} --kv_compress {method} --kv_keep_ratio {r}")


CORE = ["random", "random_sink", "attn", "l2"]
RATIOS = [0.75, 0.5, 0.25, 0.125]

# P1: split 87, budget grid with CT-KV (the teammates' plan + sink control)
for m in CORE:
    for r in RATIOS:
        comp(m, r, 200, 87)
comp("random_segstart", 0.25, 200, 87)
# P2: tune only V or only K (follow-up to A5)
for which in ("v", "k"):
    add(f"a5_trainkv/ctkv_{which}_e200_s87", f"--epochs 200 --eval_split 87 --train_kv {which}")
# P3: split 87, same grid without training, plus the uncompressed ICL reference with memory/time
comp("none", 1.0, 0, 87)
for m in CORE + ["random_segstart"]:
    for r in RATIOS if m != "random_segstart" else [0.25]:
        comp(m, r, 0, 87)
# P4: second and third split for the core grid
for s in (13, 21):
    comp("none", 1.0, 0, s)
    for m in CORE:
        for r in RATIOS:
            comp(m, r, 200, s)
    for m in CORE:
        for r in RATIOS:
            comp(m, r, 0, s)
    for which in ("v", "k"):
        add(f"a5_trainkv/ctkv_{which}_e200_s{s}", f"--epochs 200 --eval_split {s} --train_kv {which}")
# P5: remaining epoch sweep (A3) and noise floor (A4), then the old B3 list (other controls on split 13 etc.)
for line in open(os.path.join(ROOT, "a3_epochs/jobs.txt")):
    if line.strip():
        jobs.append(line.strip())
b3 = [l.strip() for l in open(os.path.join(ROOT, "b3_compress/jobs.txt")) if l.strip()]
jobs += [l for l in b3 if "a4_noise" in l] + [l for l in b3 if "a4_noise" not in l]

seen, out = set(), []
for j in jobs:
    k = j.split(None, 1)[0]
    if k not in seen:
        seen.add(k)
        out.append(j)
open(os.path.join(ROOT, "full_jobs.txt"), "w").write("\n".join(out) + "\n")
todo = [j for j in out if not os.path.exists(os.path.join(ROOT, "..", "..", "context-tuning", j.split()[0]))]
print(f"{len(out)} jobs, {len(todo)} not done yet")
