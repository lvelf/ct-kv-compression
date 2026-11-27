"""Build Stage 2 job lists from the frozen Stage 1 manifests (results/ctkv_refinement).

Every Stage 2 job reuses a Stage 1 job verbatim (demonstration IDs and order, prompt, optimizer seed 0, lr, dropout,
schedule, scoring) and adds only `adapt`, `reads`, `checkpoints` and `mask_seed`. Per (task, seed):

  arm D  dense   adapt dense (diagnostic path, r = 1) ; reads @0 and @final: prod, dense, topk@.75, topk@.5,
                 fixed@.75, random@.75   -> Stage 1 baseline, dense-adapt x sparse-read cells, step-0 (ICL) reads
         topk75  adapt topk r=.75        ; reads @final: dense, topk@.75            (main experiment)
         topk50  adapt topk r=.5         ; reads @final: dense, topk@.5             (budget sensitivity)
         fixed75 adapt fixed (answer_attention) r=.75 ; reads @final: dense, fixed@.75
         rand75  adapt random r=.75      ; reads @final: dense, random@.75 (mask seed 0)
         prune75/prune50 (BBH only)      physical pruning with Ryan's answer_attention (storage-cost check)
  arm A  dense   adapt dense ; reads @0/@final: prod, dense, topk@.75
         topk75  adapt topk r=.75 ; reads @final: dense, topk@.75                    (Stage 1 x Stage 2 synergy)
Step-0 reads are taken only in the dense job: before any update every adaptation mode has the identical cache.

Usage (from context-tuning/): python stage2_jobs.py --stage1 ../results/ctkv_refinement --out ../results/ctkv_stage2
  writes OUT/jobs_nlplr_dev.jsonl, jobs_nlplr.jsonl, jobs_bbh_dev.jsonl, jobs_bbh.jsonl, jobs_dev_randseed_*.jsonl
  and OUT/bench/jobs_bench_{nlplr,bbh}.jsonl: the serial cost benchmark (run alone on the GPU after the accuracy
  panel): arm D exploration cells, one job per adaptation with only its own matching read at the endpoint, so that
  adaptation time, peak memory and eval throughput are measured per mode (prod = production SDPA throughout).
"""
import argparse
import json
from pathlib import Path

DEV_NLPLR = ("financial_phrasebank", "ai2_arc")
DEV_SEEDS = (1000, 1001, 1002)

R = lambda m, r: {"name": f"{m}@{r}" if m not in ("prod", "dense") else m, "mode": m, "r": r}
D_DENSE_READS = [R("prod", 1.0), R("dense", 1.0), R("topk", 0.75), R("topk", 0.5), R("fixed", 0.75),
                 R("random", 0.75)]
A_DENSE_READS = [R("prod", 1.0), R("dense", 1.0), R("topk", 0.75)]
VARIANTS = {
    "D": [("dense", "dense", 1.0, D_DENSE_READS, True),
          ("topk75", "topk", 0.75, [R("dense", 1.0), R("topk", 0.75)], False),
          ("topk50", "topk", 0.5, [R("dense", 1.0), R("topk", 0.5)], False),
          ("fixed75", "fixed", 0.75, [R("dense", 1.0), R("fixed", 0.75)], False),
          ("rand75", "random", 0.75, [R("dense", 1.0), R("random", 0.75)], False)],
    "A": [("dense", "dense", 1.0, A_DENSE_READS, True),
          ("topk75", "topk", 0.75, [R("dense", 1.0), R("topk", 0.75)], False)],
}
BENCH = [("prod", "prod", 1.0, [R("prod", 1.0)], False), ("dense", "dense", 1.0, [R("dense", 1.0)], False),
         ("topk75", "topk", 0.75, [R("topk", 0.75)], False), ("topk50", "topk", 0.5, [R("topk", 0.5)], False),
         ("fixed75", "fixed", 0.75, [R("fixed", 0.75)], False), ("rand75", "random", 0.75, [R("random", 0.75)], False)]
BBH_PRUNE = [("prune75", "prune", 0.75, [R("prod", 1.0)], False), ("prune50", "prune", 0.5, [R("prod", 1.0)], False)]


def expand(base, fam, out_dir, final, variants, mask_seed=0, tag=""):
    arm, seed, task = base["meta"]["arm"], base["meta"]["seed"], base["task"]
    jobs = []
    for name, mode, r, reads, with0 in variants:
        j = dict(base)
        j["meta"] = dict(base["meta"], stage1_id=base["id"], variant=name)
        j["id"] = f"s2_{task}_{arm}_s{seed}_{name}{tag}"
        j["out"] = str(out_dir / f"raw_{fam}" / (j["id"] + ".json"))
        j["adapt"] = {"mode": mode, "r": r}
        j["reads"] = reads
        j["checkpoints"] = [0, final] if with0 else [final]
        j["mask_seed"] = mask_seed
        jobs.append(j)
    return jobs


def write(path, jobs):
    with open(path, "w") as f:
        for j in jobs:
            f.write(json.dumps(j, sort_keys=True) + "\n")
    print(path, len(jobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    nl = [json.loads(l) for l in open(a.stage1 / "nlplr_jobs.jsonl")]
    bb = [json.loads(l) for l in open(a.stage1 / "raw_bbh/jobs_bbh.jsonl")]
    bb = [j for j in bb if j["id"].startswith("generic_bbh_")]
    for j in nl + bb:
        assert j.get("loss") == "raw" and j.get("seed", j.get("optimizer_seed", 0)) == 0, j["id"]

    def build(rows, fam, final, dev):
        jobs = []
        for b in rows:
            arm, seed, task = b["meta"]["arm"], b["meta"]["seed"], b["task"]
            if arm not in VARIANTS:
                continue
            is_dev = seed in DEV_SEEDS and (fam == "bbh" or task in DEV_NLPLR)
            if dev != is_dev:
                continue
            v = list(VARIANTS[arm]) + (BBH_PRUNE if fam == "bbh" and arm == "D" else [])
            jobs += expand(b, fam, a.out, final, v)
        # dense jobs first (they carry the step-0 and Stage 1 baseline reads), then the main sparse runs
        order = {"dense": 0, "topk75": 1, "topk50": 3, "fixed75": 4, "rand75": 5, "prune75": 6, "prune50": 6}
        jobs.sort(key=lambda j: (order[j["meta"]["variant"]] + (1 if j["meta"]["arm"] == "A" else 0) * 0.5,
                                 j["task"], j["meta"]["seed"]))
        return jobs

    write(a.out / "jobs_nlplr_dev.jsonl", build(nl, "nlplr", 200, True))
    write(a.out / "jobs_nlplr.jsonl", build(nl, "nlplr", 200, False))
    write(a.out / "jobs_bbh_dev.jsonl", build(bb, "bbh", 16, True))
    write(a.out / "jobs_bbh.jsonl", build(bb, "bbh", 16, False))
    # random-mask seed variation on the dev cells (guideline: check the spread across mask seeds)
    rs = []
    for rows, fam, final in ((nl, "nlplr", 200), (bb, "bbh", 16)):
        for b in rows:
            if b["meta"]["arm"] == "D" and b["meta"]["seed"] in DEV_SEEDS and (fam == "bbh" or b["task"] in DEV_NLPLR):
                for ms in (1, 2):
                    rs += expand(b, fam, a.out, final, [VARIANTS["D"][4]], mask_seed=ms, tag=f"_ms{ms}")
    write(a.out / "jobs_dev_randseed_nlplr.jsonl", [j for j in rs if "boolean" not in j["task"]])
    write(a.out / "jobs_dev_randseed_bbh.jsonl", [j for j in rs if "boolean" in j["task"]])
    # serial cost benchmark (absolute output paths: the BBH runner runs from another directory)
    bench = (a.out / "bench").resolve()
    bench.mkdir(exist_ok=True)
    for rows, fam, final in ((nl, "nlplr", 200), (bb, "bbh", 16)):
        jobs = []
        for b in rows:
            if b["meta"]["arm"] == "D" and b["meta"]["seed"] in DEV_SEEDS and (fam == "bbh" or b["task"] in DEV_NLPLR):
                jobs += expand(b, fam, bench, final, BENCH + (BBH_PRUNE if fam == "bbh" else []), tag="_bench")
        write(bench / f"jobs_bench_{fam}.jsonl", jobs)


if __name__ == "__main__":
    main()
