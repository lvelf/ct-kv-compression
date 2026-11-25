#!/usr/bin/env python3
"""Audit the expected BBH runs, then summarize matched contrasts without imputing missing cells."""
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from run_matrix import jobs_for, job_status, run_dir

COSTS = ("retained_cache_mib", "peak_gpu_gib", "adaptation_s", "inference_s")
METRICS = ("accuracy_pct",) + COSTS
DELTA_METRICS = ("accuracy_delta_pp",) + tuple(m + "_delta" for m in COSTS)
CONFIG = ("shots", "method", "retention", "updates")
GROUP = ("design", "comparison") + CONFIG
ENV = ("model", "revision", "gpu", "dtype", "torch", "transformers", "cuda", "scoring", "prompt_protocol")
DETAIL = ("n_scored", "n_correct", "n_retained_demo_tokens", "cache_byte_source")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def configuration(job):
    method = "full_cache" if job.method in ("icl", "full") else job.method
    return dict(shots=job.shots, method=method, retention=job.budget, updates=job.steps)


def write_csv(folder, name, columns, records):
    path = folder / (name + ".csv")
    with tempfile.NamedTemporaryFile(mode="w", dir=folder, prefix="." + name,
                                     newline="", encoding="utf-8", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


@dataclass
class Record:
    values: dict
    env: tuple
    eval_hash: str
    eligible: tuple
    selected: tuple
    positions_hash: str
    cap: int
    retained_bytes: int


def read_result(path, job):
    meta = read_json(path / "metadata.json")
    pairs, correct = [], 0
    with (path / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            ok = row["exact_match"]
            if type(ok) is not bool:
                raise ValueError(f"{path}: expected boolean exact_match")
            pairs.append((row["row_id"], row["gold"]))
            correct += ok
    ids = [row[0] for row in pairs]
    if (not ids or ids != meta["evaluation_ids"] or len(ids) != len(set(ids))
            or (len(ids), correct) != (meta["n_scored"], meta["n_correct"])):
        raise ValueError(f"{path}: prediction IDs, count or correctness disagree with metadata")
    env = tuple(meta[k] for k in ENV)
    if any(v is None or v == "" for v in env):
        raise ValueError(f"{path}: missing run environment")
    rank_positions = meta.get("retained_demo_indices")
    if job.method == "answer_attention" and rank_positions is None:
        raise ValueError(f"{path}: missing selected token indices")
    values = {
        **{k: meta[k] for k in DETAIL}, "accuracy_pct": 100 * correct / len(ids),
        "retained_cache_mib": meta["retained_cache_bytes"] / (2 ** 20),
        "peak_gpu_gib": meta["peak_allocated_bytes"] / (2 ** 30),
        "adaptation_s": meta["adaptation_s"], "inference_s": meta["inference_s"]
    }
    if any(not math.isfinite(values[k]) or values[k] < 0 for k in METRICS):
        raise ValueError(f"{path}: invalid score or cost")
    return Record(values, env, digest(pairs), tuple(meta["eligible_demo_ids"]),
                  tuple(meta["selected_demo_ids"]), digest(rank_positions), meta["max_new_tokens"],
                  meta["retained_cache_bytes"])


def pair_check(left, right, kind, job):
    if left.eval_hash != right.eval_hash or left.cap != right.cap or left.env != right.env:
        raise ValueError(f"{job}: question IDs/golds, decoding cap or run environment differ")
    if kind == "shots_minus_10":
        short, long = sorted((left.eligible, right.eligible), key=len)
        if short != long[:len(short)]:
            raise ValueError(f"{job}: demonstration lists are not nested")
        return
    if kind == "retention_minus_full":
        if left.eligible != right.eligible:
            raise ValueError(f"{job}: compressed and full-cache runs used different original demos")
        return
    if left.eligible != right.eligible or left.retained_bytes != right.retained_bytes:
        raise ValueError(f"{job}: paired runs differ in original demos or retained cache bytes")
    if left.selected != right.selected:
        raise ValueError(f"{job}: paired runs chose different demonstrations")
    if kind == "16_minus_0" and job.method == "answer_attention":
        if left.positions_hash != right.positions_hash:
            raise ValueError(f"{job}: 16-update and zero-update runs retained different positions")


def untrained(job):
    method = "icl" if job.method == "full" else job.method
    return replace(job, method=method, steps=0)


def contrast_jobs(job, design):
    if job.budget < 1:
        reference = "full" if job.steps else "icl"
        yield "retention_minus_full", (job, replace(job, method=reference, budget=1.))
    if job.steps == 16:
        yield "16_minus_0", (job, untrained(job))
    if design == "shots" and job.shots != 10:
        yield "shots_minus_10", (job, replace(job, shots=10))


def compare(jobs, records, kind):
    left, right = (records[job] for job in jobs)
    pair_check(left, right, kind, jobs[0])
    return {delta: left.values[metric] - right.values[metric]
            for metric, delta in zip(METRICS, DELTA_METRICS)}


def grouped(rows, group):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in group)].append(row)
    return sorted(buckets.items())


def summary_metrics(members, metrics):
    # Each task gets equal weight within a seed. SD describes the three seed means.
    by_seed = defaultdict(list)
    for row in members:
        by_seed[row["seed"]].append(row)
    result = {}
    for metric in metrics:
        means = [statistics.mean(m[metric] for m in items) for items in by_seed.values()]
        result[metric] = statistics.mean(means)
        result[metric + "_sd_seeds"] = statistics.stdev(means) if len(means) > 1 else None
    return result


def summaries(rows, group, metrics):
    for names, members in grouped(rows, group):
        complete = [r for r in members if r["status"] == "complete"]
        result = dict(zip(group, names))
        result.update(n_expected=len(members), n_complete=len(complete),
                      status="complete" if len(members) == len(complete) else "partial")
        if result["status"] == "complete":
            result.update(summary_metrics(members, metrics))
        yield result


def eligible_summaries(rows, metrics):
    for names, members in grouped(rows, GROUP):
        included, infeasible, outstanding = [], [], []
        eligible = []
        for (task,), task_rows in grouped(members, ("task",)):
            if sorted(r["seed"] for r in task_rows) != [0, 1, 2]:
                raise ValueError(f"Expected exactly one row for each of three seeds for {names}, {task}.")
            states = {r["status"] for r in task_rows}
            if states == {"complete"}:
                included.append(task)
                eligible.extend(task_rows)
            elif states <= {"complete", "infeasible"}:
                infeasible.append(task)
            else:
                outstanding.append(task)
        result = dict(zip(GROUP, names))
        result.update(n_tasks_expected=len(included) + len(infeasible) + len(outstanding),
                      n_tasks_eligible=len(included), included_tasks=" ".join(included),
                      infeasible_tasks=" ".join(infeasible), outstanding_tasks=" ".join(outstanding))
        if outstanding:
            result["status"] = "awaiting_jobs"
        elif not included:
            result["status"] = "no_eligible_tasks"
        else:
            result["status"] = "eligible_subset" if infeasible else "full_grid"
            result.update(summary_metrics(eligible, metrics))
        yield result


def summary_tables(out, name, rows, metrics):
    measures = tuple(k for metric in metrics for k in (metric, metric + "_sd_seeds"))
    for suffix, fields in (("per_task", GROUP + ("task",)), ("macro", GROUP)):
        columns = fields + ("status", "n_complete", "n_expected")
        write_csv(out, name + "_" + suffix, columns + measures, summaries(rows, fields, metrics))
    coverage = ("status", "n_tasks_expected", "n_tasks_eligible", "included_tasks",
                "infeasible_tasks", "outstanding_tasks")
    write_csv(out, name + "_eligible_macro", GROUP + coverage + measures, eligible_summaries(rows, metrics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--design", choices=("shots",), default="shots")
    parser.add_argument("--shot-tasks", choices=("two",), default="two")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    designs = (args.design,)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error("experiment root does not exist")
    out = args.out or root / ("analysis-" + args.design + "-" + args.shot_tasks)
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    manifest_hash = hashlib.sha256(raw).hexdigest()
    grids = {d: list(jobs_for(manifest, d, args.shot_tasks)) for d in designs}
    all_jobs = dict.fromkeys(j for grid in grids.values() for j in grid)
    records, states, reasons, envs = {}, {}, {}, set()
    by_task_heldout = {}
    # Each task uses the same held-out questions across all seeds and conditions.
    for job in all_jobs:
        folder = run_dir(root, job)
        states[job] = job_status(root, job, manifest_hash, manifest["tasks"][job.task])
        if states[job] == "complete":
            record = read_result(folder, job)
            if by_task_heldout.setdefault(job.task, record.eval_hash) != record.eval_hash:
                raise ValueError(f"{job}: question IDs or golds disagree across conditions")
            records[job] = record
            envs.add(record.env)
        elif states[job] == "infeasible":
            meta = read_json(folder / "metadata.json")
            reasons[job] = meta.get("reason") or meta.get("infeasible_reason", "")
    if len(envs) > 1:
        raise ValueError("Mixed model/GPU/software/scoring environments in this grid: " + repr(envs))
    coverage, pairs, absolute = [], [], []
    for design, jobs in grids.items():
        for job in jobs:
            common = dict(design=design, **configuration(job), task=job.task, seed=job.seed)
            status = states[job]
            row = dict(**common, comparison="absolute", status=status)
            if status == "complete":
                row.update(records[job].values)
            absolute.append(row)
            coverage.append(dict(**row, reason=reasons.get(job, ""), run=str(run_dir(root, job))))
            for kind, counterparts in contrast_jobs(job, design):
                missing = [j for j in counterparts if states[j] != "complete"]
                result = dict(**common, comparison=kind, status="unavailable" if missing else "complete")
                if missing:
                    if all(states[j] == "infeasible" for j in missing):
                        result["status"] = "infeasible"
                    result["reason"] = " | ".join(f"{run_dir(root,j).name}: {states[j]}" for j in missing)
                else:
                    result.update(compare(counterparts, records, kind))
                pairs.append(result)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out, "runs", GROUP + ("task", "seed", "status", "reason", "run") + DETAIL + METRICS, coverage)
    write_csv(out, "paired_runs", GROUP + ("task", "seed", "status", "reason") + DELTA_METRICS, pairs)
    summary_tables(out, "absolute", absolute, METRICS)
    summary_tables(out, "contrasts", pairs, DELTA_METRICS)
    summary = {
        "manifest_sha256": manifest_hash, "design": args.design, "shot_tasks": args.shot_tasks,
        "unique_run_counts": dict(Counter(states.values())), "unique_runs_expected": len(states),
        "environment": dict(zip(ENV, next(iter(envs)))) if envs else {},
        "pair_signs": {
            "retention_minus_full": "compressed minus uncompressed, at the same update count, eligible demonstrations and test IDs. The zero-update full reference is direct ICL.",
            "16_minus_0": "within the same method. For the full cache, the 0-update baseline is direct ICL.",
            "shots_minus_10": "requested shot count minus the same method's 10-shot result, on common questions with nested eligible demonstrations. Cache bytes may change."
        },
        "units": "accuracy percentage; accuracy differences are percentage points. Costs are left-minus-right in float MiB, GiB or seconds. SD is across demo-seed means.",
        "eligibility": "Strict macros require every task and seed. Supplemental eligible macros wait for every constituent job in that condition or contrast to be complete or infeasible, then average tasks with all three seeds complete on all sides. They print the task names. Compare across rows only when the included task lists match, or use the paired contrasts.",
        "costs": "For direct ICL, retained cache bytes are analytical and prefix prefill is included in inference; for fitted or zero-update cached methods bytes come from tensors."
    }
    with tempfile.NamedTemporaryFile(mode="w", dir=out, encoding="utf-8", delete=False) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(stream.name, out / "analysis.json")
    print(f"{len(records)}/{len(all_jobs)} unique runs complete. {dict(Counter(states.values()))}. Tables: {out}")
    if len(records) < len(all_jobs):
        print("Strict means require complete grids; eligible-subset means require every job in the contrast to be terminal. See coverage columns.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        sys.exit(f"Analysis failed: {exc}")
