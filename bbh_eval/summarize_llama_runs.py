"""Summarize Llama BBH runs and compare predictions at identical cache sizes."""
import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


CONDITIONS = (
    "task", "split", "n_scored", "model", "revision", "gpu", "dtype",
    "protocol", "prompt_protocol", "manifest_sha256", "microbatch", "optimizer_seed",
)
GROUP = CONDITIONS + ("max_new_tokens",) + ("method", "budget", "whole_demo_budget", "steps")
MEASURES = (
    "accuracy_pct", "antecedent_accuracy_pct", "retained_cache_bytes", "adaptation_s", "inference_s",
    "peak_allocated_bytes", "adaptation_peak_bytes", "inference_peak_bytes",
)
DELTAS = ("accuracy_delta_pp", "adaptation_delta_s", "peak_delta_bytes")


def read_run(path):
    meta = json.loads(path.read_text())
    if meta.get("status") != "complete" or meta.get("method") not in (
        "icl", "full", "random", "answer_attention", "value_norm", "kvzip", "kvzip_native", "kvzip_native_random",
    ):
        return None
    predictions = [
        json.loads(line)
        for line in (path.parent / "predictions.jsonl").read_text().splitlines()
    ]
    rows = {r["row_id"]: (r["gold"], r["exact_match"]) for r in predictions}
    if any(type(correct) is not bool for _, correct in rows.values()):
        raise ValueError(f"{path.parent}: expected boolean exact_match values.")
    if (not rows or len(rows) != len(predictions) or len(rows) != meta["n_scored"]
            or sum(correct for _, correct in rows.values()) != meta["n_correct"]):
        raise ValueError(f"{path.parent}: prediction counts disagree with metadata.")
    expected = meta.get("evaluation_ids")
    if expected is not None and list(rows) != expected:
        raise ValueError(f"{path.parent}: prediction IDs disagree with the run metadata.")
    result = {
        "run": str(path.parent), **{k: meta.get(k) for k in GROUP},
        "seed": meta["seed"], "selection_seed": meta.get("selection_seed"),
        "accuracy_pct": 100 * meta["n_correct"] / len(rows),
        "n_correct": meta["n_correct"], "n_invalid": meta.get("n_invalid"), "n_shots": meta.get("n_shots"),
        "requested_demo_tokens": meta.get("requested_demo_tokens"),
        "n_available_eval": meta.get("n_available_eval"),
        "cache_byte_source": meta.get("cache_byte_source"),
        "n_retained_demo_tokens": meta.get("n_retained_demo_tokens"),
        **{k: meta.get(k) for k in MEASURES if k != "accuracy_pct"},
        "protocol": meta.get("protocol") or meta.get("prompt_version"),
        "_demos": tuple(meta.get("eligible_demo_ids") or meta["demo_ids"]),
        "_rows": rows,
    }
    if "antecedent_n_correct" in meta:
        marks = [r.get("antecedent_exact_match") for r in predictions]
        if any(type(ok) is not bool for ok in marks) or sum(marks) != meta["antecedent_n_correct"]:
            raise ValueError(f"{path.parent}: normalized score disagrees with metadata.")
        result["antecedent_accuracy_pct"] = 100 * sum(marks) / len(rows)
    return result


def grouped(records, fields):
    buckets = defaultdict(list)
    for row in records:
        buckets[tuple(row.get(f) for f in fields)].append(row)
    return buckets


def aggregate(records, fields, measures):
    for key, members in grouped(records, fields).items():
        by_seed = grouped(members, ("seed",))
        summary = dict(zip(fields, key))
        summary.update(n_runs=len(members), n_demo_seeds=len(by_seed))
        for metric in measures:
            means = []
            for runs in by_seed.values():
                values = [run[metric] for run in runs if run.get(metric) is not None]
                if len(values) == len(runs):
                    means.append(statistics.mean(values))
            summary[metric + "_n_seeds"] = len(means)
            summary[metric + "_mean"] = statistics.mean(means) if means else None
            summary[metric + "_sd"] = statistics.stdev(means) if len(means) > 1 else None
        yield summary


def difference(left, right, key):
    if left.get(key) is not None and right.get(key) is not None:
        return left[key] - right[key]
    return None


def compare(runs):
    pair_fields = CONDITIONS + ("seed", "budget", "steps", "max_new_tokens")
    candidates = grouped([r for r in runs if r["method"] in ("random", "kvzip_native_random")],
                         pair_fields + ("method",))
    pairs = []
    for selected in (r for r in runs if r["method"] in ("answer_attention", "value_norm", "kvzip", "kvzip_native")):
        key = tuple(selected.get(f) for f in pair_fields)
        random_method = "kvzip_native_random" if selected["method"] == "kvzip_native" else "random"
        matched = 0
        for random_run in candidates.get(key + (random_method,), []):
            if selected["_demos"] != random_run["_demos"]:
                continue
            if not selected["_rows"].keys() == random_run["_rows"].keys():
                continue
            if any(row[0] != random_run["_rows"][i][0] for i, row in selected["_rows"].items()):
                continue
            saved = selected.get("retained_cache_bytes")
            if saved is None or saved != random_run.get("retained_cache_bytes"):
                continue
            wins = sum(ok and not random_run["_rows"][i][1]
                       for i, (_, ok) in selected["_rows"].items())
            losses = sum(not ok and random_run["_rows"][i][1]
                         for i, (_, ok) in selected["_rows"].items())
            pairs.append({
                **dict(zip(pair_fields, key)), "selected_method": selected["method"],
                "random_method": random_method, "selected": selected["run"],
                "random": random_run["run"], "random_selection_seed": random_run["selection_seed"],
                "retained_cache_bytes": saved, "selected_only_correct": wins,
                "random_only_correct": losses,
                "accuracy_delta_pp": 100 * (wins - losses) / len(selected["_rows"]),
                "adaptation_delta_s": difference(selected, random_run, "adaptation_s"),
                "peak_delta_bytes": difference(selected, random_run, "peak_allocated_bytes"),
            })
            matched += 1
        if not matched:
            print(f"No equal-byte, question-matched random comparison: {selected['run']}")
    return pairs, pair_fields


def save_table(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        out = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        out.writeheader()
        out.writerows(rows)


def summary_fields(keys, measures):
    return list(keys) + ["n_runs", "n_demo_seeds"] + [
        m + suffix for m in measures for suffix in ("_n_seeds", "_mean", "_sd")
    ]


def summarize_benchmark(runs, manifest_path, directory):
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    tasks = manifest["tasks"]
    if len(tasks) != 27 or manifest.get("protocol") != "antecedent_bbh27_v1":
        raise ValueError("Expected the reconstructed benchmark's 27-file manifest.")
    digest = hashlib.sha256(raw).hexdigest()
    fields = tuple(k for k in GROUP if k not in ("task", "n_scored", "max_new_tokens"))
    members = [r for r in runs if r.get("manifest_sha256") == digest]
    if not members:
        raise ValueError("No runs have the requested benchmark manifest SHA-256.")
    coverage = []
    statistics_rows = []
    for key, group in grouped(members, fields).items():
        condition = dict(zip(fields, key))
        seed_macros, normalized_seed_macros = [], []
        for seed in manifest["seeds"]:
            complete = defaultdict(list)
            for record in group:
                if record["seed"] != seed or record["task"] not in tasks:
                    continue
                task = tasks[record["task"]]
                expected = task["seeds"][str(seed)]
                if (list(record["_rows"]) == expected["test_ids"]
                        and list(record["_demos"]) == expected["demo_ids"]):
                    complete[record["task"]].append(record)
            for task_name, task_runs in complete.items():
                if len({r["max_new_tokens"] for r in task_runs}) != 1:
                    raise ValueError(f"Mixed output caps for {task_name}, seed {seed}.")
            missing = sorted(tasks.keys() - complete.keys())
            macro = normalized_macro = None
            if not missing:
                macro = statistics.mean(statistics.mean(r["accuracy_pct"] for r in records)
                                        for records in complete.values())
                seed_macros.append(macro)
                if all(r.get("antecedent_accuracy_pct") is not None
                       for task_runs in complete.values() for r in task_runs):
                    normalized_macro = statistics.mean(
                        statistics.mean(r["antecedent_accuracy_pct"] for r in records)
                        for records in complete.values())
                    normalized_seed_macros.append(normalized_macro)
            coverage.append({**condition, "seed": seed, "n_complete_files": len(complete),
                             "expected_files": len(tasks), "missing_task_files": " ".join(missing),
                             "macro_accuracy_pct": macro, "antecedent_macro_accuracy_pct": normalized_macro})
        all_done = len(seed_macros) == len(manifest["seeds"])
        all_normalized = len(normalized_seed_macros) == len(manifest["seeds"])
        statistics_rows.append({**condition,
                                "antecedent_macro_pct_mean": statistics.mean(normalized_seed_macros)
                                if all_normalized else None,
                                "antecedent_macro_pct_sd": statistics.stdev(normalized_seed_macros)
                                if all_normalized and len(normalized_seed_macros) > 1 else None, "complete_seeds": len(seed_macros),
                                "expected_seeds": len(manifest["seeds"]), "complete_grid": all_done,
                                "benchmark_accuracy_pct_mean": statistics.mean(seed_macros) if all_done else None,
                                "benchmark_accuracy_pct_sd": statistics.stdev(seed_macros)
                                if all_done and len(seed_macros) > 1 else None})
    save_table(directory / "benchmark_seeds.csv", coverage,
               list(fields) + ["seed", "n_complete_files", "expected_files",
                               "missing_task_files", "macro_accuracy_pct", "antecedent_macro_accuracy_pct"])
    save_table(directory / "benchmark_summary.csv", statistics_rows, list(fields) + [
        "complete_seeds", "expected_seeds", "complete_grid",
        "benchmark_accuracy_pct_mean", "benchmark_accuracy_pct_sd",
        "antecedent_macro_pct_mean", "antecedent_macro_pct_sd",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Directory containing run folders.")
    parser.add_argument("--out", required=True, type=Path, help="Directory for the generated CSV tables.")
    parser.add_argument("--benchmark-manifest", type=Path,
                        help="Also check all tasks and seeds against this benchmark manifest.")
    args = parser.parse_args()
    runs = []
    for path in sorted(args.root.rglob("metadata.json")):
        row = read_run(path)
        if row is None:
            print(f"Skipped incomplete or unrelated run: {path.parent}")
        else:
            runs.append(row)
    if not runs:
        parser.error(f"No complete Llama runs found under {args.root}")
    for bucket in grouped(runs, GROUP + ("seed", "selection_seed")).values():
        if len(bucket) > 1:
            parser.error("Duplicate runs in input. Use a root with one copy of each run: "
                         + ", ".join(row["run"] for row in bucket))
    args.out.mkdir(parents=True, exist_ok=True)
    save_table(args.out / "runs.csv", runs,
               list(k for k in runs[0] if not k.startswith("_")))
    save_table(args.out / "task_summary.csv", aggregate(runs, GROUP, MEASURES),
               summary_fields(GROUP, MEASURES))
    pairs, pair_fields = compare(runs)
    save_table(args.out / "paired_runs.csv", pairs,
               list(pair_fields) + ["selected_method", "random_method", "selected", "random", "random_selection_seed",
                                    "retained_cache_bytes", "selected_only_correct",
                                    "random_only_correct"] + list(DELTAS))
    summary_keys = tuple(k for k in pair_fields if k != "seed") + ("selected_method",)
    save_table(args.out / "paired_task_summary.csv", aggregate(pairs, summary_keys, DELTAS),
               summary_fields(summary_keys, DELTAS))
    if args.benchmark_manifest:
        summarize_benchmark(runs, args.benchmark_manifest, args.out)
    print(f"Saved {len(runs)} runs and {len(pairs)} matched comparisons to {args.out}")
    print("Blank means unavailable. SD uses demo seeds and is blank for fewer than two.")


if __name__ == "__main__":
    main()
