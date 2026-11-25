"""Make a side-by-side 5/10/20-shot table from the completed, audited two-task grid."""
import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_rerun import configuration, write_csv
from run_matrix import HERE, SEEDS, SHOTS, jobs_for


def cell(config, task, seed):
    return (config["method"], float(config["retention"]), int(config["updates"]),
            task, int(seed), int(config["shots"]))


def load_scores(folder):
    # analyze_rerun.py checks the prediction IDs, gold labels, demo prefixes
    # and environments before writing the summary. Here we check its grid.
    manifest = json.loads((HERE / "data/manifest.json").read_text())
    jobs = jobs_for(manifest, "shots", "two")
    expected = {cell(configuration(job), job.task, job.seed) for job in jobs}
    audit = json.loads((folder / "analysis.json").read_text())
    scores, states = {}, Counter()
    with (folder / "runs.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = cell(row, row["task"], row["seed"])
            if key in scores:
                raise ValueError(f"Duplicate result for {key}")
            state = row["status"]
            states[state] += 1
            scores[key] = float(row["accuracy_pct"]) if state == "complete" else None
    if scores.keys() != expected or audit["unique_runs_expected"] != len(expected):
        raise ValueError("Result inventory differs from the planned grid. Run the analysis again.")
    if dict(states) != audit["unique_run_counts"]:
        raise ValueError("The run table and analysis summary disagree. Run the analysis again.")
    if states.keys() - {"complete", "infeasible"}:
        raise ValueError(f"Grid is unfinished: {dict(states)}. See runs.csv for the outstanding runs.")
    return scores


def make_rows(scores):
    conditions = defaultdict(dict)
    for (method, retention, updates, task, seed, shots), accuracy in scores.items():
        conditions[method, retention, updates][task, seed, shots] = accuracy
    for (method, retention, updates), results in sorted(conditions.items()):
        tasks = sorted({task for task, _, _ in results})
        for label, group in [("macro_2_tasks", tasks)] + [(task, [task]) for task in tasks]:
            row = dict(task=label, method=method, retention=retention, updates=updates,
                       n_expected_per_shot=len(group) * len(SEEDS))
            for shots in SHOTS:
                values = [results[task, seed, shots] for task in group for seed in SEEDS]
                complete = [x for x in values if x is not None]
                row[f"n_complete_{shots}"] = len(complete)
                if len(complete) == len(values):
                    row[f"accuracy_{shots}_pct"] = statistics.mean(complete)
            for low, high in ((5, 10), (10, 20), (5, 20)):
                if f"accuracy_{low}_pct" not in row or f"accuracy_{high}_pct" not in row:
                    continue
                # Pair each task and seed before averaging. Macro SD describes
                # variation across the three equally task-weighted seed means.
                deltas = [statistics.mean(results[task, seed, high] - results[task, seed, low]
                                          for task in group) for seed in SEEDS]
                row[f"delta_{high}_minus_{low}_pp"] = statistics.mean(deltas)
                row[f"sd_{high}_minus_{low}_pp"] = statistics.stdev(deltas)
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True,
                        help="Output directory just written by analyze_rerun.py.")
    args = parser.parse_args()
    scores = load_scores(args.analysis)
    columns = ["task", "method", "retention", "updates", "n_expected_per_shot"]
    columns += [f"n_complete_{k}" for k in SHOTS] + [f"accuracy_{k}_pct" for k in SHOTS]
    for low, high in ((5, 10), (10, 20), (5, 20)):
        columns.extend([f"delta_{high}_minus_{low}_pp", f"sd_{high}_minus_{low}_pp"])
    write_csv(args.analysis, "shot_effects", columns, make_rows(scores))
    print(f"{args.analysis / 'shot_effects.csv'} | {len(scores)} planned runs accounted for. "
          "Blank scores or deltas indicate an infeasible constituent run.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"Shot comparison unavailable: {error}")
