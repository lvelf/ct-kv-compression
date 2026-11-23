"""Stage and resume independent BBH jobs. Run from the team repository root."""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

PAPER = ("meta-llama/Meta-Llama-3-8B-Instruct", "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2")
PROJECT = ("meta-llama/Llama-3.2-3B-Instruct", "0cb88a4f764b7a12671c53f0838cd831a0843b95")
TASKS = ("boolean_expressions", "date_understanding", "word_sorting", "logical_deduction_three_objects")
RULES = ("answer_attention", "value_norm")
FRACTIONS = (0.25, 0.5, 0.75)
TEAM_MANIFEST = Path("bbh_eval/splits.json")
STAGES = ("smoke", "paper-four", "paper-all", "develop", "freeze", "project-test", "whole-match")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def persist(path, obj):
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"Different locked configuration already exists: {path}. Choose a new output root.")
    if not path.exists():
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Run jobs; without this flag, print the plan.")
    parser.add_argument("--project-model", choices=("8b", "3b"), default="8b")
    parser.add_argument("--only-task", help="Optional subset, useful for timing a single task.")
    a = parser.parse_args()
    root = a.root.resolve()
    phase_a = a.stage in ("smoke", "paper-four", "paper-all")
    model, revision = PAPER if phase_a or a.project_model == "8b" else PROJECT
    split = "test" if a.stage in ("paper-all", "project-test", "whole-match") else "dev"
    manifest = (root / "benchmark27.json" if a.stage == "paper-all" else root / "four5.json" if phase_a else TEAM_MANIFEST.resolve())
    tasks = sorted(read(manifest)["tasks"]) if a.stage == "paper-all" else list(TASKS)
    if a.only_task:
        if a.only_task not in tasks or a.stage in ("smoke", "freeze"):
            parser.error("--only-task is unavailable for the specified task or stage.")
        tasks = [a.only_task]
    seed_range = range(5 if phase_a else 3)
    manifest_hash = digest(manifest)
    if a.stage == "paper-all" and a.execute and not a.only_task:
        print("Detailed paper-reconstruction outputs are logged under paper-all; reserve project selection for project dev only.")
    freeze_file = root / f"selection-{a.project_model}.json"
    jobs = []
    def add(task, seed, method, steps, budget=1.0, whole=None, limit=None, variant=""):
        name = f"{task}-s{seed}-{method}-b{budget:g}-u{steps}"
        if whole is not None:
            name += f"-whole{whole:g}"
        if variant:
            name += "-" + variant
        folder = root / a.stage / ("8b" if phase_a else a.project_model) / name
        cmd = [sys.executable, "-m", "bbh_eval.run_llama_icl", "--model", model, "--revision", revision,
               "--task", task, "--seed", str(seed), "--split", split, "--manifest", str(manifest),
               "--method", method, "--steps", str(steps), "--budget", str(budget), "--selection-seed", str(seed),
               "--optimizer-seed", "0", "--out", str(folder)]
        if whole is not None:
            cmd.extend(["--whole-demo-budget", str(whole)])
        if limit is not None:
            cmd.extend(["--limit", str(limit)])
        if split == "test":
            cmd.append("--frozen")
        jobs.append((folder, cmd))

    if a.stage == "smoke":
        for method, steps in (("icl", 0), ("full", 0), ("full", 1), ("full", 16)):
            add(TASKS[0], 0, method, steps, limit=3)
    elif a.stage.startswith("paper"):
        for method, steps in (("icl", 0), ("full", 16)):
            for task in tasks:
                for seed in seed_range:
                    add(task, seed, method, steps)
    elif a.stage == "develop":
        for task in tasks:
            for seed in seed_range:
                for method, steps, budget in (("icl", 0, 1.), ("full", 16, 1.), ("random", 16, .5),
                                              (RULES[0], 16, .5), (RULES[1], 16, .5)):
                    add(task, seed, method, steps, budget)
    elif a.stage == "freeze":
        if not a.execute:
            print("Select from both candidates on four tasks x three dev seeds at 50% retention.")
            return
        evidence = {}
        for method in RULES:
            rows = []
            for task in TASKS:
                for seed in range(3):
                    meta = read(root / "develop" / a.project_model / f"{task}-s{seed}-{method}-b0.5-u16" / "metadata.json")
                    if (meta["status"], meta["n_scored"], meta["split"], meta["revision"], meta["manifest_sha256"]) != (
                            "complete", 40, "dev", revision, manifest_hash):
                        raise ValueError("Mismatched development run.")
                    rows.append((meta["n_correct"]/40, meta["adaptation_s"]))
            evidence[method] = {"correct_out_of_480": round(sum(r[0] * 40 for r in rows)),
                                "mean_adaptation_s": statistics.mean(r[1] for r in rows)}
        winner = min(RULES, key=lambda k: (-evidence[k]["correct_out_of_480"], evidence[k]["mean_adaptation_s"], RULES.index(k)))
        persist(freeze_file, {"selection_rule": winner, "development": evidence, "model": model,
                              "revision": revision, "manifest_sha256": manifest_hash, "steps": 16,
                              "test_fractions": FRACTIONS, "rank_by": "equal-weight dev accuracy, then lower mean adaptation time"})
        print(f"Locked {winner} in {freeze_file}")
        return
    else:
        locked = read(freeze_file)
        if locked["revision"] != revision or locked["manifest_sha256"] != manifest_hash:
            raise ValueError("Model or split differs from locked development protocol.")
        winner = locked["selection_rule"]
        for task in tasks:
            for seed in seed_range:
                if a.stage == "whole-match":
                    base = root / "project-test" / a.project_model
                    for frac in FRACTIONS:
                        whole = read(base / f"{task}-s{seed}-full-b1-u16-whole{frac:g}" / "metadata.json")
                        reference = read(base / f"{task}-s{seed}-full-b1-u16" / "metadata.json")
                        tokens = whole["n_retained_demo_tokens"]
                        total = reference["n_retained_demo_tokens"]
                        if whole["status"] != "complete" or reference["status"] != "complete":
                            raise ValueError("Run project-test before whole-match.")
                        if tokens == max(1, int(total * frac)):
                            continue
                        actual = (tokens + .25) / total
                        add(task, seed, winner, 16, actual, variant=f"actual-whole{frac:g}")
                        add(task, seed, "random", 16, actual, variant=f"actual-whole{frac:g}")
                    continue
                add(task, seed, "icl", 0)
                add(task, seed, "full", 16)
                for frac in FRACTIONS:
                    add(task, seed, "random", 16, frac)
                    add(task, seed, winner, 16, frac)
                    add(task, seed, winner, 0, frac)
                    add(task, seed, "icl", 0, whole=frac)
                    add(task, seed, "full", 16, whole=frac)

    print(f"{a.stage}: {len(jobs)} jobs; model={model}; manifest={manifest}; output={root}", flush=True)
    if not a.execute:
        print("Preview only. Add --execute to launch.")
        return
    failures = []
    for folder, cmd in jobs:
        metadata = folder / "metadata.json"
        # The child creates a fresh directory. Keep launch contracts outside each result.
        contract = root / "requests" / folder.relative_to(root) / "job-request.json"
        contract.parent.mkdir(parents=True, exist_ok=True)
        persist(contract, {"args": cmd[3:], "manifest_sha256": manifest_hash})
        if metadata.exists() and read(metadata).get("status") == "complete":
            continue
        if folder.exists():
            # Retain interrupted output outside all result-summary roots.
            destination = root / "interrupted" / folder.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            n = 1
            while destination.with_name(destination.name + f".{n}").exists():
                n += 1
            folder.rename(destination.with_name(destination.name + f".{n}"))
        print("Running", folder.name, flush=True)
        if a.stage == "paper-all":
            log = contract.with_name("latest-run.log")
            with log.open("w") as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            if result.returncode:
                print(f"See {log}", flush=True)
        else:
            result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append(folder.name)
            break
    if failures:
        print("Unfinished configurations:", *failures, sep="\n")
        raise SystemExit(1)
    if a.stage == "smoke":
        def predictions(folder):
            return [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
        direct, cached = predictions(jobs[0][0]), predictions(jobs[1][0])
        diffs = [(x["row_id"], y["row_id"]) for x, y in zip(direct, cached)
                 if x["row_id"] != y["row_id"] or x["generated_token_ids"] != y["generated_token_ids"]]
        if len(direct) != 3 or len(cached) != 3 or diffs:
            raise SystemExit(f"Review direct ICL vs cached ICL greedy parity before the larger run: {diffs}")
        print("All three direct and cached greedy outputs match. Inspect the scored answers; low accuracy alone is not a software failure.")


if __name__ == "__main__":
    main()
