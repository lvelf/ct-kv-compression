"""Plan, execute and resume the Boolean and Date answer-attention shot comparison. Run from the installed study directory."""
import argparse
import collections
import dataclasses
import datetime
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from bbh_eval.prepare_benchmark_splits import BENCHMARK_HASHES
from bbh_eval.rerun_errors import CUDA_OOM_EXIT, CUDA_OOM_MARKER
from bbh_eval.run_llama_icl import MODEL, MODEL_REV

HERE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
SHOTS = (5, 10, 20)
BUDGETS = (.25, .5, .75)
TASKS = ("boolean_expressions", "date_understanding")


@dataclasses.dataclass(frozen=True)
class Job:
    task: str
    seed: int
    shots: int
    method: str
    budget: float
    steps: int
    whole_demo_budget: float | None = None

    @property
    def relative_path(self):
        name = f"{self.method}-b{self.budget:g}-u{self.steps}"
        if self.whole_demo_budget is not None:
            name += f"-whole{self.whole_demo_budget:g}"
        return Path(self.task, f"s{self.seed}", f"k{self.shots}", name)


def conditions(task, seed, shots):
    yield Job(task, seed, shots, "icl", 1., 0)
    yield Job(task, seed, shots, "full", 1., 16)
    for budget in BUDGETS:
        for steps in (0, 16):
            yield Job(task, seed, shots, "answer_attention", budget, steps)


def jobs_for(manifest, design="shots", shot_tasks="two"):
    if set(manifest["tasks"]) != set(BENCHMARK_HASHES):
        raise ValueError("Expected the pinned BBH manifest.")
    if (design, shot_tasks) != ("shots", "two"):
        raise ValueError("This kit runs Boolean Expressions and Date Understanding at 5, 10 and 20 shots.")
    return [job for k in SHOTS for task in sorted(TASKS) for seed in SEEDS
            for job in conditions(task, seed, k)]


def run_dir(root, job):
    return Path(root) / "runs" / job.relative_path


def request_dir(root, job):
    return Path(root) / "requests" / job.relative_path


def load(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


@functools.lru_cache(maxsize=1)
def worker_fingerprint():
    """Behavior and task splits are pinned alongside the model revision and run request."""
    h = hashlib.sha256()
    module_names = ("rerun_worker", "rerun_data", "rerun_errors", "ctkv", "whole_demo", "run_llama_icl",
                    "prepare_splits", "prepare_benchmark_splits")
    for name in module_names:
        p = HERE / "bbh_eval" / f"{name}.py"
        if p.exists():
            h.update(p.name.encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def expected_request(job, manifest_hash, limit):
    return dict(**dataclasses.asdict(job), manifest_sha256=manifest_hash,
                code_sha256=worker_fingerprint(), model=MODEL, revision=MODEL_REV,
                selection_seed=job.seed, optimizer_seed=0, microbatch=2, limit=limit)


def job_status(root, job, manifest_hash, task_config, limit=None):
    """Check request provenance, terminal settings and all ordered prediction IDs and scores."""
    folder = run_dir(root, job)
    request = request_dir(root, job) / "request.json"
    if request.exists():
        try:
            stored_request = load(request)
        except (ValueError, UnicodeError):
            return "mismatch"
        if stored_request != expected_request(job, manifest_hash, limit):
            return "mismatch"
    elif folder.exists():
        return "mismatch"
    try:
        if not folder.exists():
            return "pending"
        if (folder / "oom.json").is_file() or not (folder / "metadata.json").is_file():
            return "incomplete"
        meta = load(folder / "metadata.json")
        state = meta.get("status")
        if state not in ("complete", "infeasible"):
            return "incomplete"
        expected = dict(task=job.task, seed=job.seed, requested_shots=job.shots,
                        manifest_sha256=manifest_hash, method=job.method, steps=job.steps,
                        budget=job.budget, whole_demo_budget=job.whole_demo_budget,
                        model=MODEL, revision=MODEL_REV, optimizer_seed=0,
                        selection_seed=job.seed, microbatch=2)
        if any(meta.get(key) != value for key, value in expected.items()):
            return "mismatch"
        if state == "infeasible":
            return "infeasible" if meta.get("reason") else "incomplete"
        evaluation = task_config["test_ids"][:limit]
        eligible = task_config["seeds"][str(job.seed)]["demo_ids"][:job.shots]
        if (meta.get("evaluation_ids") != evaluation or meta.get("eligible_demo_ids") != eligible
                or meta.get("n_scored") != len(evaluation)):
            return "mismatch"
        n = correct = 0
        with (folder / "predictions.jsonl").open() as src:
            for line in src:
                row = json.loads(line)
                if (n >= len(evaluation) or row.get("row_id") != evaluation[n]
                        or type(row.get("exact_match")) is not bool):
                    return "incomplete"
                n += 1
                correct += row["exact_match"]
        if n != len(evaluation) or meta.get("n_correct") != correct:
            return "incomplete"
        return "complete"
    except (FileNotFoundError, ValueError, TypeError, KeyError, AttributeError):
        return "incomplete"


def run_one(args, root, job, manifest_hash, task_config, lock):
    folder = run_dir(root, job)
    records = request_dir(root, job)
    records.mkdir(parents=True, exist_ok=True)
    snapshot = expected_request(job, manifest_hash, args.limit)
    write_json(records / "request.json", snapshot)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if folder.exists():
        # Keep all partial records; each new worker starts in a fresh directory.
        backup = root / "interrupted" / job.relative_path / stamp
        backup.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(backup)
    cmd = [sys.executable, "-u", "-m", "bbh_eval.rerun_worker",
           "--task", job.task, "--shots", str(job.shots), "--seed", str(job.seed),
           "--method", job.method, "--budget", str(job.budget), "--steps", str(job.steps),
           "--manifest", str(args.manifest), "--out", str(folder)]
    if job.whole_demo_budget is not None:
        cmd += ["--whole-demo-budget", str(job.whole_demo_budget)]
    if args.source_dir is not None:
        cmd += ["--source-dir", str(args.source_dir)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    log = records / f"{stamp}.log"
    print(f"  output: {log}", flush=True)
    with log.open("x") as sink:
        worker = subprocess.run(cmd, cwd=HERE, stdout=sink, stderr=subprocess.STDOUT,
                                pass_fds=(lock.fileno(),))
    final = job_status(root, job, manifest_hash, task_config, args.limit)
    if worker.returncode or final not in ("complete", "infeasible"):
        # Keep stderr, code and command even when a failed worker created no output directory.
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        verified_oom = False
        if worker.returncode == CUDA_OOM_EXIT:
            with log.open("rb") as src:
                src.seek(max(0, os.fstat(src.fileno()).st_size - 256))
                tail = src.read().splitlines()
            verified_oom = bool(tail) and tail[-1] == CUDA_OOM_MARKER.encode()
        failure = dict(command=cmd, log=str(log), exit_code=worker.returncode, validation=final,
                       kind="cuda_oom" if verified_oom else "worker_error")
        write_json(records / "last-failure.json", failure)
        if verified_oom:
            write_json(folder / "oom.json", failure)
            if args.continue_on_oom:
                print(f"  CUDA OOM. Outputs incomplete; retry this job later. Log: {log}", flush=True)
                return "incomplete"
        raise RuntimeError(f"Worker returned {worker.returncode}, result {final}. Inspect {log}")
    if final == "infeasible":
        print("  infeasible:", load(folder / "metadata.json")["reason"], flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "status", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--design", choices=("shots",), default="shots")
    parser.add_argument("--shot-tasks", choices=("two",), default="two",
                        help="Boolean Expressions and Date Understanding.")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--task", action="append", choices=sorted(TASKS))
    parser.add_argument("--shots", type=int, action="append", choices=SHOTS)
    parser.add_argument("--seed", type=int, action="append", choices=SEEDS)
    parser.add_argument("--limit", type=int, help="Pilot only; stores outputs separately from full tests.")
    parser.add_argument("--max-jobs", type=int, help="Stop after this many additional jobs in this invocation.")
    parser.add_argument("--continue-on-oom", action="store_true",
                        help="Keep CUDA OOM jobs incomplete for retry and continue other jobs. Run exits nonzero if any occurred.")
    a = parser.parse_args()
    if (a.limit is not None and a.limit < 1) or (a.max_jobs is not None and a.max_jobs < 1):
        parser.error("--limit and --max-jobs must be positive.")
    if (a.max_jobs is not None or a.continue_on_oom) and a.action != "run":
        parser.error("--max-jobs and --continue-on-oom are run-only options.")
    a.manifest = a.manifest.resolve()
    a.source_dir = a.source_dir.resolve() if a.source_dir is not None else None
    raw = a.manifest.read_bytes()
    manifest = json.loads(raw)
    manifest_hash = hashlib.sha256(raw).hexdigest()
    root = a.root.resolve()
    if a.limit is not None:
        root = root / "pilots" / f"limit-{a.limit}"
    jobs = [j for j in jobs_for(manifest, a.design, a.shot_tasks)
            if (not a.task or j.task in a.task) and (not a.seed or j.seed in a.seed)
            and (not a.shots or j.shots in a.shots)]
    if not jobs:
        parser.error("No configurations match these design and filter options.")
    print(f"Design {a.design} / shot scope {a.shot_tasks}: {len(jobs)} unique configurations. Root {root}", flush=True)
    if a.action == "run":
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / ".matrix.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, f"Another matrix process holds {root}/.matrix.lock\n")
    else:
        lock = None
    try:
        states = [job_status(root, j, manifest_hash, manifest["tasks"][j.task], a.limit) for j in jobs]
        totals = collections.Counter(states)
        print("  " + ", ".join(f"{s}={totals[s]}" for s in
              ("complete", "infeasible", "pending", "incomplete", "mismatch")), flush=True)
        if a.action == "plan":
            planned_questions = sum(min(len(manifest["tasks"][j.task]["test_ids"]), a.limit)
                                    if a.limit else len(manifest["tasks"][j.task]["test_ids"])
                                    for j in jobs)
            print(f"  Scheduled question predictions before feasibility checks: {planned_questions:,}")
            return
        if a.action == "status":
            for j, state in zip(jobs, states):
                if state in ("incomplete", "infeasible", "mismatch"):
                    path = run_dir(root, j)
                    note = load(path / "metadata.json")["reason"] if state == "infeasible" else ""
                    if state == "incomplete" and (path / "oom.json").is_file():
                        note = f"CUDA OOM, retryable; log {load(path / 'oom.json')['log']}"
                    print(f"  {state}: {j.relative_path} {note}")
            return
        if totals["mismatch"]:
            example = jobs[states.index("mismatch")].relative_path
            raise ValueError(f"Stored run/contract disagrees with the current plan ({example}). Use the matching code and manifest or a fresh root.")
        invoked = 0
        oom_count = 0
        for index, (j, state) in enumerate(zip(jobs, states), 1):
            if state in ("complete", "infeasible"):
                continue
            if a.max_jobs is not None and invoked >= a.max_jobs:
                break
            if (root / ".pause-after-current").exists():
                print("Paused before the next configuration. Use study.sh start to resume.", flush=True)
                break
            print(f"[{index}/{len(jobs)}] {j.relative_path}", flush=True)
            result = run_one(a, root, j, manifest_hash, manifest["tasks"][j.task], lock)
            totals[state] -= 1
            totals[result] += 1
            invoked += 1
            oom_count += (result == "incomplete")
        print(f"Finished invocation: {totals['complete']} complete, {totals['infeasible']} infeasible, "
              f"{len(jobs)-totals['complete']-totals['infeasible']} outstanding.", flush=True)
        if oom_count:
            parser.exit(1, f"{oom_count} CUDA OOM job(s) remain incomplete after this invocation; inspect status before retrying.\n")
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
