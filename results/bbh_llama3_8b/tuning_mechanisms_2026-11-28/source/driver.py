"""Run one GPU job at a time. Publish a result only after checking its files."""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import design

KIT = Path(__file__).resolve().parent
ROOT = KIT / "results"
STAGES = ("all", "pilot", "main", "steps", "params", "random", "controls")
PAYLOADS = ("metrics.json", "predictions.jsonl", "validation.json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load(path):
    return json.loads(path.read_text())


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        f.write(encoded(obj) + b"\n")
        tmp = Path(f.name)
    tmp.replace(path)


def source_id():
    names = [KIT / x for x in ("driver.py", "worker.py", "design.py", "study.sh", "runtime.py") if (KIT / x).exists()]
    names += sorted((KIT / "bbh_eval").glob("*.py"))
    names += sorted((KIT / "data").rglob("*.json"))
    return digest(encoded({str(p.relative_to(KIT)): digest(p.read_bytes()) for p in names}))


def reference_needed(stage, job):
    if job["stage"] != "controls":
        return False
    if stage == "steps":
        return job["method"] == "full" and job["shots"] == 10
    if stage == "params":
        return job["method"] == "full"
    if stage == "random":
        return job["method"] == "answer_attention"
    return False


def select(jobs, stage):
    if stage == "pilot":
        return [j for j in jobs if j["stage"] == "main" and j["demo_seed"] == 0
                and j["shots"] == 10 and j["ratio"] == 0.5]
    return [j for j in jobs if stage == "all" or j["stage"] == stage or reference_needed(stage, j)]


@contextmanager
def exclusive():
    ROOT.mkdir(exist_ok=True)
    with (ROOT / ".driver.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another driver or archive holds the study lock. Run status.")
        yield lock.fileno()


def active():
    if not ROOT.is_dir():
        return False
    with (ROOT / ".driver.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def signature(job, source):
    return digest(encoded([job, source]))


def verify_output(path, job):
    meta = load(path / "metrics.json")
    predictions = [json.loads(s) for s in (path / "predictions.jsonl").read_text().splitlines()]
    if [p["row_id"] for p in predictions] != job["test_ids"]:
        raise ValueError("Held-out prediction row count, order, or identity differs from plan.")
    if meta["n_correct"] != sum(p["exact_match"] for p in predictions) or meta["n_scored"] != len(predictions):
        raise ValueError("Accuracy counts disagree with question predictions.")
    validation = load(path / "validation.json")
    expected = ({0, 2, 4, 8, 16, job["updates"]} if job["updates"] else set())
    expected = sorted(n for n in expected if n <= job["updates"])
    if [v["step"] for v in validation] != expected or any(v["n"] != len(job["val_ids"]) for v in validation):
        raise ValueError("Missing or mismatched validation checkpoints.")
    return {f: digest((path/f).read_bytes()) for f in PAYLOADS}


def completed(path, job, source):
    done = path / "DONE.json"
    if not done.exists():
        if path.exists():
            raise ValueError(f"Unrecognized result directory {path}. Move it aside after inspection.")
        return False
    stamp = load(done)
    if stamp["signature"] != signature(job, source) or stamp["job"] != job:
        raise ValueError(f"Protocol mismatch at {path}. Use a fresh kit directory for a different protocol.")
    if any(stamp["files"][f] != digest((path/f).read_bytes()) for f in PAYLOADS):
        raise ValueError(f"A completed output was modified in {path}. Check your backup.")
    return True


def run_one(job, model, source, lock_fd, smoke=False):
    dest = ROOT / ("smoke" if smoke else "runs") / job["id"]
    if completed(dest, job, source):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    area = ROOT / "attempts"
    area.mkdir(exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=job["id"]+"-", dir=area))
    save(attempt / "job.json", job)
    save(ROOT / ".current.json", {"job":job["id"],"start":time.time(),"attempt":str(attempt.relative_to(ROOT))})
    print(f"Starting {job['id']} | log {attempt / 'worker.log'}", flush=True)
    start = time.monotonic()
    with (attempt / "worker.log").open("w") as log:
        proc = subprocess.run([sys.executable,"-u",str(KIT/"worker.py"),"--job",str(attempt/"job.json"),
                               "--out",str(attempt/"output"),"--model",str(model)],stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock_fd,))
    if proc.returncode:
        tail = (attempt/"worker.log").read_text(errors="replace").splitlines()[-12:]
        raise RuntimeError(f"Worker exited with {proc.returncode}: {attempt}\n" + "\n".join(tail))
    out = attempt / "output"
    sums = verify_output(out, job)
    shutil.move(str(attempt/"worker.log"),str(out/"worker.log"))
    save(out/"DONE.json", {"job":job,"signature":signature(job,source),"files":sums,
                           "wall_seconds":round(time.monotonic()-start,3)})
    out.rename(dest)
    (attempt/"job.json").unlink()
    attempt.rmdir()
    return True


def small_job(jobs):
    choice = next(j for j in jobs if j["stage"] == "main" and j["method"] == "answer_attention")
    return {**choice,"smoke":True,"id":"boolean-attention-smoke","updates":2,
            "test_ids":choice["test_ids"][:4],"val_ids":choice["val_ids"][:2]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("plan","status","run","smoke","stop"):
        p = sub.add_parser(action)
        if action in {"run","status","plan"}:
            p.add_argument("stage", choices=STAGES, nargs="?", default="all")
        if action in {"smoke","run"}:
            p.add_argument("--model", type=Path, required=True)
    a = parser.parse_args()
    if a.action == "stop":
        if active():
            (ROOT/".stop-requested").touch()
            print("Pause requested. The driver will finish and save its current job. Check status before archiving.")
        else:
            print("No running driver.")
        return
    jobs = design.build_jobs()
    if len(jobs)!=180 or len({j['id'] for j in jobs})!=180:
        raise ValueError("Expected 180 unique jobs.")
    picked = select(jobs, getattr(a,"stage","all"))
    if a.action == "plan":
        print(f"{len(picked)} unique formal jobs: {dict(Counter(j['stage'] for j in picked))}")
        print(f"Held-out predictions: {sum(len(j['test_ids']) for j in picked):,}. Checkpoint validation is extra.")
        return
    source = source_id()
    if a.action == "status":
        done = [j for j in picked if completed(ROOT/"runs"/j["id"], j, source)]
        busy = active()
        print(f"{len(done)}/{len(picked)} complete; {len(picked)-len(done)} remaining; driver {'ACTIVE' if busy else 'idle'}.")
        if done:
            walls = [load(ROOT/"runs"/j["id"]/"DONE.json")["wall_seconds"] for j in done]
            print(f"Completed jobs: {sum(walls)/60:.1f} worker minutes, {sum(walls)/len(walls):.1f} s/job in this selection (setup excluded).")
        if busy and (ROOT/".current.json").exists():
            try:
                print("Current job:",load(ROOT/".current.json")["job"])
            except FileNotFoundError:
                pass
        if busy and (ROOT/".stop-requested").exists():
            print("Pause requested.")
        return
    with exclusive() as lock_fd:
        stop = ROOT/".stop-requested"
        stop.unlink(missing_ok=True)
        if not a.model.is_dir():
            raise ValueError("Local snapshot missing; run study.sh access.")
        plan_path = ROOT/"plan.json"
        plan = {"source_sha256":source,"jobs":jobs}
        if plan_path.exists() and load(plan_path)!=plan:
            raise ValueError("Saved plan or sources differ; extract a fresh directory.")
        if not plan_path.exists():
            save(plan_path,plan)
        queue = [small_job(jobs)] if a.action=="smoke" else picked
        try:
            for i,job in enumerate(queue,1):
                if stop.exists():
                    print("Paused. Repeat start to resume.", flush=True)
                    break
                if run_one(job,a.model.resolve(),source,lock_fd,a.action=="smoke"):
                    print(f"[{i}/{len(queue)}] saved {job['id']}",flush=True)
            else:
                print(f"Verified all {len(queue)} jobs in this selection.",flush=True)
        finally:
            (ROOT/".current.json").unlink(missing_ok=True)


if __name__=="__main__":
    main()
