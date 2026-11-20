"""Run experiment commands one by one, pausing them when other processes need GPU memory.

Usage: python scripts/gpu_queue.py JOBS_FILE [--log LOG]

The jobs file has one job per line, in the form `<output_json> <command ...>`. A job whose
output_json already exists is skipped, so the queue can be restarted after a crash.

The watchdog polls nvidia-smi every POLL seconds. "Pressure" means free GPU memory drops below
PAUSE_FREE_MB while other processes (anything outside this queue's job) hold more than
OTHERS_MIN_MB. When that happens, the job's process group gets SIGSTOP. It gets SIGCONT once free
memory is back above RESUME_FREE_MB. Pausing does not release the memory the job already holds.
So if the pause lasts more than KILL_AFTER_S, the job is killed to free its memory and goes back
on the queue. It restarts only after free memory reaches START_FREE_MB.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

POLL = 5
PAUSE_FREE_MB = 2500
RESUME_FREE_MB = 5000
OTHERS_MIN_MB = 1500
KILL_AFTER_S = 600
START_FREE_MB = 24000
MAX_ATTEMPTS = 4


def log(msg, fh):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def gpu_state():
    """Return (used_mb, total_mb, {pid: used_mb}) for GPU 0."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    used, total = (int(x) for x in out.split(","))
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    per_pid = {}
    for a in apps:
        if a.strip():
            pid, mem = (x.strip() for x in a.split(","))
            per_pid[int(pid)] = int(mem)
    return used, total, per_pid


def descendants(root_pid):
    """Return root_pid and all its descendant pids."""
    pids = {root_pid}
    try:
        out = subprocess.run(["ps", "-e", "-o", "pid=,ppid="], capture_output=True, text=True).stdout
    except Exception:
        return pids
    children = {}
    for line in out.splitlines():
        p, pp = (int(x) for x in line.split())
        children.setdefault(pp, []).append(p)
    stack = [root_pid]
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in pids:
                pids.add(c)
                stack.append(c)
    return pids


def wait_for_free(min_free, fh):
    warned = False
    while True:
        used, total, _ = gpu_state()
        if total - used >= min_free:
            return
        if not warned:
            log(f"waiting for {min_free} MB free before starting (free {total - used} MB)", fh)
            warned = True
        time.sleep(POLL * 6)


def run_job(out_json, cmd, job_log, fh):
    """Run one job under the watchdog. Return 'done', 'failed' or 'requeue'."""
    with open(job_log, "w") as lf:
        proc = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    paused_since = None
    while True:
        rc = proc.poll()
        if rc is not None:
            if rc == 0 and os.path.exists(out_json):
                return "done"
            log(f"job exited rc={rc} (output exists={os.path.exists(out_json)})", fh)
            return "failed"
        try:
            used, total, per_pid = gpu_state()
        except Exception as e:  # nvidia-smi hiccup
            log(f"nvidia-smi error: {e}", fh)
            time.sleep(POLL)
            continue
        ours = sum(per_pid.get(p, 0) for p in descendants(proc.pid))
        others = used - ours
        free = total - used
        if paused_since is None and free < PAUSE_FREE_MB and others > OTHERS_MIN_MB:
            os.killpg(proc.pid, signal.SIGSTOP)
            paused_since = time.time()
            log(f"PAUSE: free={free} MB, ours={ours} MB, others={others} MB", fh)
        elif paused_since is not None:
            if free >= RESUME_FREE_MB or others <= OTHERS_MIN_MB:
                os.killpg(proc.pid, signal.SIGCONT)
                log(f"RESUME after {time.time() - paused_since:.0f}s: free={free} MB, others={others} MB", fh)
                paused_since = None
            elif time.time() - paused_since > KILL_AFTER_S:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                log(f"KILL after {KILL_AFTER_S}s paused to release {ours} MB; job requeued", fh)
                return "requeue"
        time.sleep(POLL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_file")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    log_path = args.log or os.path.splitext(args.jobs_file)[0] + ".queue.log"
    fh = open(log_path, "a")

    jobs = []
    for line in open(args.jobs_file):
        line = line.strip()
        if line and not line.startswith("#"):
            out_json, cmd = line.split(None, 1)
            jobs.append((out_json, cmd))
    log(f"queue start: {len(jobs)} jobs from {args.jobs_file}", fh)

    pending = list(jobs)
    failed = []
    attempts = {}
    while pending:
        out_json, cmd = pending.pop(0)
        attempts[out_json] = attempts.get(out_json, 0) + 1
        if os.path.exists(out_json):
            continue
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        wait_for_free(START_FREE_MB, fh)
        job_log = os.path.splitext(out_json)[0] + ".log"
        log(f"START {out_json}", fh)
        t0 = time.time()
        status = run_job(out_json, cmd, job_log, fh)
        log(f"{status.upper()} {out_json} ({(time.time() - t0) / 60:.1f} min)", fh)
        if status == "requeue":
            pending.insert(0, (out_json, cmd))
        elif status == "failed":
            if attempts[out_json] < MAX_ATTEMPTS:
                # often an OOM caused by another process; retry at the end of the queue
                log(f"retry {out_json} later (attempt {attempts[out_json]}/{MAX_ATTEMPTS})", fh)
                os.replace(job_log, job_log + f".fail{attempts[out_json]}")
                pending.append((out_json, cmd))
            else:
                failed.append(out_json)
    log(f"queue finished; failed={failed}", fh)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
