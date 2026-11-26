"""Serial GPU queue for the CT-KV accuracy study.

The plan file has one step per line: `<name> <shell command>`. Lines starting with # are ignored. The file is
re-read after every step, so steps can be appended while the queue runs. A step is done once it exits 0; its
name is then appended to <plan>.done. A step that exits nonzero is retried immediately (the runners skip
finished jobs, so a retry resumes where it stopped), up to MAX_ATTEMPTS times; then the queue moves on and
records it in <plan>.failed. When nothing is left the queue waits for new lines (poll every 60 s) and exits
after IDLE_EXIT_S without new work.

Log lines go to <plan>.log: START / DONE / FAIL rc=... / RETRY / GIVEUP / IDLE / EXIT.
Usage: python scripts/study_queue.py PLAN
"""
import os
import subprocess
import sys
import time

MAX_ATTEMPTS = 5
IDLE_EXIT_S = 6 * 3600


def main():
    plan = sys.argv[1]
    log = open(plan + ".log", "a")

    def say(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    def names(path):
        return set(open(path).read().split()) if os.path.exists(path) else set()

    attempts = {}
    idle_since = None
    say(f"queue start pid={os.getpid()}")
    while True:
        done, failed = names(plan + ".done"), names(plan + ".failed")
        steps = []
        for line in open(plan):
            line = line.strip()
            if line and not line.startswith("#"):
                name, cmd = line.split(None, 1)
                steps.append((name, cmd))
        todo = [(n, c) for n, c in steps if n not in done and n not in failed]
        if not todo:
            if idle_since is None:
                idle_since = time.time()
                say("IDLE: nothing left; waiting for new steps")
            if time.time() - idle_since > IDLE_EXIT_S:
                say("EXIT after idle timeout")
                return
            time.sleep(60)
            continue
        idle_since = None
        name, cmd = todo[0]
        attempts[name] = attempts.get(name, 0) + 1
        say(f"START {name} (attempt {attempts[name]}): {cmd}")
        t = time.time()
        step_log = plan + f".{name}.out"
        with open(step_log, "a") as out:
            out.write(f"\n===== attempt {attempts[name]} {time.ctime()} =====\n")
            out.flush()
            rc = subprocess.call(cmd, shell=True, stdout=out, stderr=subprocess.STDOUT)
        if rc == 0:
            say(f"DONE {name} ({(time.time() - t) / 60:.1f} min)")
            with open(plan + ".done", "a") as f:
                f.write(name + "\n")
            continue
        say(f"FAIL {name} rc={rc} after {(time.time() - t) / 60:.1f} min; see {step_log}")
        if attempts[name] < MAX_ATTEMPTS:
            say(f"RETRY {name} now")
            time.sleep(10)
        else:
            say(f"GIVEUP {name}")
            with open(plan + ".failed", "a") as f:
                f.write(name + "\n")


if __name__ == "__main__":
    main()
