#!/bin/bash
# Keep re-running gpu_queue.py on a jobs file until it finishes with no failed jobs (max 20 restarts).
# Run from context-tuning/: bash scripts/supervise_queue.sh JOBS_FILE
JOBS=$1
LOG="${JOBS%.*}.queue.log"
for i in $(seq 1 20); do
    .venv/bin/python scripts/gpu_queue.py "$JOBS" && { echo "[$(date '+%m-%d %H:%M:%S')] SUPERVISOR all done" >> "$LOG"; exit 0; }
    echo "[$(date '+%m-%d %H:%M:%S')] SUPERVISOR queue exited rc=$?, restarting ($i/20)" >> "$LOG"
    sleep 30
done
echo "[$(date '+%m-%d %H:%M:%S')] SUPERVISOR giving up" >> "$LOG"
exit 1
