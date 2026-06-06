#!/usr/bin/env bash
# start_run_monitors.sh — monitors for a training run:
#   1. driver.log   — 30s polling snapshot from ray job logs
#   2. gpu_util.txt — 30s snapshot, TRAIN trays only, appends history
#   3. hang_watchdog — auto py-spy on GPU stall (remote)
#
# Usage: bash start_run_monitors.sh <RUN_DIR> <JOB_ID>

set -uo pipefail

RUN_DIR="${1:?RUN_DIR required}"
JOB="${2:?JOB required}"
RAY_ADDR="${RAY_ADDR:-http://10.0.0.13:8265}"
HEAD="${HEAD:-pod4-gb300-3-tray01-f3}"   # stable SSH hop
GPU_UTIL_SCRIPT="${GPU_UTIL_SCRIPT:-$(dirname "$0")/fleet_gpu_snapshot.sh}"
WATCHDOG_SCRIPT="${WATCHDOG_SCRIPT:-$(dirname "$0")/hang_watchdog.sh}"
INTERVAL_S="${INTERVAL_S:-30}"

mkdir -p "$RUN_DIR"
echo "$JOB" > "$RUN_DIR/job_id.txt"
PIDFILE="$RUN_DIR/monitor_pids.txt"
: > "$PIDFILE"

# ── 1. driver.log ────────────────────────────────────────────────────────────
# Full snapshot each cycle — --follow stalls silently via ssh after ~10 min.
(
  while true; do
    ssh "$HEAD" "docker exec verl-bench ray job logs --address $RAY_ADDR $JOB" \
      > "$RUN_DIR/driver.log.new" 2>/dev/null \
      && mv "$RUN_DIR/driver.log.new" "$RUN_DIR/driver.log"
    sleep "$INTERVAL_S"
  done
) &
echo "stream:$!" >> "$PIDFILE"

# ── 2. gpu_util.txt (TRAIN only) ─────────────────────────────────────────────
(
  while true; do
    bash "$GPU_UTIL_SCRIPT" "$RUN_DIR/gpu_util.txt" 2>/dev/null
    cat "$RUN_DIR/gpu_util.txt" >> "$RUN_DIR/gpu_util.history" 2>/dev/null
    STATUS=$(ssh -o ConnectTimeout=5 "$HEAD" \
      "docker exec verl-bench ray job status --address $RAY_ADDR $JOB 2>&1 | grep -oE 'JobStatus\.[A-Z]+'" 2>/dev/null)
    if [ "$STATUS" = "JobStatus.SUCCEEDED" ] || \
       [ "$STATUS" = "JobStatus.FAILED" ]    || \
       [ "$STATUS" = "JobStatus.STOPPED" ]; then exit 0; fi
    sleep "$INTERVAL_S"
  done
) &
echo "gpu_util:$!" >> "$PIDFILE"

# ── 3. hang_watchdog (remote, fire-and-forget) ───────────────────────────────
TAG="$(basename "$RUN_DIR")"
ssh -o ConnectTimeout=10 "$HEAD" \
  "nohup bash -c 'RUN_TAG=$TAG bash $WATCHDOG_SCRIPT' \
   > /mnt/shared/user/pyspy_dumps/hang_watchdog_${TAG}.log 2>&1 &" 2>&1
echo "hang_watchdog:remote_$HEAD" >> "$PIDFILE"

echo "Monitors started for $JOB"
echo "  RUN_DIR=$RUN_DIR"
echo "  driver.log   — every ${INTERVAL_S}s (full snapshot)"
echo "  gpu_util.txt — every ${INTERVAL_S}s (TRAIN trays only)"
echo "  hang_watchdog — remote on $HEAD"
echo "  pids → $PIDFILE"
