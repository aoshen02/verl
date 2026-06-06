#!/usr/bin/env bash
# submit_train.sh — single entry point for launching a training run:
#   1. submit the launcher to ray
#   2. auto-stage RUN_DIR with run_dir.txt + submit.log
#   3. fire start_run_monitors.sh (driver.log + gpu_util.txt + prom_live.txt + hang_watchdog)
#   4. print all file paths
#
# Usage:
#   bash submit_train.sh <launcher_path_in_container> [<run_dir_label>]
# Example:
#   bash submit_train.sh /mnt/shared/user/scripts/launcher_iter3_prodfix.sh iter3_prodfix

set -uo pipefail

LAUNCHER="${1:?launcher path inside container required}"
LABEL="${2:-$(basename "$LAUNCHER" .sh)}"

PROJ=/home/user/setup_new_cluster/vllm/projects/semianalysis-swebench-solo
SCRIPTS_DIR="$PROJ/scripts"
TS=$(date -u +%Y-%m-%d_%H%MZ)
RUN_DIR="$PROJ/agent_run/results/training_run_${TS}_${LABEL}"
TRAY02="${TRAY02:-pod4-gb300-3-tray02-f3}"
RAY_ADDR="${RAY_ADDR:-http://10.0.0.13:8265}"

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "$RUN_DIR/run_dir.txt"

# ── Pre-flight: verify CUDA is available in all containers ──────────────────
echo "[submit_train] pre-flight: checking CUDA on all 16 trays..."
TRAYS="01 02 03 04 05 06 07 08 09 11 12 13 14 15 16 17"
BROKEN=""
for t in $TRAYS; do
  RES=$(ssh -o ConnectTimeout=6 -o BatchMode=yes pod4-gb300-3-tray${t}-f3 \
    "sudo docker exec verl-bench python3 -W ignore -c \
     'import torch; print(torch.cuda.is_available())' 2>/dev/null" 2>/dev/null)
  if [ "$RES" != "True" ]; then
    BROKEN="$BROKEN tray$t"
  fi
done
if [ -n "$BROKEN" ]; then
  echo "[submit_train] CUDA unavailable on:$BROKEN — restarting containers..."
  for t in $BROKEN; do
    N=${t#tray}
    ssh -o ConnectTimeout=10 pod4-gb300-3-tray${N}-f3 \
      "sudo docker restart verl-bench" >/dev/null 2>&1 &
  done
  wait
  sleep 8
  # Re-verify
  STILL_BROKEN=""
  for t in $BROKEN; do
    N=${t#tray}
    RES=$(ssh -o ConnectTimeout=10 pod4-gb300-3-tray${N}-f3 \
      "sudo docker exec verl-bench python3 -W ignore -c \
       'import torch; print(torch.cuda.is_available())' 2>/dev/null" 2>/dev/null)
    [ "$RES" != "True" ] && STILL_BROKEN="$STILL_BROKEN $t"
  done
  if [ -n "$STILL_BROKEN" ]; then
    echo "[submit_train] ERROR: CUDA still broken after restart:$STILL_BROKEN — aborting." >&2
    exit 1
  fi
  echo "[submit_train] containers restarted, CUDA verified OK"
else
  echo "[submit_train] pre-flight OK — all trays have CUDA"
fi

# Optionally clean per-rank log dirs from prior runs (opt-in via env)
if [ "${CLEAN_LOG_DIRS:-0}" = "1" ]; then
  ssh "$TRAY02" 'rm -rf /mnt/shared/user/mc_layer_1530Z/* /mnt/shared/user/nccl_logs_1530Z/* 2>/dev/null' || true
fi

# Submit
echo "[submit_train] submitting $LAUNCHER ..."
ssh "$TRAY02" "docker exec verl-bench bash -c 'bash $LAUNCHER 2>&1'" > "$RUN_DIR/submit.log" 2>&1
JOB=$(grep -oE "raysubmit_[a-zA-Z0-9]+" "$RUN_DIR/submit.log" | head -1)
if [ -z "$JOB" ]; then
  echo "[submit_train] ERROR no raysubmit id in submit.log; check $RUN_DIR/submit.log" >&2
  tail -20 "$RUN_DIR/submit.log" >&2
  exit 1
fi
echo "$JOB" > "$RUN_DIR/job_id.txt"
echo "[submit_train] JOB=$JOB"

# Fire monitors
bash "$SCRIPTS_DIR/start_run_monitors.sh" "$RUN_DIR" "$JOB"

cat <<EOF

=== Run paths ===
  RUN_DIR    = $RUN_DIR
  job_id     = $RUN_DIR/job_id.txt   ($JOB)
  driver.log = $RUN_DIR/driver.log
  gpu_util   = $RUN_DIR/gpu_util.txt
  prom_live  = $RUN_DIR/prom_live.txt
  submit.log = $RUN_DIR/submit.log
  monitor pids = $RUN_DIR/monitor_pids.txt

=== Watch live ===
  tail -F $RUN_DIR/driver.log
  watch -n 5 cat $RUN_DIR/gpu_util.txt
EOF
