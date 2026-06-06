#!/usr/bin/env bash
# Submit a training job AND start the driver.log + gpu_util.history sidecars.
#
# The base script `train_qwen3_235b_prod_verda.sh` only submits a Ray job; the
# observability files (driver.log tail piped to NFS, periodic GPU snapshots)
# used to be wired up by ad-hoc `/tmp/launcher_*.sh` wrappers that vanished
# whenever the verl-bench container was restarted. This script makes them
# durable.
#
# Usage:
#   bash scripts/launch_with_monitors.sh \
#     [--exp-name <name>] [--head-node <hostname>] \
#     -- <env=VAL ... base-script args>
#
# Example:
#   EXP_NAME=qwen3_235b_smoke_pp3 \
#   ACTOR_PP=3 ACTOR_EP=16 NNODES_TRAIN=12 NNODES_ROLLOUT=4 \
#     bash scripts/launch_with_monitors.sh -- \
#       actor_rollout_ref.rollout.n=8 \
#       actor_rollout_ref.actor.ppo_mini_batch_size=4 \
#       trainer.resume_mode=disable \
#       +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=31 \
#       +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=31
#
# Outputs (in $RUN_DIR):
#   submit.log          full stdout/stderr of the ray job submit call
#   job_id.txt          one-line ray submission id
#   run_dir.txt         echo of $RUN_DIR (useful for downstream tooling)
#   driver.log          full `ray job logs` snapshot, atomically refreshed every 20 s
#                       (NOT --follow; see feedback_no_ssh_follow_for_driver_log.md)
#   gpu_util.history    one train-only snapshot every 30 s, matches c96 format
#   monitor_pids.txt    PIDs of the two background sidecars (kill these to stop)
#
# Cleanup: `kill $(grep -oE '[0-9]+' .../monitor_pids.txt)` stops both sidecars.
# Job lifetime is independent — sidecars only observe, they do not own the job.

set -euo pipefail

HEAD_NODE="${HEAD_NODE:-pod4-gb300-3-tray02-f3}"
BASE_SCRIPT_IN_CONTAINER="/mnt/shared/user/scripts/train_qwen3_235b_prod_verda.sh"
PROJECT_ROOT="/home/user/setup_new_cluster/vllm/projects/semianalysis-swebench-solo"

# ---- parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-name) EXP_NAME="$2"; shift 2;;
    --head-node) HEAD_NODE="$2"; shift 2;;
    --) shift; break;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

EXP_NAME="${EXP_NAME:-untagged_$(date -u +%H%M)}"
TS_UTC="$(date -u +%Y-%m-%d_%H%M)Z"
RUN_DIR="${PROJECT_ROOT}/agent_run/results/training_run_${TS_UTC}_${EXP_NAME}"
mkdir -p "$RUN_DIR"

echo "[wrapper] RUN_DIR=$RUN_DIR"
echo "$RUN_DIR" > "$RUN_DIR/run_dir.txt"

# ---- 1. submit the ray job via verl-bench container on head node ----
# Pass through all env vars the base script understands. The user's shell-level
# env (EXP_NAME, ACTOR_*, NNODES_*, RUNTIME_ENV, OFFLOAD_FRACTION, ...) reaches
# the base script via the `env -S` envelope below.
ENV_PASSTHROUGH=$(env | grep -E '^(EXP_NAME|ACTOR_|NNODES_|RUNTIME_ENV|OFFLOAD_FRACTION|SAVE_FREQ|TEST_FREQ|MODEL_PATH|TRAIN_FILE|TEST_FILE|RAY_DATA_HOME|SOLO_ROOT|AGENT_CONFIG_PATH|VAL_BEFORE_TRAIN|INFER_TP|INFER_EP|INFER_MEM_UTILIZATION|CKPTS_DIR|MAX_CKPT_TO_KEEP|ENABLE_PROMETHEUS_MONITORING|PROMETHEUS_PORT|PROMETHEUS_CONFIG_FILE|PROMETHEUS_SERVED_MODEL_NAME|PROJECT_NAME)=' || true)
EXP_NAME="$EXP_NAME"  # ensure it is exported even when wrapper was given the flag form
export EXP_NAME

# Build the inner command string. We use `env` to inject explicit overrides
# without expanding $@ on the remote side.
HYDRA_ARGS_ESCAPED=$(printf "%q " "$@")

REMOTE_CMD="bash -lc 'cd /mnt/shared/user && $ENV_PASSTHROUGH EXP_NAME=$EXP_NAME bash $BASE_SCRIPT_IN_CONTAINER $HYDRA_ARGS_ESCAPED'"

echo "[wrapper] submitting via $HEAD_NODE (verl-bench)..."
ssh -o StrictHostKeyChecking=no "$HEAD_NODE" "sudo docker exec verl-bench $REMOTE_CMD" > "$RUN_DIR/submit.log" 2>&1 || {
  echo "[wrapper] FATAL: submit failed; check $RUN_DIR/submit.log" >&2
  tail -20 "$RUN_DIR/submit.log" >&2
  exit 1
}

# ---- 2. parse submission id ----
JOB_ID=$(grep -oE "raysubmit_[A-Za-z0-9]+" "$RUN_DIR/submit.log" | head -1)
if [[ -z "$JOB_ID" ]]; then
  echo "[wrapper] FATAL: could not parse submission_id from submit.log" >&2
  tail -20 "$RUN_DIR/submit.log" >&2
  exit 1
fi
echo "$JOB_ID" > "$RUN_DIR/job_id.txt"
echo "[wrapper] job_id=$JOB_ID"

# ---- 3. background driver.log mirror via POLL-OVERWRITE (never --follow) ----
# WHY: `ssh ... docker exec ... ray job logs --follow <id>` silently buffers
# through three layers (ssh, docker exec, ray cli) and goes stale for many
# minutes without warning. The job keeps running but the local file freezes,
# producing false "hang" reports. See:
#   ~/.claude/projects/-home-.../memory/feedback_no_ssh_follow_for_driver_log.md
# Verified harm: 2026-05-23 smoke wasted 20 min user time.
#
# Pattern: every 20s, fetch the full current tail (no --follow) and atomically
# replace driver.log. Slightly higher cluster load but eliminates staleness.
DRIVER_POLL_SECS=${DRIVER_POLL_SECS:-20}
nohup bash -c "while true; do
  ssh -o StrictHostKeyChecking=no '$HEAD_NODE' \
    \"sudo docker exec verl-bench ray job logs $JOB_ID 2>&1\" \
    > '$RUN_DIR/driver.log.tmp' && mv '$RUN_DIR/driver.log.tmp' '$RUN_DIR/driver.log'
  sleep $DRIVER_POLL_SECS
done" > /dev/null 2>&1 &
DRIVER_PID=$!
echo "[wrapper] driver.log poll-overwrite PID=$DRIVER_PID (every ${DRIVER_POLL_SECS}s)"

# ---- 4. background GPU snapshot loop (every 30 s, train-only, matches c96 format) ----
# Uses scripts/train_gpu_snapshot.sh which reads driver.log for the WorkerDict
# IPs and only probes those trays (no rollout side, no idle trays).
GPU_SCRIPT="$PROJECT_ROOT/scripts/train_gpu_snapshot.sh"
nohup bash -c "while true; do bash '$GPU_SCRIPT' '$RUN_DIR/driver.log' >> '$RUN_DIR/gpu_util.history' 2>&1; sleep 30; done" \
  > /dev/null 2>&1 &
GPU_PID=$!
echo "[wrapper] gpu_util.history loop PID=$GPU_PID  (every 30s, train-only)"

# ---- 5. record PIDs for cleanup ----
cat > "$RUN_DIR/monitor_pids.txt" <<EOF
driver_pid=$DRIVER_PID
driver_mode=poll-overwrite-${DRIVER_POLL_SECS}s
gpu_pid=$GPU_PID
gpu_script=$GPU_SCRIPT
EOF

# ---- 6. summary ----
cat <<EOF

[wrapper] ===== launched =====
  run_dir   : $RUN_DIR
  job_id    : $JOB_ID
  driver.log: tail -F $RUN_DIR/driver.log
  gpu_util  : tail -F $RUN_DIR/gpu_util.history
  to stop sidecars: kill $DRIVER_PID $GPU_PID

EOF
