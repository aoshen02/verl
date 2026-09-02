#!/usr/bin/env bash
# Run the merge=false LoRA gate on a two-node colocated Ray cluster.

set -euo pipefail

: "${SLURM_JOB_ID:?run inside the two-node Slurm allocation}"
if [[ "${SLURM_PROCID:-0}" != 0 ]]; then
    exit 0
fi

WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${IMAGE:-/mnt/ephemeral/aoshen/qwen38/images/qwen38-verl-fsdp-vllm-uv-20260901.sqsh}
TRAIN_MODEL_HEAD=${TRAIN_MODEL_HEAD:?set the h200-0 derived BF16 checkpoint}
TRAIN_MODEL_WORKER=${TRAIN_MODEL_WORKER:?set the h200-1 derived BF16 checkpoint}
ROLLOUT_MODEL=${ROLLOUT_MODEL:?set the official FP8 checkpoint available on both nodes}
DATA=${DATA:?set the frozen dataset available on both nodes}
RUN_DIR=${RUN_DIR:?set a writable h200-0 result directory}
RAY_PORT=${RAY_PORT:-6379}
NETWORK_INTERFACE=${NETWORK_INTERFACE:-ens7}
TRITON_CACHE_ROOT=${TRITON_CACHE_ROOT:-/mnt/ephemeral/aoshen/qwen38/cache/triton}
RAY_MEMORY_USAGE_THRESHOLD=${RAY_MEMORY_USAGE_THRESHOLD:-0.98}

allocation_nodes=$(scontrol show job -o "$SLURM_JOB_ID" |
    sed -n 's/.* NodeList=\([^ ]*\).*/\1/p')
mapfile -t NODES < <(scontrol show hostnames "$allocation_nodes")
if ((${#NODES[@]} != 2)); then
    echo "expected exactly two Slurm nodes, got ${#NODES[@]}" >&2
    exit 2
fi
HEAD_NODE=${NODES[0]}
WORKER_NODE=${NODES[1]}

node_ipv4() {
    srun --overlap --nodes=1 --ntasks=1 -w "$1" /bin/sh -c \
        "ip -4 -o addr show dev '$NETWORK_INTERFACE' scope global | awk 'NR == 1 {split(\$4, ip, \"/\"); print ip[1]}'"
}

head_ip=$(node_ipv4 "$HEAD_NODE")
if [[ -z "$head_ip" ]]; then
    echo "failed to resolve the Ray head address" >&2
    exit 2
fi
RAY_ADDRESS=${head_ip}:${RAY_PORT}

export ENROOT_CACHE_PATH=/mnt/ephemeral/aoshen/enroot/cache
export ENROOT_DATA_PATH=/mnt/ephemeral/aoshen/enroot/data
export ENROOT_RUNTIME_PATH=/mnt/ephemeral/aoshen/enroot/runtime
export ENROOT_TEMP_PATH=/mnt/ephemeral/aoshen/enroot/tmp
mkdir -p "$RUN_DIR"

ray_container=(
    enroot start --rc "$WORKTREE/scripts/enroot_exec.sh"
    -e UV_PROJECT_ENVIRONMENT=/opt/verl-uv-final
    -e XDG_CACHE_HOME=/run/xdg-cache
    -e TRITON_CACHE_DIR=/var/tmp
    -e VLLM_DO_NOT_TRACK=1
    -e PYTHONPATH=/workspace
    -e VERL_ALLOW_UNSUPPORTED_VLLM_VERSION=1
    -e "RAY_memory_usage_threshold=$RAY_MEMORY_USAGE_THRESHOLD"
    -e FLASHINFER_WORKSPACE_BASE=/run/flashinfer
    -e "GLOO_SOCKET_IFNAME=$NETWORK_INTERFACE"
    -e "NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE"
    -m "$WORKTREE:/workspace:none:bind,ro"
)

start_ray() {
    local node=$1 node_run=$2 train_model=$3
    shift 3
    local triton_cache="$TRITON_CACHE_ROOT/$node"
    srun --overlap --nodes=1 --ntasks=1 -w "$node" \
        mkdir -p "$node_run" "$triton_cache"
    srun --overlap --nodes=1 --ntasks=1 -w "$node" \
        --output="$node_run/ray.log" --error="$node_run/ray.log" \
        "${ray_container[@]}" \
        -m "$train_model:/models/q0:none:bind,ro" \
        -m "$ROLLOUT_MODEL:/models/qwen38:none:bind,ro" \
        -m "$DATA:/opt/data:none:bind,ro" \
        -m "$node_run:/run:none:bind,rw" \
        -m "$node_run:/tmp:none:bind,rw" \
        -m "$triton_cache:/var/tmp:none:bind,rw" \
        "$IMAGE" /opt/verl-uv-final/bin/ray start "$@" \
        --temp-dir=/tmp/ray --num-gpus=8 --block &
    RAY_PID=$!
}

head_pid=
worker_pid=
stop_ray() {
    local node=$1 node_run=$2 train_model=$3
    local triton_cache="$TRITON_CACHE_ROOT/$node"
    srun --overlap --nodes=1 --ntasks=1 -w "$node" \
        "${ray_container[@]}" \
        -m "$train_model:/models/q0:none:bind,ro" \
        -m "$ROLLOUT_MODEL:/models/qwen38:none:bind,ro" \
        -m "$DATA:/opt/data:none:bind,ro" \
        -m "$node_run:/run:none:bind,rw" \
        -m "$node_run:/tmp:none:bind,rw" \
        -m "$triton_cache:/var/tmp:none:bind,rw" \
        "$IMAGE" /opt/verl-uv-final/bin/ray stop --force || true
}
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$worker_pid" ]]; then
        stop_ray "$WORKER_NODE" "$RUN_DIR/ray-worker" "$TRAIN_MODEL_WORKER"
        wait "$worker_pid" 2>/dev/null || true
    fi
    if [[ -n "$head_pid" ]]; then
        stop_ray "$HEAD_NODE" "$RUN_DIR/ray-head" "$TRAIN_MODEL_HEAD"
        wait "$head_pid" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

ray_status() {
    local triton_cache="$TRITON_CACHE_ROOT/$HEAD_NODE"
    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        "${ray_container[@]}" \
        -m "$TRAIN_MODEL_HEAD:/models/q0:none:bind,ro" \
        -m "$ROLLOUT_MODEL:/models/qwen38:none:bind,ro" \
        -m "$DATA:/opt/data:none:bind,ro" \
        -m "$RUN_DIR/ray-head:/run:none:bind,rw" \
        -m "$RUN_DIR/ray-head:/tmp:none:bind,rw" \
        -m "$triton_cache:/var/tmp:none:bind,rw" \
        "$IMAGE" /opt/verl-uv-final/bin/ray status --address="$RAY_ADDRESS" \
        >"$RUN_DIR/ray-status.log" 2>&1
}
wait_for_ray() {
    local pattern=$1
    for _ in $(seq 1 30); do
        if ray_status && grep -q "$pattern" "$RUN_DIR/ray-status.log"; then
            return
        fi
        sleep 2
    done
    cat "$RUN_DIR/ray-status.log" >&2
    return 1
}

start_ray "$HEAD_NODE" "$RUN_DIR/ray-head" "$TRAIN_MODEL_HEAD" \
    --head --node-ip-address="$head_ip" --port="$RAY_PORT"
head_pid=$RAY_PID
wait_for_ray 'Resources'
worker_ip=$(node_ipv4 "$WORKER_NODE")
if [[ -z "$worker_ip" ]]; then
    echo "failed to resolve the Ray worker address" >&2
    exit 2
fi
start_ray "$WORKER_NODE" "$RUN_DIR/ray-worker" "$TRAIN_MODEL_WORKER" \
    --address="$RAY_ADDRESS" --node-ip-address="$worker_ip"
worker_pid=$RAY_PID
wait_for_ray '/16\.0 GPU'

driver_run="$RUN_DIR/driver"
ray_head_run="$RUN_DIR/ray-head"
RAY_ADDRESS="$RAY_ADDRESS" NNODES=2 NGPUS_PER_NODE=8 \
RUN_DIR="$driver_run" IMAGE="$IMAGE" \
TRAIN_MODEL="$TRAIN_MODEL_HEAD" ROLLOUT_MODEL="$ROLLOUT_MODEL" DATA="$DATA" \
RAY_TMPDIR_HOST="$ray_head_run" NETWORK_INTERFACE="$NETWORK_INTERFACE" \
TRITON_CACHE_DIR_HOST="$TRITON_CACHE_ROOT/$HEAD_NODE" \
ENABLE_MTP="${ENABLE_MTP:-true}" ENFORCE_EAGER="${ENFORCE_EAGER:-false}" \
SMOKE_STEPS="${SMOKE_STEPS:-4}" TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}" \
EXPECTED_TRAIN_ROWS="${EXPECTED_TRAIN_ROWS:-32}" \
EXPECTED_VAL_ROWS="${EXPECTED_VAL_ROWS:-8}" \
ROLLOUT_TP="${ROLLOUT_TP:-2}" \
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}" \
PROJECT_NAME="${PROJECT_NAME:-qwen38_full_rl_smoke}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fsdp16_lora_adapter_only_mtp_cudagraph}" \
DRY_RUN="${DRY_RUN:-0}" \
    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    "$WORKTREE/scripts/run_qwen_l8_verl_smoke_container.sh"
