#!/usr/bin/env bash
# Run the L8 RL smoke inside the immutable specialized-image uv profile.

set -euo pipefail

WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_DIR=${RUN_DIR:-/mnt/ephemeral/aoshen/qwen38/run/l8-e16-fsdp8-dapo-rl-20260901}
IMAGE=${IMAGE:-/mnt/ephemeral/aoshen/qwen38/images/qwen38-verl-fsdp-vllm-uv-20260901.sqsh}
TRAIN_MODEL=${TRAIN_MODEL:-/mnt/ephemeral/aoshen/qwen38/derived/qwen38-flash-next-bf16-trainer-smoke-l8-e16-mtp-ple2048-p8-v1}
ROLLOUT_MODEL=${ROLLOUT_MODEL:-/mnt/ephemeral/aoshen/qwen38/derived/qwen38-flash-next-fp8-serving-smoke-l8-e16-mtp-ple2048-p8-v1}
DATA=${DATA:-/mnt/ephemeral/aoshen/qwen38/data/dapo-math-17k-smoke-l8-32x8-20260831}
TRITON_CACHE_DIR_HOST=${TRITON_CACHE_DIR_HOST:-/mnt/ephemeral/aoshen/qwen38/cache/triton}
RAY_TMPDIR_HOST=${RAY_TMPDIR_HOST:-}
NETWORK_INTERFACE=${NETWORK_INTERFACE:-}

mkdir -p "$RUN_DIR" "$TRITON_CACHE_DIR_HOST"
export ENROOT_CACHE_PATH=/mnt/ephemeral/aoshen/enroot/cache
export ENROOT_DATA_PATH=/mnt/ephemeral/aoshen/enroot/data
export ENROOT_RUNTIME_PATH=/mnt/ephemeral/aoshen/enroot/runtime
export ENROOT_TEMP_PATH=/mnt/ephemeral/aoshen/enroot/tmp

ray_tmpdir_mount=()
if [[ -n "$RAY_TMPDIR_HOST" ]]; then
    ray_tmpdir_mount=(-m "$RAY_TMPDIR_HOST:/tmp:none:bind,rw")
fi
network_env=()
if [[ -n "$NETWORK_INTERFACE" ]]; then
    network_env=(
        -e "GLOO_SOCKET_IFNAME=$NETWORK_INTERFACE"
        -e "NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE"
    )
fi

exec enroot start --rc "$WORKTREE/scripts/enroot_exec.sh" \
    -e UV_PROJECT_ENVIRONMENT=/opt/verl-uv-final \
    -e UV_CACHE_DIR=/tmp/uv-cache \
    -e XDG_CACHE_HOME=/run/xdg-cache \
    -e FLASHINFER_WORKSPACE_BASE=/run \
    -e RAY_ADDRESS="${RAY_ADDRESS:-}" \
    -e NNODES="${NNODES:-1}" \
    -e NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}" \
    -e TRITON_CACHE_DIR=/var/tmp \
    -e VLLM_DO_NOT_TRACK=1 \
    "${network_env[@]}" \
    -e DRY_RUN="${DRY_RUN:-0}" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e TRAIN_MODEL_PATH=/models/q0 \
    -e ROLLOUT_MODEL_PATH=/models/qwen38 \
    -e ENABLE_MTP="${ENABLE_MTP:-false}" \
    -e ENFORCE_EAGER="${ENFORCE_EAGER:-true}" \
    -e ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}" \
    -e TRAINING_STEPS="${TRAINING_STEPS:-${SMOKE_STEPS:-4}}" \
    -e TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}" \
    -e EXPECTED_TRAIN_ROWS="${EXPECTED_TRAIN_ROWS:-}" \
    -e EXPECTED_VAL_ROWS="${EXPECTED_VAL_ROWS:-}" \
    -e ROLLOUT_N="${ROLLOUT_N:-2}" \
    -e ROLLOUT_TP="${ROLLOUT_TP:-2}" \
    -e MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}" \
    -e MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-64}" \
    -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}" \
    -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}" \
    -e LORA_RANK="${LORA_RANK:-8}" \
    -e LORA_ALPHA="${LORA_ALPHA:-16}" \
    -e LEARNING_RATE="${LEARNING_RATE:-1e-5}" \
    -e PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-8}}" \
    -e USE_KL_LOSS="${USE_KL_LOSS:-true}" \
    -e KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}" \
    -e SAVE_FREQ="${SAVE_FREQ:-${TRAINING_STEPS:-${SMOKE_STEPS:-4}}}" \
    -e TEST_FREQ="${TEST_FREQ:-${TRAINING_STEPS:-${SMOKE_STEPS:-4}}}" \
    -e REWARD_MODE="${REWARD_MODE:-smoke}" \
    -e VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}" \
    -e RESUME_MODE="${RESUME_MODE:-disable}" \
    -e PROJECT_NAME="${PROJECT_NAME:-qwen38_l8_rl_smoke}" \
    -e EXPERIMENT_NAME="${EXPERIMENT_NAME:-fsdp2_lora_adapter_only}" \
    -e TRAIN_FILE=/opt/data/train.parquet \
    -e VAL_FILE=/opt/data/val.parquet \
    -e OUTPUT_DIR=/run/output \
    -m "$WORKTREE:/workspace:none:bind,ro" \
    -m "$TRAIN_MODEL:/models/q0:none:bind,ro" \
    -m "$ROLLOUT_MODEL:/models/qwen38:none:bind,ro" \
    -m "$DATA:/opt/data:none:bind,ro" \
    -m "$RUN_DIR:/run:none:bind,rw" \
    -m "$TRITON_CACHE_DIR_HOST:/var/tmp:none:bind,rw" \
    "${ray_tmpdir_mount[@]}" \
    "$IMAGE" /bin/bash -lc \
    'set -eu; source /opt/verl-uv-final/bin/activate; cd /workspace; bash scripts/run_qwen_l8_verl_smoke.sh'
